# -*- coding: utf-8 -*-
"""
基于先验知识与双支路架构的岩体裂隙场智能反演模型训练脚本

本脚本实现了论文第五章所述的深度学习反演框架，并深度融合了第四章的物理机理。
主要包含：
1. 数据层：支持时序关联的 SequentialFractureDataset，处理 step_001 起始逻辑。
2. 模型层：静态(MLP) + 动态(Transformer) 的双支路特征融合架构。
3. 损失层：集成了 MSE(信息)、SSIM(拓扑)、TV(连续性)、Arch(移动活动区)、Evo(动态演化) 的复合物理损失。
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --- 0. 全局配置与超参数 ---

# 路径配置
DATASET_DIR = "../final_dataset"  # 请根据实际路径修改
OUTPUT_DIR = "trained_models_physics_informed"

# 模型结构参数
STATIC_FEATURES = 11      # 静态岩石力学参数数量
OUTPUT_HEIGHT = 64        # 输出网格高度
OUTPUT_WIDTH = 64         # 输出网格宽度
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 

# 物理模拟参数 (用于第四章活动拱计算)
MODEL_LENGTH_M = 500.0    # 物理模型长度 (米)
MODEL_HEIGHT_M = 150.0    # 物理模型高度 (米)
STEP_DISTANCE_M = 10.0    # 每一步开采距离 (米)

# 训练超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# 损失函数权重 (物理先验的强度)
LAMBDA_SSIM = 0.3         # 结构相似性权重
LAMBDA_TV = 1e-5          # 连续性正则化权重
LAMBDA_ARCH = 0.5         # 移动活动区(第四章理论)权重
LAMBDA_EVO = 0.2          # 动态演化一致性权重

# --- 1. 物理机理先验：损失函数定义 ---

class SSIMLoss(nn.Module):
    """
    结构相似性损失 (SSIM)
    作为拓扑先验，强制模型学习裂隙网络的结构特征（连通性、边缘），而非单纯的像素数值。
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class TVLoss(nn.Module):
    """
    全变分损失 (Total Variation Loss)
    作为连续性机理先验，惩罚图像中的高频噪声，迫使裂隙场在空间上保持物理连续性。
    """
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class ActivityArchPrior(nn.Module):
    """
    移动活动区先验 (Activity Arch Prior)
    基于论文第四章公式(4.3)，生成随开采进度移动的'发育拱'掩膜。
    该掩膜对拱内区域施加更高的权重，迫使模型关注应力集中与裂隙萌生的核心区域。
    """
    def __init__(self, output_size=64):
        super(ActivityArchPrior, self).__init__()
        self.H = output_size
        self.W = output_size
        
        # 网格坐标系预计算
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, self.H), 
            torch.linspace(0, MODEL_LENGTH_M, self.W),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def forward(self, pred, target, mining_distances):
        """
        mining_distances: [Batch] 当前开采距离 (米)
        """
        batch_size = pred.size(0)
        masks = []
        
        # 依据第四章参数设定
        H_max = 100.0  # 最大发育高度
        W_arch = 94.0  # 拱宽
        Beta = 7.5     # 关键层影响指数
        
        for i in range(batch_size):
            d = mining_distances[i]
            
            # 1. 计算活动拱中心 xc (假设滞后工作面一定距离，如20m)
            xc = d - 20.0
            
            # 2. 计算当前发育高度 H(t) - 简化的S型增长
            curr_H = H_max * torch.tanh(d / 100.0)
            
            # 3. 生成拱形边界 y_b (公式 4.3)
            # 归一化横坐标
            x_term = (self.x_grid - xc) / (W_arch + 1e-6)
            
            # 仅在拱宽范围内计算
            in_arch_mask = (x_term.abs() <= 1.0).float()
            
            # 计算边界高度
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), Beta) * in_arch_mask
            
            # 4. 生成空间权重 Mask
            # 拱形下方(裂隙活跃区)权重高，上方权重低
            # 使用 Sigmoid 实现软边界
            spatial_mask = torch.sigmoid((y_boundary - self.y_grid) * 0.5) 
            masks.append(spatial_mask)
            
        masks = torch.stack(masks).unsqueeze(1) # [Batch, 1, H, W]
        
        # 加权 MSE 计算：拱内误差放大 5 倍
        weighted_diff = (pred - target) ** 2
        weighted_loss = weighted_diff * (1.0 + 4.0 * masks)
        
        return weighted_loss.mean()

class EvolutionLoss(nn.Module):
    """
    动态演化一致性损失 (Evolution Consistency Loss)
    基于第四章公式(4.2)，约束模型预测的'增量场'与物理真实的'增量场'一致。
    L_evo = || (Pred_t - GT_{t-1}) - (GT_t - GT_{t-1}) ||^2
    """
    def __init__(self):
        super(EvolutionLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_t, target_t, target_prev):
        # 真实的物理增量
        true_delta = target_t - target_prev
        
        # 预测的增量 (基于上一时刻真实值)
        # 这种 "Teacher Forcing" 策略有助于模型学习局部变化
        pred_delta = pred_t - target_prev
        
        return self.mse(pred_delta, true_delta)

class PhysicsInformedLoss(nn.Module):
    """
    复合物理先验损失函数
    L_total = MSE + λ1*SSIM + λ2*TV + λ3*Arch + λ4*Evo
    """
    def __init__(self):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.tv = TVLoss()
        self.arch_prior = ActivityArchPrior(output_size=OUTPUT_HEIGHT)
        self.evo_loss = EvolutionLoss()

    def forward(self, pred_flat, target_flat, target_prev_flat, mining_dists):
        # 1. 形状变换 [Batch, 4096] -> [Batch, 1, 64, 64]
        pred_img = pred_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_img = target_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_prev_img = target_prev_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        # 2. 计算各项损失
        l_mse = self.mse(pred_flat, target_flat)       # 信息先验
        l_ssim = self.ssim(pred_img, target_img)       # 拓扑先验
        l_tv = self.tv(pred_img)                       # 连续性先验
        l_arch = self.arch_prior(pred_img, target_img, mining_dists) # 第四章移动活动区先验
        l_evo = self.evo_loss(pred_img, target_img, target_prev_img) # 动态演化先验

        # 3. 加权求和
        total_loss = l_mse + \
                     LAMBDA_SSIM * l_ssim + \
                     LAMBDA_TV * l_tv + \
                     LAMBDA_ARCH * l_arch + \
                     LAMBDA_EVO * l_evo
                     
        return total_loss, l_mse, l_ssim, l_arch, l_evo

# --- 2. 数据加载层：支持时序关联与文件名解析 ---

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    """
    时序裂隙数据集
    功能：
    1. 解析文件名中的 sample_id 和 step_id (从001开始)。
    2. 自动查找上一时刻 (step-1) 的数据，用于计算演化损失。
    3. 返回 (x_t, y_t, y_prev, mining_dist)
    """
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.index_map = {} # 快速查找表

        # 建立索引：(sample_id, step_id) -> file_index
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError:
                continue

    def _parse_filename(self, filepath):
        """
        解析格式：...sample_XXXX_step_XXX.npz
        """
        filename = os.path.basename(filepath)
        
        # 提取 Sample ID
        s_pos = filename.rfind("sample")
        if s_pos == -1: raise ValueError
        s_start = s_pos + len("sample") + 1
        s_id = int(filename[s_start : s_start + 4])
        
        # 提取 Step ID
        st_pos = filename.rfind("step")
        if st_pos == -1: raise ValueError
        st_start = st_pos + len("step") + 1
        st_id = int(filename[st_start : st_start + 3])
        
        return s_id, st_id

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        
        # 计算开采距离 (假设 step 1 = 10m)
        mining_dist = st_id * STEP_DISTANCE_M

        # 加载当前时刻 t
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))

        # 加载上一时刻 t-1
        # 如果 step > 1，尝试找上一步；否则(step==1)上一步为全0
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
            else:
                y_prev = y_t.clone() # 缺失回退策略
        else:
            y_prev = torch.zeros_like(y_t)

        if self.transform:
            x_t = self.transform(x_t)

        return x_t, y_t, y_prev, np.float32(mining_dist)

# --- 3. 模型架构层：双支路网络 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DualBranchModel(nn.Module):
    """
    双支路反演模型
    - 静态支路(MLP)：处理地质参数 (机理/条件先验)
    - 动态支路(Transformer)：处理沉降序列 (信息先验)
    """
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model

        # 1. 静态支路 MLP
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 2. 动态支路 Transformer
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 3. 融合与解码头
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
            nn.Sigmoid() # 确保输出在 [0,1] 区间，符合 SSIM 要求
        )

    def forward(self, x):
        # 分离输入
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]

        # 静态支路处理
        static_out = self.static_branch(x_static)

        # 动态支路处理
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        dynamic_out = dynamic_transformed.mean(dim=1) # 全局平均池化

        # 特征融合
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. 训练逻辑 ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    print("\nStarting Physics-Informed Training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # 累加器
        acc_loss = 0.0
        acc_mse = 0.0
        acc_ssim = 0.0
        acc_arch = 0.0
        acc_evo = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets, targets_prev, dists in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_prev = targets_prev.to(device)
            dists = dists.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算复合物理损失
            loss, l_mse, l_ssim, l_arch, l_evo = criterion(outputs, targets, targets_prev, dists)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
            optimizer.step()

            # 记录
            acc_loss += loss.item()
            acc_mse += l_mse.item()
            acc_ssim += l_ssim.item()
            acc_arch += l_arch.item()
            acc_evo += l_evo.item()
            
            progress.set_postfix(
                Loss=f"{loss.item():.4f}", 
                SSIM=f"{l_ssim.item():.3f}",
                Arch=f"{l_arch.item():.3f}"
            )

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, targets_prev, dists in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                targets_prev = targets_prev.to(device)
                dists = dists.to(device)
                
                outputs = model(inputs)
                loss, _, _, _, _ = criterion(outputs, targets, targets_prev, dists)
                val_loss += loss.item()

        # 平均统计
        n_train = len(train_loader)
        n_val = len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={acc_loss/n_train:.4f} "
              f"(MSE={acc_mse/n_train:.4f}, SSIM={acc_ssim/n_train:.3f}, "
              f"Arch={acc_arch/n_train:.3f}, Evo={acc_evo/n_train:.3f}) | "
              f"Val Loss={val_loss/n_val:.4f}")

        # 保存最佳模型
        if (val_loss/n_val) < best_val_loss:
            best_val_loss = val_loss/n_val
            path = os.path.join(OUTPUT_DIR, "best_physics_model.pth")
            torch.save(model.state_dict(), path)
            print(f" -> Best model saved: {path}")

# --- 5. 主程序入口 ---

def main():
    print("========================================================")
    print("   Dual-Branch Physics-Informed Inversion Model         ")
    print("   (Integrated with Chapter 4 Mechanism Priors)         ")
    print("========================================================")

    # 1. 准备数据
    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files:
        print("No .npz files found!")
        return
        
    # 获取特征维度
    with np.load(all_files[0]) as f:
        total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES
    print(f"Input Features: {total_feats} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_feats})")

    # 计算归一化统计量
    print("Calculating normalization stats...")
    temp_dataset = SequentialFractureDataset(all_files) # 使用基础加载器计算均值
    temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, num_workers=0)
    all_x = []
    # 仅需部分数据估算即可，全量计算太慢
    for i, (x, _, _, _) in enumerate(temp_loader):
        all_x.append(x)
        if i > 50: break 
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    print("Normalization stats ready.")

    # 划分数据集
    np.random.shuffle(all_files)
    split = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split], all_files[split:]

    train_ds = SequentialFractureDataset(train_files, transform=transform)
    val_ds = SequentialFractureDataset(val_files, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 初始化模型与损失
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_feats,
        output_size=OUTPUT_FEATURES
    ).to(device)

    criterion = PhysicsInformedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()