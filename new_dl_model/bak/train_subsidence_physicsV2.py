# -*- coding: utf-8 -*-
"""
Main script for training the PHYSICS-INFORMED Subsidence-Based Surrogate Model.
(Integrated with Pre-calculated Physics Parameters)

Key Improvements:
1. Loads 'subsidence_physics_params.json' to apply condition-specific mechanism priors.
2. Uses Linear Output (No Sigmoid) for better convergence and clean background.
3. Incorporates 5-part Physics Loss: MSE + SSIM + TV + Dynamic Arch + Evolution.
4. FIXED: Dataset loading logic (Correct shapes for x and y).
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --- 0. Configuration & Hyperparameters ---

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset") 
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_subsidence_physics")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")

# Model Hyperparameters
STATIC_FEATURES = 11      # Subsidence dataset usually has 11 static features
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 

# Physics Simulation Constants
MODEL_LENGTH_M = 500.0    
MODEL_HEIGHT_M = 150.0    
STEP_DISTANCE_M = 10.0    

# Training Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# Loss Weights
LAMBDA_SSIM = 0.3         
LAMBDA_TV = 1e-5          
LAMBDA_ARCH = 0.5         # Weight for the Dynamic Arch Prior
LAMBDA_EVO = 0.2          

# --- 1. Physics Mechanism Priors: Loss Functions ---

class SSIMLoss(nn.Module):
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
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2; C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda: window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class TVLoss(nn.Module):
    def __init__(self): super(TVLoss, self).__init__()
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]; w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    def _tensor_size(self, t): return t.size()[1] * t.size()[2] * t.size()[3]

# --- [关键修改] 新的物理 Loss 模块 (内嵌 ODE) ---

class TheoryConsistentLoss(nn.Module):
    """
    基于关键层动力学 (KS-Dynamics) 的物理约束 Loss。
    替代原有的 ActivityArchPrior。
    """
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentLoss, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        
        # 物理坐标网格 (Y: 0-150m, X: 0-500m)
        y_vals = torch.linspace(0, 150.0, self.H) 
        x_vals = torch.linspace(0, 500.0, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        """数值积分求解 dH/dx (支持 Batch)"""
        batch_size = dist.size(0)
        max_d = dist.max().item()
        # 至少保证有 steps
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        
        h_curr = torch.zeros(batch_size, device=dist.device)
        h_trace = [h_curr.clone()]
        
        for _ in range(steps):
            # KS 阻滞项
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            # dH update
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh
            h_curr = torch.min(h_curr, h_max)
            h_trace.append(h_curr.clone())
            
        h_trace = torch.stack(h_trace)
        indices = dist.long().clamp(max=steps)
        # Gather correct height for each sample
        h_final = h_trace.gather(0, indices.unsqueeze(0)).squeeze(0)
        return h_final

    def forward(self, pred, target, mining_distances, phys_params):
        """
        pred, target: [Batch, 1, 64, 64]
        phys_params: [Batch, 9] -> h, w, b, l, k, ks_h1, ks_h2, ks_b1, ks_b2
        """
        # 1. 解包参数
        h_max = phys_params[:, 0]
        w_arch = phys_params[:, 1]
        beta = phys_params[:, 2]
        lag = phys_params[:, 3]
        k_growth = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]
        ks_b = phys_params[:, 7:9]
        
        # 2. 求解 ODE 得到当前物理高度 H_curr
        curr_H = self.solve_height_ode(mining_distances, h_max, k_growth, ks_h, ks_b)
        
        # 3. 生成 Mask (累积逻辑)
        xc = (mining_distances - lag).view(-1, 1, 1)
        curr_H = curr_H.view(-1, 1, 1)
        w_arch = w_arch.view(-1, 1, 1)
        beta_shape = beta.view(-1, 1, 1)
        
        # 相对坐标
        x_term = (self.xx.unsqueeze(0) - xc) / (w_arch + 1e-6)
        
        # 前方曲线
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta_shape)
        
        # 区域定义
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        is_front = (self.xx.unsqueeze(0) > xc)
        
        # 组合高度场
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(len(mining_distances), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        # 生成 Sigmoid Soft Mask
        y_diff = height_limit - self.yy.unsqueeze(0)
        spatial_mask = torch.sigmoid(y_diff * 0.5).unsqueeze(1) # [B, 1, 64, 64]
        
        # 4. 计算加权 Loss
        diff = (pred - target) ** 2
        
        # 策略 B: 惩罚 Mask 外部的“泄漏”
        weighted_loss = diff * (1.0 + 5.0 * spatial_mask) 
        
        return weighted_loss.mean()

class EvolutionLoss(nn.Module):
    def __init__(self): super(EvolutionLoss, self).__init__(); self.mse = nn.MSELoss()
    def forward(self, pred_t, target_t, target_prev):
        true_delta = target_t - target_prev
        pred_delta = pred_t - target_prev
        return self.mse(pred_delta, true_delta)

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.tv = TVLoss()
        # 确保这里使用的是 TheoryConsistentLoss
        self.arch_prior = TheoryConsistentLoss(output_size=64)
        self.evo_loss = EvolutionLoss()

    def forward(self, pred_raw, target_raw, target_prev_raw, mining_dists, phys_params):
        """
        pred_raw: Model output [Batch, 4096]
        target_raw: GT from Dataset [Batch, 64, 64]
        """
        
        # --- [关键修复] 统一维度 ---
        
        # 1. 为 MSE 准备：全部展平为 [Batch, 4096]
        # pred_raw 已经是扁平的，只需要把 target 展平
        target_flat = target_raw.view(target_raw.size(0), -1)
        
        # 2. 为 SSIM/TV/Arch 准备：全部变为图像 [Batch, 1, 64, 64]
        pred_img = pred_raw.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_img = target_raw.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_prev_img = target_prev_raw.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        # --- 计算各项 Loss ---

        # 1. 信息先验 (MSE) - 使用展平后的 Tensor
        l_mse = self.mse(pred_raw, target_flat)

        # 2. 拓扑先验 (SSIM) - 使用图像 Tensor
        # 手动截断以适配 SSIM 计算，但不截断 MSE 梯度
        pred_clamped = torch.clamp(pred_img, 0.0, 1.0)
        l_ssim = self.ssim(pred_clamped, target_img)

        # 3. 其他机理先验 - 使用图像 Tensor
        l_tv = self.tv(pred_img)
        
        # 传入 phys_params 实现 Condition-Specific 约束
        l_arch = self.arch_prior(pred_img, target_img, mining_dists, phys_params)
        
        l_evo = self.evo_loss(pred_img, target_img, target_prev_img)

        # 总 Loss
        total_loss = l_mse + \
                     LAMBDA_SSIM * l_ssim + \
                     LAMBDA_TV * l_tv + \
                     LAMBDA_ARCH * l_arch + \
                     LAMBDA_EVO * l_evo
        
        return total_loss, l_mse, l_ssim, l_arch, l_evo

# --- 2. Data Loading with Params Integration ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        
        # 1. 加载反算的物理参数库
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f:
                self.physics_params = json.load(f)
            print(f"Loaded physics params for {len(self.physics_params)} samples.")
        else:
            print(f"WARNING: Physics params not found at {params_json_path}. Using defaults.")
            self.physics_params = {}

        # 2. 构建索引
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample")
        if s_pos == -1: raise ValueError
        s_start = s_pos + len("sample") + 1
        s_id = int(filename[s_start : s_start + 4])
        st_pos = filename.rfind("step")
        if st_pos == -1: raise ValueError
        st_start = st_pos + len("step") + 1
        st_id = int(filename[st_start : st_start + 3])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M

        # 加载数据
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            
            # --- [FIXED] 维度修正逻辑 ---
            # 1. x_t (Input Features): 保持 1D，不要 Reshape，不要 Transpose
            # 2. y_t (Target Image): 如果是扁平的(4096)，Reshape成(64,64)，并且 Transpose (.T)
            
            if y_t.ndim == 1: 
                y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            
            # 关键：转置标签以匹配物理 Mask 的方向 (Height, Length)
            y_t = y_t.T
            
            x_t = torch.from_numpy(x_t) if not isinstance(x_t, torch.Tensor) else x_t
            y_t = torch.from_numpy(y_t) if not isinstance(y_t, torch.Tensor) else y_t

        # 加载上一帧 (同样逻辑)
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev_np = data['y'].astype(np.float32)
                    
                    if y_prev_np.ndim == 1: 
                        y_prev_np = y_prev_np.reshape(IMG_SIZE, IMG_SIZE)
                    
                    y_prev_np = y_prev_np.T # Transpose
                    y_prev = torch.from_numpy(y_prev_np)
            else: 
                y_prev = y_t.clone()
        else:
            y_prev = torch.zeros_like(y_t)

        # 获取当前工况的专属物理参数 (9 parameters)
        default_params = {
            "h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0,
            "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]
        }
        
        params = self.physics_params.get(str(s_id), default_params)
        
        # Flatten param list
        p_list = [
            params['h_max'], params['width'], params['beta'], params['lag'],
            params['k_growth']
        ]
        p_list.extend(params['ks_heights'])
        p_list.extend(params['ks_betas'])
        
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 3. Model Architecture (Linear Output) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # [修正点] 这里原来写成了 self.pe = ... 导致下面报错
        # 必须先定义为局部变量 pe
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 现在 pe 是已定义的局部变量，可以赋值了
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # 最后再注册为模型的 buffer (不作为参数更新，但随模型保存)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, Features]
        # 截取对应长度的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model

        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Fusion Head (无 Sigmoid，纯线性输出)
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        dynamic_out = dynamic_transformed.mean(dim=1)
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. Training Logic ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    print("\nStarting Physics-Informed Subsidence Training...")
    print(f"Params Source: {PARAMS_JSON_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        acc_loss, acc_mse, acc_ssim, acc_arch, acc_evo = 0.0, 0.0, 0.0, 0.0, 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        # 注意：DataLoader 现在返回 5 个值
        for inputs, targets, targets_prev, dists, phys_params in progress:
            inputs = inputs.to(device); targets = targets.to(device)
            targets_prev = targets_prev.to(device); dists = dists.to(device)
            phys_params = phys_params.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 传入 phys_params 计算 Loss
            loss, l_mse, l_ssim, l_arch, l_evo = criterion(outputs, targets, targets_prev, dists, phys_params)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
            optimizer.step()

            acc_loss += loss.item(); acc_mse += l_mse.item(); acc_ssim += l_ssim.item()
            acc_arch += l_arch.item(); acc_evo += l_evo.item()
            progress.set_postfix(Loss=f"{loss.item():.4f}", Arch=f"{l_arch.item():.3f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, targets_prev, dists, phys_params in val_loader:
                inputs = inputs.to(device); targets = targets.to(device)
                targets_prev = targets_prev.to(device); dists = dists.to(device)
                phys_params = phys_params.to(device)
                
                outputs = model(inputs)
                loss, _, _, _, _ = criterion(outputs, targets, targets_prev, dists, phys_params)
                val_loss += loss.item()

        n_train = len(train_loader); n_val = len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={acc_loss/n_train:.4f} "
              f"(MSE={acc_mse/n_train:.4f}, SSIM={acc_ssim/n_train:.3f}, "
              f"Arch={acc_arch/n_train:.3f}) | Val Loss={val_loss/n_val:.4f}")

        if (val_loss/n_val) < best_val_loss:
            best_val_loss = val_loss/n_val
            path = os.path.join(OUTPUT_DIR, "best_subsidence_physics_model.pth")
            torch.save(model.state_dict(), path)
            print(f" -> Best model saved: {path}")

# --- 5. Main Execution ---

def main():
    print("========================================================")
    print("   Physics-Informed Subsidence Inversion Model          ")
    print("========================================================")

    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}"); return

    if not os.path.exists(PARAMS_JSON_PATH):
        print(f"WARNING: JSON Params not found at {PARAMS_JSON_PATH}")
        print("Please run 'fit_subsidence_params_ks.py' first!")
        return

    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No .npz files found!"); return
        
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES
    print(f"Input Features: {total_feats} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_feats})")

    print("Calculating normalization stats...")
    # 注意：为了计算 stats，我们需要一个不依赖 JSON 的简单加载器，或者给 Dataset 传空 JSON
    # 这里简单起见，直接实例化带 JSON 的加载器，不影响 x 的加载
    temp_dataset = SequentialFractureDataset(all_files, PARAMS_JSON_PATH)
    temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, num_workers=0)
    all_x = []
    for i, (x, _, _, _, _) in enumerate(temp_loader):
        all_x.append(x); 
        if i > 50: break 
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0); std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    print("Stats ready.")

    np.random.shuffle(all_files)
    split = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split], all_files[split:]

    # 初始化带参数的 Dataset
    train_ds = SequentialFractureDataset(train_files, PARAMS_JSON_PATH, transform=transform)
    val_ds = SequentialFractureDataset(val_files, PARAMS_JSON_PATH, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_feats,
        output_size=OUTPUT_FEATURES
    ).to(device)

    criterion = PhysicsInformedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()