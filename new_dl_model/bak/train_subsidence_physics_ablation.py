# -*- coding: utf-8 -*-
"""
Physics-Informed Subsidence Model Training Script (Ablation Ready + CSV Logging)

本脚本专为沉降反演模型设计，支持全套消融实验配置。

Usage Examples:
1. 训练完整模型: python train_subsidence_physics_ablation.py --ablation full --branch_mode dual
2. 训练无活动拱: python train_subsidence_physics_ablation.py --ablation no_arch
...
"""

import os
import glob
import json
import argparse
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --- 0. 基础配置 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset") 
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_subsidence_ablation") # 输出目录
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_log.csv") 

STATIC_FEATURES = 11      # 沉降数据集特征数
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 

MODEL_LENGTH_M = 500.0    
MODEL_HEIGHT_M = 150.0    
STEP_DISTANCE_M = 10.0    

# 训练超参
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# 默认物理权重
DEFAULT_LAMBDA_SSIM = 0.3         
DEFAULT_LAMBDA_TV = 1e-5          
DEFAULT_LAMBDA_ARCH = 0.5         
DEFAULT_LAMBDA_EVO = 0.2          

# --- 1. 物理先验损失定义 ---

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

class ActivityArchPrior(nn.Module):
    def __init__(self, output_size=64):
        super(ActivityArchPrior, self).__init__()
        self.H = output_size
        self.W = output_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, self.H), 
            torch.linspace(0, MODEL_LENGTH_M, self.W),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def forward(self, pred, target, mining_distances, phys_params):
        batch_size = pred.size(0)
        masks = []
        for i in range(batch_size):
            d = mining_distances[i]
            h_max = phys_params[i, 0]; w_arch = phys_params[i, 1]
            beta = phys_params[i, 2]; lag = phys_params[i, 3]
            
            xc = d - lag 
            curr_H = h_max * torch.tanh(d / 100.0)
            x_term = (self.x_grid - xc) / (w_arch + 1e-6)
            in_arch_mask = (x_term.abs() <= 1.0).float()
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * in_arch_mask
            spatial_mask = torch.sigmoid((y_boundary - self.y_grid) * 0.5) 
            masks.append(spatial_mask)
            
        masks = torch.stack(masks).unsqueeze(1)
        weighted_diff = (pred - target) ** 2
        # 仅在 Arch 权重 > 0 时计算加权
        weighted_loss = weighted_diff * (1.0 + 4.0 * masks)
        return weighted_loss.mean()

class EvolutionLoss(nn.Module):
    def __init__(self): super(EvolutionLoss, self).__init__(); self.mse = nn.MSELoss()
    def forward(self, pred_t, target_t, target_prev):
        true_delta = target_t - target_prev
        pred_delta = pred_t - target_prev
        return self.mse(pred_delta, true_delta)

class PhysicsInformedLoss(nn.Module):
    """
    支持动态权重配置的损失函数，用于消融实验。
    """
    def __init__(self, weights):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.tv = TVLoss()
        self.arch_prior = ActivityArchPrior(output_size=OUTPUT_HEIGHT)
        self.evo_loss = EvolutionLoss()
        
        # 从字典中读取权重
        self.w_ssim = weights['ssim']
        self.w_tv = weights['tv']
        self.w_arch = weights['arch']
        self.w_evo = weights['evo']

    def forward(self, pred_flat, target_flat, target_prev_flat, mining_dists, phys_params):
        pred_img = pred_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_img = target_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_prev_img = target_prev_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        l_mse = self.mse(pred_flat, target_flat)
        
        l_ssim = torch.tensor(0.0, device=pred_flat.device)
        if self.w_ssim > 0:
            pred_clamped = torch.clamp(pred_img, 0.0, 1.0)
            l_ssim = self.ssim(pred_clamped, target_img)

        l_tv = torch.tensor(0.0, device=pred_flat.device)
        if self.w_tv > 0:
            l_tv = self.tv(pred_img)
            
        l_arch = torch.tensor(0.0, device=pred_flat.device)
        if self.w_arch > 0:
            l_arch = self.arch_prior(pred_img, target_img, mining_dists, phys_params)
            
        l_evo = torch.tensor(0.0, device=pred_flat.device)
        if self.w_evo > 0:
            l_evo = self.evo_loss(pred_img, target_img, target_prev_img)

        total_loss = l_mse + \
                     self.w_ssim * l_ssim + \
                     self.w_tv * l_tv + \
                     self.w_arch * l_arch + \
                     self.w_evo * l_evo
        
        return total_loss, l_mse, l_ssim, l_arch, l_evo

# --- 2. 数据加载 ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f:
                self.physics_params = json.load(f)
        else:
            self.physics_params = {}
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); s_start = s_pos + 7; s_id = int(filename[s_start : s_start + 4])
        st_pos = filename.rfind("step"); st_start = st_pos + 5; st_id = int(filename[st_start : st_start + 3])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data: y_prev = torch.from_numpy(data['y'].astype(np.float32))
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)
        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 3. 模型架构 (支持支路消融) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)

class DualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode 

        # Static Branch
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        # Dynamic Branch
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Fusion Head (No Sigmoid for Linear Output)
        # 关键修正：确保 fusion_input_size 定义正确
        fusion_input_size = 32 + d_model 
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        # --- 架构消融逻辑 ---
        
        # 1. 静态支路处理
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            # 屏蔽静态支路
            static_out = torch.zeros(x.size(0), 32, device=x.device)

        # 2. 动态支路处理
        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic = x_dynamic.unsqueeze(-1)
            dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
            dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
            dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
            dynamic_out = dynamic_transformed.mean(dim=1)
        else:
            # 屏蔽动态支路
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. 训练循环 (带返回值) ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device, save_name):
    print(f"\nStarting Training -> {save_name}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        acc_loss, acc_mse = 0.0, 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets, targets_prev, dists, phys_params in progress:
            inputs = inputs.to(device); targets = targets.to(device)
            targets_prev = targets_prev.to(device); dists = dists.to(device)
            phys_params = phys_params.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss, l_mse, _, _, _ = criterion(outputs, targets, targets_prev, dists, phys_params)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
            optimizer.step()

            acc_loss += loss.item(); acc_mse += l_mse.item()
            progress.set_postfix(Loss=f"{loss.item():.4f}", MSE=f"{l_mse.item():.4f}")

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

        n_val = len(val_loader)
        if (val_loss/n_val) < best_val_loss:
            best_val_loss = val_loss/n_val
            path = os.path.join(OUTPUT_DIR, save_name)
            torch.save(model.state_dict(), path)
            
    print(f"Training Finished. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss

# --- 5. 主程序 ---

def main():
    parser = argparse.ArgumentParser(description="Subsidence Model Ablation Training")
    
    parser.add_argument("--ablation", type=str, default="full", 
                        choices=["full", "baseline", "no_ssim", "no_arch", "no_evo", "no_tv"],
                        help="Loss function configuration.")
    
    parser.add_argument("--branch_mode", type=str, default="dual",
                        choices=["dual", "static_only", "dynamic_only"],
                        help="Network architecture configuration.")
    
    args = parser.parse_args()

    # 1. 配置权重
    weights = {
        'ssim': DEFAULT_LAMBDA_SSIM,
        'tv': DEFAULT_LAMBDA_TV,
        'arch': DEFAULT_LAMBDA_ARCH,
        'evo': DEFAULT_LAMBDA_EVO
    }

    if args.ablation == 'baseline':
        weights = {'ssim': 0.0, 'tv': 0.0, 'arch': 0.0, 'evo': 0.0}
    elif args.ablation == 'no_ssim':
        weights['ssim'] = 0.0
    elif args.ablation == 'no_arch':
        weights['arch'] = 0.0
    elif args.ablation == 'no_evo':
        weights['evo'] = 0.0
    elif args.ablation == 'no_tv':
        weights['tv'] = 0.0
    
    save_name = f"best_subsidence_{args.ablation}_{args.branch_mode}.pth"

    print("========================================================")
    print(f"   Physics-Informed SUBSIDENCE Ablation Training       ")
    print(f"   Loss Config : {args.ablation}")
    print(f"   Branch Mode : {args.branch_mode}")
    print(f"   Save File   : {save_name}")
    print("========================================================")

    # 2. 数据准备
    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No files found"); return
        
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES

    print("Calculating normalization stats...")
    temp_dataset = SequentialFractureDataset(all_files, PARAMS_JSON_PATH)
    temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, num_workers=0)
    all_x = []
    for i, (x, _, _, _, _) in enumerate(temp_loader):
        all_x.append(x); 
        if i > 50: break 
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0); std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)

    np.random.shuffle(all_files)
    split = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split], all_files[split:]

    train_ds = SequentialFractureDataset(train_files, PARAMS_JSON_PATH, transform=transform)
    val_ds = SequentialFractureDataset(val_files, PARAMS_JSON_PATH, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_feats,
        output_size=OUTPUT_FEATURES,
        branch_mode=args.branch_mode 
    ).to(device)

    criterion = PhysicsInformedLoss(weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练
    best_loss = train_model(model, train_loader, val_loader, criterion, optimizer, device, save_name)

    # 5. 记录结果到 CSV
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = [
        timestamp,
        args.ablation,
        args.branch_mode,
        f"{best_loss:.6f}",
        save_name
    ]
    
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ablation_Mode", "Branch_Mode", "Best_Val_Loss", "Model_File"])
        writer.writerow(log_data)
    
    print(f"Logged to {LOG_FILE}")

if __name__ == "__main__":
    main()