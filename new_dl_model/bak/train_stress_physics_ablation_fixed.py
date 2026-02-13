# -*- coding: utf-8 -*-
"""
Physics-Informed Stress Model Training Script (Ablation Ready + CSV Logging)
Fixed with KS-Dynamics and Correct Dataset Logic.

Usage Examples:
1. Full Model: python train_stress_physics_ablation.py --ablation full --branch_mode dual
2. No Arch:    python train_stress_physics_ablation.py --ablation no_arch
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
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress") 
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_stress_ablation") 
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_log.csv")

STATIC_FEATURES = 17      
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64             # [Fix] Added definition
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

# --- [Fix] TheoryConsistentLoss (ODE-based) ---

class TheoryConsistentLoss(nn.Module):
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentLoss, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        
        y_vals = torch.linspace(0, 150.0, self.H) 
        x_vals = torch.linspace(0, 500.0, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        batch_size = dist.size(0)
        max_d = dist.max().item()
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        
        h_curr = torch.zeros(batch_size, device=dist.device)
        h_trace = [h_curr.clone()]
        
        for _ in range(steps):
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh
            h_curr = torch.min(h_curr, h_max)
            h_trace.append(h_curr.clone())
            
        h_trace = torch.stack(h_trace)
        indices = dist.long().clamp(max=steps)
        h_final = h_trace.gather(0, indices.unsqueeze(0)).squeeze(0)
        return h_final

    def forward(self, pred, target, mining_distances, phys_params):
        h_max = phys_params[:, 0]
        w_arch = phys_params[:, 1]
        beta = phys_params[:, 2]
        lag = phys_params[:, 3]
        k_growth = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]
        ks_b = phys_params[:, 7:9]
        
        curr_H = self.solve_height_ode(mining_distances, h_max, k_growth, ks_h, ks_b)
        
        xc = (mining_distances - lag).view(-1, 1, 1)
        curr_H = curr_H.view(-1, 1, 1)
        w_arch = w_arch.view(-1, 1, 1)
        beta_shape = beta.view(-1, 1, 1)
        
        x_term = (self.xx.unsqueeze(0) - xc) / (w_arch + 1e-6)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta_shape)
        
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        is_front = (self.xx.unsqueeze(0) > xc)
        
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(len(mining_distances), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        y_diff = height_limit - self.yy.unsqueeze(0)
        spatial_mask = torch.sigmoid(y_diff * 0.5).unsqueeze(1) 
        
        diff = (pred - target) ** 2
        weighted_loss = diff * (1.0 + 5.0 * spatial_mask) 
        
        return weighted_loss.mean()

class EvolutionLoss(nn.Module):
    def __init__(self): super(EvolutionLoss, self).__init__(); self.mse = nn.MSELoss()
    def forward(self, pred_t, target_t, target_prev):
        true_delta = target_t - target_prev
        pred_delta = pred_t - target_prev
        return self.mse(pred_delta, true_delta)

class PhysicsInformedLoss(nn.Module):
    def __init__(self, weights):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.tv = TVLoss()
        self.arch_prior = TheoryConsistentLoss(output_size=OUTPUT_HEIGHT)
        self.evo_loss = EvolutionLoss()
        
        self.w_ssim = weights['ssim']
        self.w_tv = weights['tv']
        self.w_arch = weights['arch']
        self.w_evo = weights['evo']

    def forward(self, pred_flat, target_flat, target_prev_flat, mining_dists, phys_params):
        # [Fix] Reshape Logic for MSE and Image Losses
        target_flat_vec = target_flat.view(target_flat.size(0), -1)
        
        pred_img = pred_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_img = target_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_prev_img = target_prev_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        # 1. MSE
        l_mse = self.mse(pred_flat, target_flat_vec)
        
        # 2. SSIM
        l_ssim = torch.tensor(0.0, device=pred_flat.device)
        if self.w_ssim > 0:
            pred_clamped = torch.clamp(pred_img, 0.0, 1.0)
            l_ssim = self.ssim(pred_clamped, target_img)

        # 3. TV
        l_tv = torch.tensor(0.0, device=pred_flat.device)
        if self.w_tv > 0:
            l_tv = self.tv(pred_img)
            
        # 4. Arch
        l_arch = torch.tensor(0.0, device=pred_flat.device)
        if self.w_arch > 0:
            l_arch = self.arch_prior(pred_img, target_img, mining_dists, phys_params)
            
        # 5. Evo
        l_evo = torch.tensor(0.0, device=pred_flat.device)
        if self.w_evo > 0:
            l_evo = self.evo_loss(pred_img, target_img, target_prev_img)

        total_loss = l_mse + \
                     self.w_ssim * l_ssim + \
                     self.w_tv * l_tv + \
                     self.w_arch * l_arch + \
                     self.w_evo * l_evo
        
        return total_loss, l_mse, l_ssim, l_arch, l_evo

# --- 2. Data Loading (Corrected) ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
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
            
            # [Fix] Correct Reshape & Transpose Logic
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T # Align with Physics Coordinates
            
            x_t = torch.from_numpy(x_t) if not isinstance(x_t, torch.Tensor) else x_t
            y_t = torch.from_numpy(y_t) if not isinstance(y_t, torch.Tensor) else y_t

        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev_np = data['y'].astype(np.float32)
                    if y_prev_np.ndim == 1: y_prev_np = y_prev_np.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev_np = y_prev_np.T 
                    y_prev = torch.from_numpy(y_prev_np)
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)
        
        # Load 9 Physics Params (including Key Strata)
        default_params = {
            "h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0,
            "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]
        }
        params = self.physics_params.get(str(s_id), default_params)
        
        p_list = [
            params['h_max'], params['width'], params['beta'], params['lag'],
            params['k_growth']
        ]
        p_list.extend(params['ks_heights'])
        p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)
        
        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 3. 模型架构 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # [Fix] Define local var first
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
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode 

        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            static_out = torch.zeros(x.size(0), 32, device=x.device)

        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic = x_dynamic.unsqueeze(-1)
            dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
            dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
            dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
            dynamic_out = dynamic_transformed.mean(dim=1)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. 训练循环 ---

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
    parser = argparse.ArgumentParser(description="Stress Model Ablation Training")
    
    parser.add_argument("--ablation", type=str, default="full", 
                        choices=["full", "baseline", "no_ssim", "no_arch", "no_evo", "no_tv"],
                        help="Choose the set of losses to enable.")
    
    parser.add_argument("--branch_mode", type=str, default="dual",
                        choices=["dual", "static_only", "dynamic_only"],
                        help="Choose model architecture configuration.")
    
    args = parser.parse_args()

    print("========================================================")
    print(f"   Physics-Informed Stress Model Training (Ablation)   ")
    print(f"   Loss Mode   : {args.ablation}")
    print(f"   Branch Mode : {args.branch_mode}")
    print("========================================================")

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
    
    save_name = f"best_stress_{args.ablation}_{args.branch_mode}.pth"

    # 2. 数据准备
    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No files found"); return
        
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES

    print("Calculating normalization stats...")
    temp_dataset = SequentialStressDataset(all_files, PARAMS_JSON_PATH)
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

    train_ds = SequentialStressDataset(train_files, PARAMS_JSON_PATH, transform=transform)
    val_ds = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform=transform)
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

    # 4. 开始训练
    best_loss = train_model(model, train_loader, val_loader, criterion, optimizer, device, save_name)

    # 5. 记录结果
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = [timestamp, args.ablation, args.branch_mode, f"{best_loss:.6f}", save_name]
    
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ablation_Mode", "Branch_Mode", "Best_Val_Loss", "Model_File"])
        writer.writerow(log_data)
    
    print(f"Experiment results logged to {LOG_FILE}")

if __name__ == "__main__":
    main()