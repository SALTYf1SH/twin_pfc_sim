# -*- coding: utf-8 -*-
"""
Robustness Check Script for Subsidence Inversion (Multi-Seed).
Based on: train_subsidence_physics_mamba.py

Changes:
1. Adds --seed argument.
2. Changes OUTPUT_DIR to separate 'robustness_results_subsidence'.
3. Saves models with seed suffix.
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
import argparse
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it to use Mamba architecture.")
    Mamba = None

# --- 0. Configuration ---

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--ablation_name", type=str, default="full", help="Name of ablation study")
parser.add_argument("--branch_mode", type=str, default="dual", choices=["dual", "static_only", "dynamic_only"])
parser.add_argument("--no_physics", action="store_true", help="Disable physics loss")
args, _ = parser.parse_known_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset") 
# Robustness Output Dir
OUTPUT_DIR = os.path.join(BASE_DIR, "../robustness_results_subsidence")

PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

STATIC_FEATURES = 11      
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 

MODEL_LENGTH_M = 500.0    
MODEL_HEIGHT_M = 150.0    
STEP_DISTANCE_M = 10.0    

# Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# Loss Weights
LAMBDA_SSIM = 0.3         
LAMBDA_TV = 1e-5          
LAMBDA_ARCH = 0.0 if args.no_physics else 0.1         
LAMBDA_EVO = 0.0 if args.no_physics else 0.05          

# Scheduler Config
WARMUP_EPOCHS = 10
MAX_LR = 1e-4
MIN_LR = 1e-6

# --- 1. Loss Functions ---

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

class TheoryConsistentLoss(nn.Module):
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentLoss, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        y_vals = torch.linspace(0, 150.0, self.H); x_vals = torch.linspace(0, 500.0, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid); self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        batch_size = dist.size(0); max_d = dist.max().item()
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        h_curr = torch.zeros(batch_size, device=dist.device); h_trace = [h_curr.clone()]
        for _ in range(steps):
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh; h_curr = torch.min(h_curr, h_max)
            h_trace.append(h_curr.clone())
        h_trace = torch.stack(h_trace)
        indices = dist.long().clamp(max=steps)
        h_final = h_trace.gather(0, indices.unsqueeze(0)).squeeze(0)
        return h_final

    def forward(self, pred, target, mining_distances, phys_params):
        h_max = phys_params[:, 0]; w_arch = phys_params[:, 1]; beta = phys_params[:, 2]
        lag = phys_params[:, 3]; k_growth = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]; ks_b = phys_params[:, 7:9]
        curr_H = self.solve_height_ode(mining_distances, h_max, k_growth, ks_h, ks_b)
        
        xc = (mining_distances - lag).view(-1, 1, 1); curr_H = curr_H.view(-1, 1, 1)
        w_arch = w_arch.view(-1, 1, 1); beta_shape = beta.view(-1, 1, 1)
        
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
    def __init__(self): super(EvolutionLoss, self).__init__()
    def forward(self, pred_t, target_t, target_prev):
        true_delta = target_t - target_prev
        pred_delta = pred_t - target_prev
        return F.mse_loss(pred_delta, true_delta) * 100.0

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.tv = TVLoss()
        self.arch_prior = TheoryConsistentLoss(output_size=64)
        self.evo_loss = EvolutionLoss()

    def forward(self, pred_flat, target_flat, target_prev_flat, mining_dists, phys_params):
        target_flat_vec = target_flat.view(target_flat.size(0), -1)
        pred_img = pred_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_img = target_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
        target_prev_img = target_prev_flat.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)

        l_mse = self.mse(pred_flat, target_flat_vec)
        pred_clamped = torch.clamp(pred_img, 0.0, 1.0)
        l_ssim = self.ssim(pred_clamped, target_img)
        l_tv = self.tv(pred_img)
        l_arch = self.arch_prior(pred_img, target_img, mining_dists, phys_params)
        l_evo = self.evo_loss(pred_img, target_img, target_prev_img)

        total_loss = l_mse + LAMBDA_SSIM * l_ssim + LAMBDA_TV * l_tv + LAMBDA_ARCH * l_arch + LAMBDA_EVO * l_evo
        return total_loss, l_mse, l_ssim, l_arch, l_evo

# --- 2. Data Loading ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialSubsidenceDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None, global_index_map=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.global_index_map = global_index_map
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}
        
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
        s_id = int(filename[s_pos+7 : s_pos+11])
        st_id = int(filename[st_pos+5 : st_pos+8])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M

        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T 
            
        if st_id > 1:
            prev_path = None
            if self.global_index_map is not None:
                prev_path = self.global_index_map.get((s_id, st_id - 1))
            if prev_path is None:
                prev_idx = self.index_map.get((s_id, st_id - 1))
                if prev_idx is not None: prev_path = self.file_list[prev_idx]

            if prev_path is not None:
                with np.load(prev_path) as data:
                    y_prev_np = data['y'].astype(np.float32)
                    if y_prev_np.ndim == 1: y_prev_np = y_prev_np.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev_np = y_prev_np.T 
                    y_prev = torch.from_numpy(y_prev_np)
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)

        default_params = {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]}
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 3. Model Architecture (Dual-Branch Mamba) ---

class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, n_layers=2, dropout=0.1): 
        super(DualBranchMambaModel, self).__init__()
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
        
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        else:
            raise ImportError("mamba_ssm is required for this model")

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
            x_dynamic_seq = x_dynamic.unsqueeze(-1)
            x_mamba = self.dynamic_embedder(x_dynamic_seq)
            
            for layer, norm in zip(self.mamba_layers, self.norms):
                residual = x_mamba
                x_mamba = norm(x_mamba)
                x_mamba = layer(x_mamba)
                x_mamba = residual + x_mamba
                
            dynamic_out = x_mamba.mean(dim=1) 
        else:
             dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device) 
        
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. Training ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    print(f"\nStarting Robustness Training (Seed {args.seed})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        acc_loss, acc_mse, acc_ssim, acc_arch, acc_evo = 0.0, 0.0, 0.0, 0.0, 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets, targets_prev, dists, phys_params in progress:
            inputs = inputs.to(device); targets = targets.to(device)
            targets_prev = targets_prev.to(device); dists = dists.to(device)
            phys_params = phys_params.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss, l_mse, l_ssim, l_arch, l_evo = criterion(outputs, targets, targets_prev, dists, phys_params)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
            optimizer.step()

            acc_loss += loss.item(); acc_mse += l_mse.item(); acc_ssim += l_ssim.item()
            acc_arch += l_arch.item(); acc_evo += l_evo.item()

        scheduler.step()
        
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
        if (epoch % 5 == 0) or (epoch == NUM_EPOCHS - 1):
            print(f"Epoch {epoch+1}: Train Loss={acc_loss/n_train:.4f} | Val Loss={val_loss/n_val:.4f}")

        if (val_loss/n_val) < best_val_loss:
            best_val_loss = val_loss/n_val
            # Save with Seed and Ablation info in filename
            suffix = f"{args.ablation_name}_{args.branch_mode}"
            if args.no_physics: suffix += "_no_phys"

            path = os.path.join(OUTPUT_DIR, f"best_subsidence_{suffix}_seed{args.seed}.pth")
            torch.save(model.state_dict(), path)

# --- 5. Main ---

def main():
    print(f"=== Robustness Check: Subsidence Inversion (Seed {args.seed}) ===")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    if not os.path.exists(dataset_path): print(f"Dataset not found at {dataset_path}"); return
    
    all_files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
    if not all_files: print("No .npz files found!"); return
        
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES

    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample"); st_pos = fn.rfind("step")
            s_id = int(fn[s_pos+7 : s_pos+11])
            st_id = int(fn[st_pos+5 : st_pos+8])
            global_index_map[(s_id, st_id)] = fp
        except: continue

    # Calculate Stats or Load
    temp_dataset = SequentialSubsidenceDataset(all_files, PARAMS_JSON_PATH, global_index_map=global_index_map)
    temp_loader = DataLoader(temp_dataset, batch_size=128, num_workers=0)
    all_x = []
    print("Caching dataset...")
    for i, (x, _, _, _, _) in enumerate(tqdm(temp_loader, desc="Scanning")):
        all_x.append(x)
        
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    std[std < 1e-6] = 1.0 
    
    # [FIX] Save normalization stats for this specific run
    stats_path = os.path.join(OUTPUT_DIR, f"subsidence_stats_seed{args.seed}.pt")
    torch.save({'mean': mean, 'std': std}, stats_path)
    print(f"Stats saved to: {stats_path}") 
    
    transform = NormalizeTransform(mean, std)

    np.random.shuffle(all_files)
    split = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split], all_files[split:]

    train_ds = SequentialSubsidenceDataset(train_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    val_ds = SequentialSubsidenceDataset(val_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DualBranchMambaModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_feats,
        output_size=OUTPUT_FEATURES,
        branch_mode=args.branch_mode
    ).to(device)

    criterion = PhysicsInformedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS: return float(epoch + 1) / float(WARMUP_EPOCHS)
        else:
            progress = float(epoch - WARMUP_EPOCHS) / float(NUM_EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

if __name__ == "__main__":
    main()
