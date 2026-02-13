# -*- coding: utf-8 -*-
"""
Baseline Models Evaluation Script (Batch Processing).

Features:
1. Automatically detects and loads baseline models (CNN, LSTM, MAMBA) from a directory.
2. Uses the EXACT SAME dataset and metrics as the main model evaluation.
3. Prints a comparison table for your paper.

Usage:
python new_dl_model/evaluate_baselines.py --dataset_type stress --all
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 请补上这一行
import argparse
import math
from torch.utils.data import Dataset, DataLoader
import skimage

# Import dependencies from existing scripts if possible, or define them here.
# To keep this script standalone, we redefine the necessary classes below.

# --- 依赖库检查 ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM metrics will be skipped.")

# --- 0. 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default Paths (Can be overridden by args)
STRESS_DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")
SUBSIDENCE_DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset")

STRESS_MODEL_DIR = os.path.join(BASE_DIR, "trained_models_baselines_stress")
SUBSIDENCE_MODEL_DIR = os.path.join(BASE_DIR, "trained_models_baselines_subsidence")

# --- 1. Dataset & Transforms (Copy from evaluate_stress_model.py) ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialDataset(Dataset):
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        # We need index map for Evolution Error calculation (prev step)
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
        if s_pos == -1 or st_pos == -1: raise ValueError
        s_id = int(filename[s_pos+7 : s_pos+11])
        st_id = int(filename[st_pos+5 : st_pos+8])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            
        # Load T-1 for Evo Error
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, os.path.basename(curr_path)

# --- 2. Baseline Models (Redefined to load weights) ---

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        
        self.init_channels = 128
        self.init_size = 8
        
        # 1. 投影层
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        
        # 2. 这里的定义必须与训练脚本 train_baselines_stress.py 完全一致
        # 如果您使用的是我之前提供的 train_baselines_stress.py (无 bn1 版本)
        # 或者是带有 bn1 的旧版本？
        
        # 根据报错信息 "Unexpected key(s): bn1.weight"，说明您的 checkpoint 里包含了 bn1
        # 这意味着您训练时用的是旧版结构。
        
        # --- 恢复旧版结构 (与您的 Checkpoint 匹配) ---
        self.bn1 = nn.BatchNorm1d(self.init_channels * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # No sigmoid
        )

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x = self.fc(x)
        x = self.bn1(x) # 恢复这一步
        x = F.relu(x)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        
        # 保持与训练脚本一致的 hidden_dim = 256
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), # 256 * 2 = 512
            nn.ReLU(),
            # [关键修复] 必须补上这个 Dropout，否则层索引对不上
            nn.Dropout(0.2), 
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        
        seq_input = torch.cat([
            x_dynamic.unsqueeze(-1), 
            x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)
        ], dim=2)
        
        lstm_out, _ = self.lstm(seq_input)
        
        # Global Average Pooling
        return self.decoder(torch.mean(lstm_out, dim=1))
class MambaBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, d_model=64, n_layers=1):
        super(MambaBaseline, self).__init__()
        from mamba_ssm import Mamba
        self.dynamic_len = dynamic_len; self.static_len = static_len
        self.embedding = nn.Linear(1 + static_len, d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        x = self.embedding(seq_input).float()
        x = torch.clamp(x, min=-5.0, max=5.0)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            out = layer(x)
            if torch.isnan(out).any() or torch.isinf(out).any(): x = residual
            else: x = residual + out * 0.1
        x = self.final_norm(x)
        x = x.mean(dim=1)
        x = torch.nan_to_num(x, nan=0.0)
        return self.decoder(x)

# --- 3. Evaluation Logic ---

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating Baselines for {args.dataset_type.upper()} Dataset ---")
    
    # 1. Config based on dataset type
    if args.dataset_type == 'stress':
        dataset_dir = STRESS_DATASET_DIR
        model_dir = STRESS_MODEL_DIR
        static_feats = 17
    else:
        dataset_dir = SUBSIDENCE_DATASET_DIR
        model_dir = SUBSIDENCE_MODEL_DIR
        static_feats = 11
        
    all_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if not all_files: print("No data found."); return
    
    # Auto-detect Dynamic Features
    with np.load(all_files[0]) as f: 
        total_dim = f['x'].shape[0]
        dynamic_feats = total_dim - static_feats
    print(f"Input Dim: {total_dim} (Static: {static_feats}, Dynamic: {dynamic_feats})")

    # 2. Data Loader (Validation Split)
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    # Normalization (Train set stats)
    temp_loader = DataLoader(SequentialDataset(all_files[:split_idx][:200]), batch_size=100)
    all_x = [x for x, _, _, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    transform = NormalizeTransform(x_tensor.mean(dim=0), x_tensor.std(dim=0))
    
    val_dataset = SequentialDataset(val_files, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 3. Scan for models
    model_paths = glob.glob(os.path.join(model_dir, "*.pth"))
    print(f"Found {len(model_paths)} models in {model_dir}")
    
    results_table = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n> Evaluating: {model_name}")
        
        # Initialize Model based on filename
        if "CNN" in model_name:
            model = DeepCNNBaseline(input_size=total_dim).to(device)
        elif "LSTM" in model_name:
            model = BiLSTMBaseline(dynamic_len=dynamic_feats, static_len=static_feats).to(device)
        elif "MAMBA" in model_name:
            model = MambaBaseline(dynamic_len=dynamic_feats, static_len=static_feats, d_model=64, n_layers=1).to(device)
        else:
            print(f"Skipping unknown model type: {model_name}")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Metric Accumulators
        all_mse, all_mae, all_ssim, all_evo = [], [], [], []
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_mae = nn.L1Loss(reduction='none')
        
        with torch.no_grad():
            for inputs, targets, targets_prev, _ in tqdm(val_loader, leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                targets_prev = targets_prev.to(device)
                
                outputs = model(inputs)
                
                # Metrics
                all_mse.extend(criterion_mse(outputs, targets).mean(dim=1).cpu().numpy())
                all_mae.extend(criterion_mae(outputs, targets).mean(dim=1).cpu().numpy())
                
                pred_delta = outputs - targets_prev.view(outputs.shape)
                gt_delta = targets - targets_prev.view(targets.shape)
                all_evo.extend(criterion_mse(pred_delta, gt_delta).mean(dim=1).cpu().numpy())
                
                if SKIMAGE_AVAILABLE:
                    pred_imgs = torch.clamp(outputs, 0.0, 1.0).cpu().numpy().reshape(-1, 64, 64)
                    tgt_imgs = targets.cpu().numpy().reshape(-1, 64, 64)
                    for p, t in zip(pred_imgs, tgt_imgs):
                        dr = max(t.max()-t.min(), 1e-6)
                        all_ssim.append(ssim(t, p, data_range=dr))

        # Store Results
        res = {
            "Model": model_name.replace("best_baseline_", "").replace(".pth", ""),
            "MSE": np.mean(all_mse),
            "MAE": np.mean(all_mae),
            "SSIM": np.mean(all_ssim) if all_ssim else 0.0,
            "Evo": np.mean(all_evo)
        }
        results_table.append(res)
        print(f"  MSE: {res['MSE']:.6f} | SSIM: {res['SSIM']:.4f}")

    # 4. Print Summary Table
    print("\n" + "="*65)
    print(f"{'Model':<30} | {'MSE':<10} | {'SSIM':<8} | {'Evo Error':<10}")
    print("-" * 65)
    for res in results_table:
        print(f"{res['Model']:<30} | {res['MSE']:.6f}   | {res['SSIM']:.4f}   | {res['Evo']:.6f}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="stress", choices=["stress", "subsidence"], 
                        help="Choose dataset type (stress/subsidence)")
    args = parser.parse_args()
    evaluate(args)