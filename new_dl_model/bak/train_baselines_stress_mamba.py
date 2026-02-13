# -*- coding: utf-8 -*-
"""
Baseline Training Script with MAMBA (Stability Fixed) for STRESS-Based Inversion.

Specific Configuration for Stress Dataset:
- STATIC_FEATURES = 17
- Dataset Path: ../final_dataset_stress

Stability Fixes Applied:
1. Learning Rate = 1e-5
2. Mamba Model: Pre-Norm structure + Float32 enforcement + Residual scaling
3. Training: Gradient Clipping (0.5) + NaN skipping
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

# --- 0. Configuration ---

# >>> MODIFY THIS TO SWITCH MODELS <<<
# Options: "MAMBA", "CNN", "LSTM"
MODEL_TYPE = "MAMBA" 

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")  # Stress Data
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_baselines_stress")
# Params json is not used for training logic here
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

# Hyperparameters
STATIC_FEATURES = 17      # Stress dataset has 17 static features
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
OUTPUT_SIZE = 4096

BATCH_SIZE = 32
# === Stability Fix 1: Lower Learning Rate ===
LEARNING_RATE = 1e-5  
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset & Utilities ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.physics_params = {} 

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t

def calculate_ssim(img1, img2):
    C1 = 0.01**2; C2 = 0.03**2
    mu1 = F.avg_pool2d(img1, 3, 1, 1); mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1_sq = F.avg_pool2d(img1**2, 3, 1, 1) - mu1**2
    sigma2_sq = F.avg_pool2d(img2**2, 3, 1, 1) - mu2**2
    sigma12 = F.avg_pool2d(img1*img2, 3, 1, 1) - mu1*mu2
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2))/((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# --- 2. Baseline Model Architectures ---

# === Stability Fix 2: Armored Mamba Baseline ===
class MambaBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, d_model=128, n_layers=2):
        super(MambaBaseline, self).__init__()
        
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError("Mamba not found. Please install mamba-ssm.")

        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.input_dim = 1 + static_len
        
        self.embedding = nn.Linear(self.input_dim, d_model)
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, 
                d_state=16,      
                d_conv=4,        
                expand=2,        
            ) for _ in range(n_layers)
        ])
        
        # Pre-Norm Structure
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Dropout(0.1), # Added Dropout
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        # x: [Batch, Total_Feats]
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        
        seq_dynamic = x_dynamic.unsqueeze(-1) 
        seq_static = x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1) 
        seq_input = torch.cat([seq_dynamic, seq_static], dim=2) 
        
        # Enforce Float32 and Clamp input
        x = self.embedding(seq_input).float()
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # Pre-Norm Block
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            out = layer(x)
            
            # Check for NaNs
            if torch.isnan(out).any() or torch.isinf(out).any():
                x = residual
            else:
                x = residual + out * 0.1 # Scaled Residual
            
        x = self.final_norm(x)
        x = x.mean(dim=1)
        
        # Safety clamp before decoder
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        return self.decoder(x)

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128
        self.init_size = 8
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

# --- 3. Training Engine ---

def train_baseline(model, train_loader, val_loader):
    print(f"\n>>> Training Stress Baseline: {MODEL_TYPE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_ssim = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for x, y in progress:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- Stability Fix 3: Input Check ---
            if torch.isnan(x).any() or torch.isinf(x).any():
                continue

            optimizer.zero_grad()
            pred = model(x)
            
            if torch.isnan(pred).any():
                continue

            loss = criterion(pred, y)
            
            # --- Stability Fix 4: Loss Check ---
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            
            # --- Stability Fix 5: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_mse, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_mse += criterion(pred, y).item()
                
                pred_img = pred.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                y_img = y.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                val_ssim += calculate_ssim(torch.clamp(pred_img, 0, 1), y_img)

        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train MSE={avg_train:.5f} | Val MSE={avg_val_mse:.5f} | Val SSIM={avg_val_ssim:.4f}")
        
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_baseline_stress_{MODEL_TYPE}.pth"))
            print(f"  [*] Best Model Saved: {avg_val_ssim:.4f}")

# --- 4. Main Execution ---

def main():
    dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
    if not os.path.exists(dataset_path): print("No stress data found!"); return
    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No .npz files found!"); return
    
    # Auto-detect dimensions
    with np.load(all_files[0]) as f: real_input_dim = f['x'].shape[0]
    real_dynamic_features = real_input_dim - STATIC_FEATURES
    print(f"Auto-detected Input Dim: {real_input_dim} (Dyn: {real_dynamic_features}, Stat: {STATIC_FEATURES})")

    # Normalization with Protection
    print("Computing normalization stats...")
    temp_loader = DataLoader(SequentialStressDataset(all_files[:100], None), batch_size=32)
    all_x = [x for x, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    
    # --- Stability Fix 6: Protect Zero Variance ---
    std[std < 1e-6] = 1.0 
    print(f"Stats check - Max Mean: {mean.max():.2f}, Max Std: {std.max():.2f}")
    
    transform = NormalizeTransform(mean, std)
    
    # Split
    np.random.seed(42) 
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_ds = SequentialStressDataset(all_files[:split_idx], None, transform=transform)
    val_ds = SequentialStressDataset(all_files[split_idx:], None, transform=transform)
    
    # drop_last=True fixes Batch Norm error (also needed for CNN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize Model
    print(f"Initializing {MODEL_TYPE} on {DEVICE}...")
    if MODEL_TYPE == "MAMBA":
        # Simplified Mamba config for stability
        model = MambaBaseline(
            dynamic_len=real_dynamic_features, 
            static_len=STATIC_FEATURES,
            d_model=64,   # Kept small
            n_layers=1    # Kept shallow
        ).to(DEVICE)
    elif MODEL_TYPE == "CNN":
        model = DeepCNNBaseline(input_size=real_input_dim).to(DEVICE)
    elif MODEL_TYPE == "LSTM":
        model = BiLSTMBaseline(dynamic_len=real_dynamic_features, static_len=STATIC_FEATURES).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE")
        
    train_baseline(model, train_loader, val_loader)

if __name__ == "__main__":
    main()