# -*- coding: utf-8 -*-
"""
Baseline Training Script with MAMBA (Stability Fixed) for STRESS-Based Inversion.
FIXED VERSION: Corrects Dataset Reshaping and Transpose Logic.

Specific Configuration for Stress Dataset:
- STATIC_FEATURES = 17
- Dataset Path: ../final_dataset_stress

Stability Fixes Applied:
1. Learning Rate = 1e-5
2. Mamba Model: Pre-Norm structure + Float32 enforcement + Residual scaling
3. Training: Gradient Clipping (0.5) + NaN skipping
4. Dataset: Correctly handles 1D inputs vs 2D transposed targets.
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
import argparse

# --- 0. Configuration ---

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="MAMBA", 
                    choices=["MAMBA", "CNN", "LSTM", "TRANSFORMER"],
                    help="Choose baseline model architecture")
args, _ = parser.parse_known_args()

MODEL_TYPE = args.model_type
print(f"\n>>> CONFIGURATION: Training Model = {MODEL_TYPE} <<<\n")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")  # Stress Data
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_baselines_stress")
# Params json is not used for training logic here, but good to have reference
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

# Hyperparameters
STATIC_FEATURES = 17      # Stress dataset has 17 static features
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
OUTPUT_SIZE = 4096
IMG_SIZE = 64             # Explicitly defined

BATCH_SIZE = 32
# === Stability Fix 1: Lower Learning Rate ===
# LEARNING_RATE = 1e-5  (Replaced by Scheduler)
WARMUP_EPOCHS = 10
MAX_LR = 1e-4
MIN_LR = 1e-6
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
            x_t = data['x'].astype(np.float32)
            y_t = data['y'].astype(np.float32)
            
            # [FIX]: Dataset Logic to prevent Shape Mismatch
            # 1. x_t (Input Features): Keep as 1D (Size 34: 17 static + 17 dynamic)
            # 2. y_t (Target Image): Reshape if flattened, then Transpose for physical alignment
            
            if y_t.ndim == 1: 
                y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T # Transpose (Row=Height, Col=Length)
            
            # Convert to Tensor
            x_t = torch.from_numpy(x_t)
            y_t = torch.from_numpy(y_t)
            
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
        
        # Original CNN structure (No BN1 after FC, matched to previous scripts)
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
        # Matched to Stress architecture (Contains Dropout)
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.2), # Stress LSTM has dropout
            nn.Linear(1024, output_size)
        )
    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class TransformerDualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size=4096, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerDualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
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

        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos = self.pos_encoder(dynamic_embedded)
        dynamic_out = self.transformer_encoder(dynamic_pos).mean(dim=1)
        fused = torch.cat((static_out, dynamic_out), dim=1)
        return self.fusion_head(fused)

# --- 3. Training Engine ---

def train_baseline(model, train_loader, val_loader):
    print(f"\n>>> Training Stress Baseline (with Scheduler): {MODEL_TYPE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.MSELoss()
    
    # Scheduler Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        else:
            progress = float(epoch - WARMUP_EPOCHS) / float(NUM_EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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

            # [FIX] Flatten Target for MSE
            y_flat = y.view(y.size(0), -1)
            loss = criterion(pred, y_flat)
            
            # --- Stability Fix 4: Loss Check ---
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
                
            loss.backward()
            
            # --- Stability Fix 5: Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()

        # Step Scheduler
        scheduler_lr.step()
        current_lr = optimizer.param_groups[0]['lr']
            
        model.eval()
        val_mse, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                
                # MSE needs flat target
                y_flat = y.view(y.size(0), -1)
                val_mse += criterion(pred, y_flat).item()
                
                # SSIM needs image shapes
                pred_img = pred.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                y_img = y.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                
                val_ssim += calculate_ssim(torch.clamp(pred_img, 0, 1), y_img)

        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"Epoch {epoch+1}: LR={current_lr:.2e} | Train MSE={avg_train:.5f} | Val MSE={avg_val_mse:.5f} | Val SSIM={avg_val_ssim:.4f}")
        
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

    # Normalization with Protection (Full Dataset)
    print("Computing normalization stats (Scanning full dataset)...")
    temp_ds = SequentialStressDataset(all_files, None, transform=None)
    temp_loader = DataLoader(temp_ds, batch_size=128, num_workers=0) # Larger batch for speed
    
    all_x = []
    for x, _ in tqdm(temp_loader, desc="Scanning Dataset"):
        all_x.append(x)
        
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    
    # --- Stability Fix 6: Protect Zero Variance ---
    std[std < 1e-6] = 1.0 
    
    # Save Stats
    stats_path = os.path.join(OUTPUT_DIR, "baseline_stress_stats.pt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({'mean': mean, 'std': std}, stats_path)
    print(f"Stats saved to: {stats_path}")
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
    elif MODEL_TYPE == "TRANSFORMER":
        model = TransformerDualBranchModel(static_size=STATIC_FEATURES, dynamic_size=real_dynamic_features, output_size=OUTPUT_SIZE).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE")
        
    # Pass None for scheduler in argument if I modified signature? 
    # Wait, I modified `train_baseline` signature in chunk 2 but I am creating scheduler INSIDE `train_baseline` in chunk 3.
    # So chunk 2 signature change `def train_baseline(..., scheduler):` is WRONG if I create it inside.
    # I should NOT change the signature if I create it inside. 
    # OR I should create it outside and pass it in.
    # Creating it inside is easier given the current structure.
    # So I will revert Chunk 2 change in my mind and ONLY change the print.
    train_baseline(model, train_loader, val_loader)

if __name__ == "__main__":
    main()