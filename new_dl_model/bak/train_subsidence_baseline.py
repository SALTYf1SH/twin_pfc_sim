# -*- coding: utf-8 -*-
"""
Baseline Training Script for SUBSIDENCE-Based Inversion.
Models: 
  1. Deep CNN (ResNet-Generator Style)
  2. Bi-LSTM (Sequence-to-Image)

Features:
- STATIC_FEATURES = 11 (Specific to Subsidence Dataset)
- Auto-detects input dimension (fixes the 53 vs 107 error)
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
# Options: "CNN", "LSTM"
MODEL_TYPE = "CNN" 

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset")  # Subsidence Data
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_baselines_subsidence")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")

# Hyperparameters
STATIC_FEATURES = 11      # Subsidence dataset usually has 11 static features
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
OUTPUT_SIZE = 4096

BATCH_SIZE = 32
LEARNING_RATE = 1e-4 
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Dataset & Utilities ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
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

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128
        self.init_size = 8
        
        # 1. 投影层
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        
        # 2. 上采样卷积层
        self.conv_blocks = nn.Sequential(
            # 将 BN 移到 Reshape 之后，使用 BatchNorm2d 更符合图像生成逻辑
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
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
            # 输出层不加 BN 和 ReLU，直接回归数值
        )

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x = self.fc(x)
        
        # 关键：先 Reshape 成 4D 张量 [Batch, 128, 8, 8]
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # 然后再进入卷积块 (BN2d -> ReLU -> ConvT -> ...)
        x = self.conv_blocks(x)
        
        return x.view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        
        self.input_dim = 1 + static_len
        self.hidden_dim = 256
        self.num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        
        seq_dynamic = x_dynamic.unsqueeze(-1)
        seq_static = x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)
        lstm_input = torch.cat([seq_dynamic, seq_static], dim=2)
        
        lstm_out, _ = self.lstm(lstm_input)
        global_feat = torch.mean(lstm_out, dim=1)
        return self.decoder(global_feat)

# --- 3. Training Engine ---

def train_baseline(model, train_loader, val_loader):
    print(f"\n>>> Training Subsidence Baseline: {MODEL_TYPE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_ssim = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_mse = 0.0
        val_ssim = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_mse += criterion(pred, y).item()
                pred_img = pred.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                y_img = y.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                val_ssim += calculate_ssim(torch.clamp(pred_img, 0, 1), y_img)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train MSE={avg_train_loss:.5f} | Val MSE={avg_val_mse:.5f} | Val SSIM={avg_val_ssim:.4f}")
        
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_baseline_subsidence_{MODEL_TYPE}.pth"))
            print(f"  [*] Best Model Saved: {avg_val_ssim:.4f}")

# --- 4. Main Execution ---

def main():
    dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
    if not os.path.exists(dataset_path): print("No data found!"); return
    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No .npz files found!"); return
    
    # === 关键修正：自动读取输入维度 ===
    with np.load(all_files[0]) as f:
        real_input_dim = f['x'].shape[0]
    
    real_dynamic_features = real_input_dim - STATIC_FEATURES
    print(f"Auto-detected Input Dim: {real_input_dim}")
    print(f" -> Static: {STATIC_FEATURES}, Dynamic: {real_dynamic_features}")
    
    if real_dynamic_features <= 0:
        raise ValueError(f"Error: Calculated dynamic features is {real_dynamic_features}. Check STATIC_FEATURES setting.")

    # Normalization
    print("Computing normalization stats...")
    temp_loader = DataLoader(SequentialFractureDataset(all_files[:100], None), batch_size=32)
    all_x = [x for x, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    mean, std = x_tensor.mean(dim=0), x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    
    # Split
    np.random.seed(42) 
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    train_ds = SequentialFractureDataset(train_files, None, transform=transform)
    val_ds = SequentialFractureDataset(val_files, None, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize Model with REAL DIMENSIONS
    if MODEL_TYPE == "CNN":
        model = DeepCNNBaseline(input_size=real_input_dim).to(DEVICE)
    elif MODEL_TYPE == "LSTM":
        model = BiLSTMBaseline(
            dynamic_len=real_dynamic_features, 
            static_len=STATIC_FEATURES
        ).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE")
        
    print(f"Initialized {MODEL_TYPE} model on {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    train_baseline(model, train_loader, val_loader)

if __name__ == "__main__":
    main()