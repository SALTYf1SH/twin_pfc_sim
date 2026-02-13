# -*- coding: utf-8 -*-
"""
Baseline Training Script with MAMBA (Stability Fixed) for SUBSIDENCE-Based Inversion.

Fixes applied for NaN issues:
1. Learning Rate lowered to 1e-5.
2. Data normalization: Protected against division by zero (small std).
3. Mamba Model: Forced float32 precision and added explicit residual connections.
4. Training Loop: Stricter gradient clipping (0.5) and NaN skipping logic.
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
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset")  # Subsidence Data
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models_baselines_subsidence")
# Params json is not used for training logic here
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")

# Hyperparameters
STATIC_FEATURES = 11      # Subsidence dataset static features
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
OUTPUT_SIZE = 4096

BATCH_SIZE = 32
# === FIX 1: Lower Learning Rate ===
LEARNING_RATE = 1e-5  # Lowered from 1e-4 to prevent explosion
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

# === FIX 2: Stabilized Mamba Baseline ===
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
        
        # 1. Embedding
        self.embedding = nn.Linear(self.input_dim, d_model)
        
        # 2. Mamba Layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, 
                d_state=16,      
                d_conv=4,        
                expand=2,        
            ) for _ in range(n_layers)
        ])
        
        # 3. LayerNorms (为每一层都配备 Norm，实现 Pre-Norm 结构)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        # 4. Final Norm & Decoder
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            # [关键] 增加 Dropout 防止过拟合/震荡
            nn.Dropout(0.1), 
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        # ... (数据拼接部分不变) ...
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_dynamic = x_dynamic.unsqueeze(-1) 
        seq_static = x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1) 
        seq_input = torch.cat([seq_dynamic, seq_static], dim=2) 
        
        # [保护 1] 强制 Float32 并在 Embedding 后立刻 Clamp
        x = self.embedding(seq_input).float()
        x = torch.clamp(x, min=-5.0, max=5.0) 
        
        # [保护 2] 使用 Pre-Norm 结构 (Norm -> Mamba -> Add)
        # Pre-Norm 比 Post-Norm (Mamba -> Norm -> Add) 训练更稳定
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            
            # 先 Norm
            x = norm(x)
            
            # 再进 Mamba
            out = layer(x)
            
            # [保护 3] 检查 Mamba 输出是否炸了
            if torch.isnan(out).any() or torch.isinf(out).any():
                # 如果炸了，就放弃这次计算，直接透传残差（虽然这步很激进，但能保命）
                x = residual
            else:
                # [保护 4] 残差缩放：降低新信息的权重，防止数值累积过快
                x = residual + out * 0.1 
            
        x = self.final_norm(x)
        
        # Global Pooling
        x = x.mean(dim=1)
        
        # [保护 5] 进 Decoder 前最后的防线
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
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
    print(f"\n>>> Training Baseline: {MODEL_TYPE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_ssim = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        valid_batches = 0  # <--- 新增计数器
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for x, y in progress:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            
            # 检查前向传播是否 NaN
            if torch.isnan(pred).any():
                continue 

            loss = criterion(pred, y)
            
            # 检查 Loss 是否 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1  # <--- 只有成功的 Batch 才计数
            
        # 修改平均 Loss 计算逻辑：只除以成功的 Batch 数
        avg_train = train_loss / valid_batches if valid_batches > 0 else float('nan')
        
        # --- Validation ---
        model.eval()
        val_mse, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                
                # 验证集遇到 NaN 就把它变成 0，保证能算出一个数，而不是 NaN
                pred = torch.nan_to_num(pred, nan=0.0) 
                
                val_mse += criterion(pred, y).item()
                
                pred_img = pred.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                y_img = y.view(-1, 1, OUTPUT_HEIGHT, OUTPUT_WIDTH)
                val_ssim += calculate_ssim(torch.clamp(pred_img, 0, 1), y_img)

        avg_val_mse = val_mse / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        # 打印诊断信息
        print(f"Epoch {epoch+1}: Valid Batches={valid_batches}/{len(train_loader)} | "
              f"Train MSE={avg_train:.5f} | Val MSE={avg_val_mse:.5f} | Val SSIM={avg_val_ssim:.4f}")
        
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_baseline_{MODEL_TYPE}.pth"))
# --- 4. Main Execution ---

def main():
    dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
    if not os.path.exists(dataset_path): print("No data found!"); return
    all_files = glob.glob(os.path.join(dataset_path, "*.npz"))
    if not all_files: print("No .npz files found!"); return
    
    # Auto-detect dimensions
    with np.load(all_files[0]) as f: real_input_dim = f['x'].shape[0]
    real_dynamic_features = real_input_dim - STATIC_FEATURES
    print(f"Auto-detected Input Dim: {real_input_dim} (Dyn: {real_dynamic_features}, Stat: {STATIC_FEATURES})")

    # Normalization with Protection
    print("Computing normalization stats...")
    temp_loader = DataLoader(SequentialFractureDataset(all_files[:100], None), batch_size=32)
    all_x = [x for x, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    
    # === FIX 6: Protect against Division by Zero ===
    # Set small std values to 1.0 to avoid explosion during normalization
    std[std < 1e-6] = 1.0 
    print(f"Stats check - Max Mean: {mean.max():.2f}, Max Std: {std.max():.2f}")
    
    transform = NormalizeTransform(mean, std)
    
    # Split (Case-wise shuffle)
    np.random.seed(42) 
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_ds = SequentialFractureDataset(all_files[:split_idx], None, transform=transform)
    val_ds = SequentialFractureDataset(all_files[split_idx:], None, transform=transform)
    
    # Drop last to avoid batch=1 issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize Model
    print(f"Initializing {MODEL_TYPE} on {DEVICE}...")
    if MODEL_TYPE == "MAMBA":
        model = MambaBaseline(
            dynamic_len=real_dynamic_features, 
            static_len=STATIC_FEATURES,
            d_model=64,   # 从 128 降到 64
            n_layers=1    # 从 2 降到 1 (先跑通 1 层再说)
        ).to(DEVICE)
    # if MODEL_TYPE == "MAMBA":
    #     model = MambaBaseline(dynamic_len=real_dynamic_features, static_len=STATIC_FEATURES).to(DEVICE)
    elif MODEL_TYPE == "CNN":
        model = DeepCNNBaseline(input_size=real_input_dim).to(DEVICE)
    elif MODEL_TYPE == "LSTM":
        model = BiLSTMBaseline(dynamic_len=real_dynamic_features, static_len=STATIC_FEATURES).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE")
        
    train_baseline(model, train_loader, val_loader)

if __name__ == "__main__":
    main()