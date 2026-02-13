# -*- coding: utf-8 -*-
"""
Main script for training the DUAL-BRANCH surrogate model (V4 - Hybrid Loss).

This script uses the V1 (Late Fusion + MLP Decoder) architecture, which
proved most effective (winning on MSE/PCC).

It introduces a HYBRID LOSS FUNCTION (MSE + MAE + SSIM) to solve the
V1 model's weakness on structural metrics.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics # <-- 1. 导入 torchmetrics

# --- Configuration & Hyperparameters ---
DATASET_DIR = "../final_dataset_stress"
OUTPUT_DIR = "trained_models_stress"
STATIC_FEATURES = 17
OUTPUT_FEATURES = 64 * 64
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# --- 1. Custom PyTorch Dataset & Transforms (Unchanged) ---
class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class FractureDataset(Dataset):
    def __init__(self, npz_file_list, transform=None):
        self.file_list, self.transform = npz_file_list, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        with np.load(self.file_list[idx]) as data:
            x = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        if self.transform: x = self.transform(x)
        return x, y

# --- 2. Model Architecture (V1 - Late Fusion + MLP Decoder) ---
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
    """ (This is the V1 Model from train_stress_model.py) """
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        # The MLP Decoder
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
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

# --- 3. Training & Validation Logic (WITH HYBRID LOSS) ---

def train_model(model, train_loader, val_loader, device, ssim_data_range):
    print("\nStarting model training (V4 - Hybrid Loss)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    # --- [MODIFIED] Define all loss functions ---
    # 假设 y 的范围是 [0, 1]，如果不是，请更改 ssim_data_range
    ssim_criterion = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    mse_criterion = nn.MSELoss().to(device)
    mae_criterion = nn.L1Loss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # SSIM 损失需要 (B, C, H, W) 格式
    def ssim_loss_fn(pred, target):
        pred_img = pred.view(-1, 1, 64, 64)
        target_img = target.view(-1, 1, 64, 64)
        # ssim 范围 [0, 1], 越高越好. 损失 = 1 - ssim
        return 1.0 - ssim_criterion(pred_img, target_img)
    # --- [END MODIFIED] ---

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            # --- [MODIFIED] Hybrid Loss calculation ---
            loss_mse = mse_criterion(outputs, targets)
            loss_mae = mae_criterion(outputs, targets)
            loss_ssim = ssim_loss_fn(outputs, targets)
            
            # 权重: 50% MSE (全局), 30% MAE (像素), 20% SSIM (结构)
            # 您可以调整这些权重
            alpha = 0.5
            beta = 0.3
            gamma = 0.2
            
            loss = (alpha * loss_mse) + (beta * loss_mae) + (gamma * loss_ssim)
            # --- [END MODIFIED] ---

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_VALUE)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # 验证集也使用混合损失来选择最佳模型
                loss_mse_val = mse_criterion(outputs, targets)
                loss_mae_val = mae_criterion(outputs, targets)
                loss_ssim_val = ssim_loss_fn(outputs, targets)
                loss_val = (alpha * loss_mse_val) + (beta * loss_mae_val) + (gamma * loss_ssim_val)
                val_loss += loss_val.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # --- [MODIFIED] Save to new model name ---
            model_path = os.path.join(OUTPUT_DIR, "best_stress_model_v4_hybrid_loss.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model (V4) saved to {model_path}")

    print("\nTraining finished.")


# --- 4. Main Execution Block ---

def main():
    print("========================================================")
    print("   Stress-Based Surrogate Model Training (V4-HybridLoss) ")
    print("========================================================")

    dataset_full_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    print(f"Found {len(all_files)} total samples.")

    with np.load(all_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Input features: {total_input_features} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_features})")

    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Loading normalization stats (from V1 model)...")
    stats_file_path = os.path.join(os.path.dirname(__file__), "normalization_stats_stress.npz")
    
    try:
        stats = np.load(stats_file_path)
        mean = torch.from_numpy(stats['mean'])
        std = torch.from_numpy(stats['std'])
    except FileNotFoundError:
        print(f"FATAL: Stats file not found at {stats_file_path}. Please run 'train_stress_model.py' first.")
        return
    
    transform = NormalizeTransform(mean, std)
    print("Normalization stats loaded.")

    # --- [MODIFIED] Get data range for SSIM ---
    print("Calculating y stats for SSIM data_range...")
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")
    # --- [END MODIFIED] ---

    train_dataset = FractureDataset(train_files, transform=transform)
    val_dataset = FractureDataset(val_files, transform=transform)
    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Dual-Branch model (V1 Architecture)...")
    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_features,
        output_size=OUTPUT_FEATURES
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # criterion 和 optimizer 在 train_model 函数内部定义
    train_model(model, train_loader, val_loader, device, ssim_data_range)

if __name__ == "__main__":
    main()