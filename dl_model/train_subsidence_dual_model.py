# -*- coding: utf-8 -*-
"""
Main script for training the DUAL-BRANCH surrogate model (V4 - Hybrid Loss)
on the SUBSIDENCE dataset.

This script creates the *correct* dual-branch model for the subsidence data,
which was missing before.
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
import torchmetrics

# --- Configuration & Hyperparameters ---
DATASET_DIR = "../final_dataset" # <-- 1. Use Subsidence dataset
OUTPUT_DIR = "trained_models" # <-- 2. Save to original models dir
STATS_PATH = "normalization_stats_subsidence_full.npz" # <-- 3. New stats file name

STATIC_FEATURES = 11 # <-- 4. Subsidence model uses 11 params
OUTPUT_FEATURES = 64 * 64
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# --- 1. Custom PyTorch Dataset & Transforms ---
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

# --- 2. Model Architecture (V1/V4 - Late Fusion + MLP Decoder) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class DualBranchModel(nn.Module):
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
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )
    def forward(self, x):
        x_static, x_dynamic = x[:, :self.static_feature_size], x[:, self.static_feature_size:]
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
    print("\nStarting model training (Subsidence V4 - Hybrid Loss)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    ssim_criterion = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    mse_criterion = nn.MSELoss().to(device)
    mae_criterion = nn.L1Loss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def ssim_loss_fn(pred, target):
        pred_img = pred.view(-1, 1, 64, 64)
        target_img = target.view(-1, 1, 64, 64)
        return 1.0 - ssim_criterion(pred_img, target_img)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss_mse = mse_criterion(outputs, targets)
            loss_mae = mae_criterion(outputs, targets)
            loss_ssim = ssim_loss_fn(outputs, targets)
            
            alpha, beta, gamma = 0.5, 0.3, 0.2
            loss = (alpha * loss_mse) + (beta * loss_mae) + (gamma * loss_ssim)

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
            # --- [MODIFIED] Save to new, clear model name ---
            model_path = os.path.join(OUTPUT_DIR, "best_subsidence_model_dual_branch.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model (Subsidence Dual Branch) saved to {model_path}")

    print("\nTraining finished.")

# --- 4. Main Execution Block ---
def main():
    print("========================================================")
    print("   Subsidence Dual-Branch Model Training (V4-HybridLoss) ")
    print("========================================================")

    script_dir = os.path.dirname(__file__)
    dataset_full_path = os.path.join(script_dir, DATASET_DIR)
    stats_file_path = os.path.join(script_dir, STATS_PATH)
    
    # --- [MODIFIED] Ensure OUTPUT_DIR is correct ---
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(script_dir, OUTPUT_DIR)
    # --- [END MODIFIED] ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{dataset_full_path}'")
        return
    print(f"Found {len(all_files)} total samples.")

    with np.load(all_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    
    # Dynamic features = Total - Static (11)
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Input features: {total_input_features} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_features})")

    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats for *full* subsidence dataset...")
    if os.path.exists(stats_file_path):
        print(f"Loading pre-calculated stats from {stats_file_path}")
        stats = np.load(stats_file_path)
        mean = torch.from_numpy(stats['mean'])
        std = torch.from_numpy(stats['std'])
    else:
        print("Calculating from scratch...")
        stats_dataset = FractureDataset(train_files, transform=None)
        stats_loader = DataLoader(stats_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        all_x = [x for x, _ in tqdm(stats_loader, desc="Loading data for stats")]
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0)
        std = x_tensor.std(dim=0)
        np.savez(stats_file_path, mean=mean.numpy(), std=std.numpy())
        print(f"Saved new normalization stats to {stats_file_path}")

    transform = NormalizeTransform(mean, std)
    print("Normalization stats loaded.")

    print("Calculating y stats for SSIM data_range...")
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")

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

    train_model(model, train_loader, val_loader, device, ssim_data_range)

if __name__ == "__main__":
    main()