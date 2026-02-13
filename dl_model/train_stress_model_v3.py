# -*- coding: utf-8 -*-
"""
Main script for training the DUAL-BRANCH surrogate model (V3 - Best of Both).

This script implements an architecture that combines the successful parts
of V1 and V2:
1.  Late Fusion (from V1): Static and Dynamic branches process data *separately*.
2.  Convolutional Decoder (from V2): Replaces the final MLP with a 
    ConvTranspose2d network to spatially generate the fracture grid.
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
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class FractureDataset(Dataset):
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        if self.transform:
            x = self.transform(x)
        return x, y

# --- 2. Model Architecture (V3) ---

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

# --- [MODIFIED] New V3 Model Architecture ---
class DualBranchModel_v3(nn.Module):
    """
    An advanced dual-branch model with:
    1. Late Fusion (like V1)
    2. Convolutional Decoder (like V2)
    """
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model_static=32, d_model_dynamic=128, 
                 nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel_v3, self).__init__()
        
        self.static_feature_size = static_size
        self.d_model_static = d_model_static
        self.d_model_dynamic = d_model_dynamic
        self.d_model_fused = d_model_static + d_model_dynamic # 32 + 128 = 160

        # Branch 1: Static MLP (Processes static data independently)
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model_static), # Output: [B, 32]
            nn.ReLU()
        )

        # Branch 2: Dynamic Transformer (Processes dynamic data independently)
        self.dynamic_embedder = nn.Linear(1, d_model_dynamic) # Input: [B, Seq, 1], Output: [B, Seq, 128]
        self.pos_encoder = PositionalEncoding(d_model_dynamic, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model_dynamic, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # --- Convolutional Decoder (from V2) ---
        # It takes the FUSED vector [B, 160] and turns it into an image
        self.decoder_input = nn.Linear(self.d_model_fused, 256 * 4 * 4) # Input [B, 160] -> Output [B, 4096]

        self.decoder_upsample = nn.Sequential(
            # Input shape: [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 128, 8, 8]
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 64, 16, 16]
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 32, 32, 32]
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 1, 64, 64]
        )

    def forward(self, x):
        # 1. Split input
        x_static = x[:, :self.static_feature_size]   # [B, 17]
        x_dynamic = x[:, self.static_feature_size:]  # [B, 17]

        # 2. Process Static Branch (Independent)
        static_out = self.static_branch(x_static) # -> [B, 32]

        # 3. Process Dynamic Branch (Independent)
        x_dynamic_unsqueezed = x_dynamic.unsqueeze(-1) # -> [B, 17, 1]
        dynamic_embedded = self.dynamic_embedder(x_dynamic_unsqueezed) # -> [B, 17, 128]
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        dynamic_out = dynamic_transformed.mean(dim=1) # -> [B, 128]

        # 4. --- [NEW] Late Fusion (like V1) ---
        fused = torch.cat((static_out, dynamic_out), dim=1) # -> [B, 160]
        
        # 5. --- Pass FUSED vector through Convolutional Decoder ---
        latent = self.decoder_input(fused) # -> [B, 4096]
        latent_reshaped = latent.view(-1, 256, 4, 4) # -> [B, 256, 4, 4]
        
        output_image = self.decoder_upsample(latent_reshaped) # -> [B, 1, 64, 64]
        
        # Flatten the final image to match the target vector shape
        output = output_image.view(-1, OUTPUT_FEATURES) # -> [B, 4096]
        return output
# --- [END MODIFIED] ---


# --- 3. Training & Validation Logic ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    print("\nStarting model training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    # --- [OPTIONAL] Add Hybrid Loss (Strategy 1) ---
    # ssim_criterion = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    # mse_criterion = nn.MSELoss()
    # mae_criterion = nn.L1Loss()
    # def ssim_loss_fn(pred, target):
    #     pred_img = pred.reshape(-1, 1, 64, 64)
    #     target_img = target.reshape(-1, 1, 64, 64)
    #     return 1.0 - ssim_criterion(pred_img, target_img)
    # -----------------------------------------------

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            # --- Original MSE Loss ---
            loss = criterion(outputs, targets)
            
            # --- [OPTIONAL] Hybrid Loss calculation ---
            # loss_mse = mse_criterion(outputs, targets)
            # loss_mae = mae_criterion(outputs, targets)
            # loss_ssim = ssim_loss_fn(outputs, targets)
            # loss = (0.5 * loss_mse) + (0.3 * loss_mae) + (0.2 * loss_ssim)
            # -------------------------------------------

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
                loss = criterion(outputs, targets) # Val loss still uses main criterion
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # --- [MODIFIED] Save to new model name ---
            model_path = os.path.join(OUTPUT_DIR, "best_stress_model_v3.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model (V3) saved to {model_path}")

    print("\nTraining finished.")


# --- 4. Main Execution Block ---

def main():
    print("========================================================")
    print("      Stress-Based Surrogate Model Training (V3)        ")
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

    # --- Data Normalization Setup ---
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Loading normalization stats from training set...")
    # V3 model uses the *same* normalization file as V1
    stats_file_path = os.path.join(os.path.dirname(__file__), "normalization_stats_stress.npz")
    
    if os.path.exists(stats_file_path):
        print(f"Loading pre-calculated stats from {stats_file_path}")
        stats = np.load(stats_file_path)
        mean = torch.from_numpy(stats['mean'])
        std = torch.from_numpy(stats['std'])
    else:
        print(f"FATAL: Stats file not found at {stats_file_path}. Please run 'train_stress_model.py' first to generate it.")
        return

    transform = NormalizeTransform(mean, std)
    print("Normalization stats loaded.")

    # --- Create Datasets and DataLoaders ---
    train_dataset = FractureDataset(train_files, transform=transform)
    val_dataset = FractureDataset(val_files, transform=transform)
    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Advanced Dual-Branch model (V3 - Late Fusion + ConvDecoder)...")
    model = DualBranchModel_v3(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_features,
        output_size=OUTPUT_FEATURES
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()