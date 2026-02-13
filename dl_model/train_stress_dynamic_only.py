# -*- coding: utf-8 -*-
"""
Script for training a DYNAMIC-ONLY ablation model (Stress-Based).

This model uses only the dynamic Transformer branch from the dual-branch model
to predict the fracture field based solely on the floor stress data
from the 'final_dataset_stress' dataset.
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

STATIC_FEATURES = 17 # Needed to know where dynamic data STARTS
OUTPUT_FEATURES = 64 * 64

LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# --- 1. Custom PyTorch Dataset & Transforms ---

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class DynamicDataset(Dataset):
    """Dataset that loads full data but returns only the dynamic part."""
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            # This logic correctly grabs only the dynamic (stress) part
            x_dynamic = x_full[STATIC_FEATURES:] 
            y = torch.from_numpy(data['y'].astype(np.float32))
        
        if self.transform:
            x_dynamic = self.transform(x_dynamic)
            
        return x_dynamic, y

# --- 2. Model Architecture ---

class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""
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

class DynamicOnlyModel(nn.Module):
    """
    An ablation model using only the dynamic Transformer branch.
    """
    def __init__(self, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DynamicOnlyModel, self).__init__()
        self.d_model = d_model

        # Branch 2: Dynamic Transformer (from original DualBranchModel)
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # New prediction head for this ablation model
        # (Matches the fusion_head from the full model for a fair comparison)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 1024), # Input matches transformer output (d_model)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        # 1. Process through dynamic branch (Transformer)
        # Reshape for embedding: [batch, seq_len] -> [batch, seq_len, 1]
        x_dynamic = x.unsqueeze(-1)
        # Embedding: [batch, seq_len, 1] -> [batch, seq_len, d_model]
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        # Transformer processing
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        # Aggregate sequence info (mean pooling)
        dynamic_out = dynamic_transformed.mean(dim=1)

        # 2. Final prediction from the head
        output = self.prediction_head(dynamic_out)
        return output

# --- 3. Training & Validation Logic ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    print("\nStarting model training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

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
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # --- _MODIFIED_ (New model name) ---
            model_path = os.path.join(OUTPUT_DIR, "best_stress_model_dynamic_only.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model saved to {model_path}")

    print("\nTraining finished.")

# --- 4. Main Execution Block ---

def main():
    print("========================================================")
    print("     Dynamic-Only (Stress) Ablation Model Training      ") # <-- _MODIFIED_
    print("========================================================")

    dataset_full_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    print(f"Found {len(all_files)} total samples.")

    # Determine dynamic_features size from the first sample
    with np.load(all_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Detected Dynamic Features: {dynamic_features}")


    # --- Data Normalization Setup ---
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats from training set (dynamic data only)...")
    # This dataset correctly loads only the dynamic part
    stats_dataset = DynamicDataset(train_files, transform=None)
    stats_loader = DataLoader(stats_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_x_dynamic = []
    for x, _ in tqdm(stats_loader, desc="Loading data for stats"):
        all_x_dynamic.append(x)
    
    x_tensor = torch.cat(all_x_dynamic, dim=0)
    # Note: For sequence data, we normalize across all timesteps and samples
    mean = x_tensor.mean() 
    std = x_tensor.std()
    
    # We save these scalar stats separately
    np.savez_compressed(
        os.path.join(os.path.dirname(__file__), "normalization_stats_dynamic_stress.npz"), 
        mean=mean.numpy(), 
        std=std.numpy()
    )
    print("Normalization stats calculated and saved (scalar).")
    
    transform = NormalizeTransform(mean, std)

    # --- Create Datasets and DataLoaders ---
    train_dataset = DynamicDataset(train_files, transform=transform)
    val_dataset = DynamicDataset(val_files, transform=transform)

    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Dynamic-Only model...")
    model = DynamicOnlyModel(
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