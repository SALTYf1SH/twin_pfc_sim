# -*- coding: utf-8 -*-
"""
Main script for training the DUAL-BRANCH surrogate model (Stress-Based).

This script is modified to use the (Rock Params + Floor Stress) dataset.

1. A static branch (MLP) processes 17 time-invariant rock parameters.
2. A dynamic branch (Transformer) processes time-variant floor stress data.
3. The features are fused and passed to a final MLP head for prediction.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# --- Configuration & Hyperparameters ---

# Paths
DATASET_DIR = "../final_dataset_stress"  # <-- _MODIFIED_
OUTPUT_DIR = "trained_models_stress_physics"      # <-- _MODIFIED_

# Model Hyperparameters
STATIC_FEATURES = 17 # <-- _MODIFIED_ (Was 11, now 17 for the HQH parameter set)
# DYNAMIC_FEATURES will be determined automatically
OUTPUT_FEATURES = 64 * 64      # 4096, for the flattened 64x64 grid

# Training Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# --- 1. Custom PyTorch Dataset & Transforms ---

class NormalizeTransform:
    """A picklable transform class for normalization."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class FractureDataset(Dataset):
    """Custom dataset for loading the .npz samples."""
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

# --- 2. Model Architecture ---
# (The model architecture itself is generic and does not need changes,
# as it accepts static_size and dynamic_size as parameters.)

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

class DualBranchModel(nn.Module):
    """
    A dual-branch model combining an MLP for static features and a Transformer
    for dynamic sequence features.
    """
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model

        # Branch 1: Static MLP
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Branch 2: Dynamic Transformer
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Fusion Head
        fusion_input_size = 32 + d_model # 32 from static branch, d_model from dynamic
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        # 1. Split input into static and dynamic parts
        # This line automatically works because self.static_feature_size is now 17
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]

        # 2. Process through static branch (MLP)
        static_out = self.static_branch(x_static)

        # 3. Process through dynamic branch (Transformer)
        # Reshape for embedding: [batch, seq_len] -> [batch, seq_len, 1]
        x_dynamic = x_dynamic.unsqueeze(-1)
        # Embedding: [batch, seq_len, 1] -> [batch, seq_len, d_model]
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        # Transformer processing
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        # Aggregate sequence info (mean pooling)
        dynamic_out = dynamic_transformed.mean(dim=1)

        # 4. Fuse the outputs of the two branches
        fused = torch.cat((static_out, dynamic_out), dim=1)

        # 5. Final prediction from the fusion head
        output = self.fusion_head(fused)
        return output

# --- 3. Training & Validation Logic ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    """Main training loop."""
    print("\nStarting model training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    total_training_time = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
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
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_training_time += epoch_duration

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # --- _MODIFIED_ Save to new model name ---
            model_path = os.path.join(OUTPUT_DIR, "best_stress_physics_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model saved to {model_path}")

    print("\nTraining finished.")
    avg_epoch_time = total_training_time / NUM_EPOCHS if NUM_EPOCHS > 0 else 0
    print(f"Average epoch time: {avg_epoch_time:.2f}s")


# --- 4. Main Execution Block ---

def main():
    """Main function to set up and run the training process."""
    print("========================================================")
    print("      Stress-Based Surrogate Model Training Script      ") # <-- _MODIFIED_
    print("========================================================")

    # Adjust path to be relative to the script's location in dl_model/
    dataset_full_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)

    if not os.path.isdir(dataset_full_path):
        print(f"FATAL: Dataset directory not found at '{dataset_full_path}'")
        # --- _MODIFIED_ Updated help message ---
        print("Please run process_stress_data.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{dataset_full_path}'.")
        return
    
    print(f"Found {len(all_files)} total samples.")

    with np.load(all_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    
    # This logic now correctly uses STATIC_FEATURES = 17
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Input features: {total_input_features} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_features})")

    # --- Data Normalization Setup ---
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats from training set...")
    stats_dataset = FractureDataset(train_files)
    stats_loader = DataLoader(stats_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_x = []
    for x, _ in tqdm(stats_loader, desc="Loading data for stats"):
        all_x.append(x)
    
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    
    transform = NormalizeTransform(mean, std)
    print("Normalization stats calculated.")
    del stats_dataset, stats_loader, all_x, x_tensor

    # --- Create Datasets and DataLoaders ---
    train_dataset = FractureDataset(train_files, transform=transform)
    val_dataset = FractureDataset(val_files, transform=transform)

    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Stress-Based Dual-Branch model...") # <-- _MODIFIED_
    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_features,
        output_size=OUTPUT_FEATURES
    ).to(device)
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()