# -*- coding: utf-8 -*-
"""
Main script for training the DUAL-BRANCH surrogate model (V2 - Advanced).

This script implements an advanced architecture:
1.  Early Fusion: Fuses static parameters with the dynamic sequence *before*
    the Transformer, allowing the model to attend to both.
2.  Convolutional Decoder: Replaces the final MLP with a ConvTranspose2d
    network (like a U-Net decoder) to spatially generate the 
    fracture grid, which is much better for SSIM/MAE.
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

# Paths (Using the same Stress-Based dataset)
DATASET_DIR = "../final_dataset_stress"
OUTPUT_DIR = "trained_models_stress"

# Model Hyperparameters
STATIC_FEATURES = 17 # Number of rock property features (from HQH config)
# DYNAMIC_FEATURES will be determined automatically
OUTPUT_FEATURES = 64 * 64      # 4096, for the flattened 64x64 grid

# Training Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9
GRADIENT_CLIP_VALUE = 1.0

# --- 1. Custom PyTorch Dataset & Transforms (Unchanged) ---

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

# --- 2. Model Architecture (V2 - Advanced) ---

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

# --- [MODIFIED] New V2 Model Architecture ---
class DualBranchModel_v2(nn.Module):
    """
    An advanced dual-branch model with:
    1. Early (pre-attention) fusion
    2. Convolutional (spatial) decoder
    """
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model_static=32, d_model_dynamic=128, 
                 nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel_v2, self).__init__()
        
        self.static_feature_size = static_size
        self.d_model_fused = d_model_static + d_model_dynamic # e.g., 32 + 128 = 160

        # Branch 1: Static MLP
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model_static), # Output: [B, 32]
            nn.ReLU()
        )

        # Branch 2: Dynamic Transformer (operates on FUSED sequence)
        self.dynamic_embedder = nn.Linear(1, d_model_dynamic) # Input: [B, Seq, 1], Output: [B, Seq, 128]
        
        # Positional encoding and Transformer now operate on the FUSED dimension
        self.pos_encoder = PositionalEncoding(self.d_model_fused, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(self.d_model_fused, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # --- [NEW] Strategy 2: Convolutional Decoder ---
        # It takes the final pooled vector [B, 160] and turns it into an image
        
        # 1. Project the fused vector to a small spatial shape (e.g., 4x4)
        self.decoder_input = nn.Linear(self.d_model_fused, 256 * 4 * 4) # -> [B, 4096]

        # 2. Upsampling (ConvTranspose2d) stack
        self.decoder_upsample = nn.Sequential(
            # Input shape: [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 128, 8, 8]
            nn.ReLU(),
            nn.BatchNorm2d(128), # Add BatchNorm for stability
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 64, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [B, 1, 64, 64]
            
            # You might want a final activation depending on your Y data
            # nn.Sigmoid() # If Y is normalized to [0, 1]
            # nn.ReLU()    # If Y is guaranteed non-negative
        )

    def forward(self, x):
        # 1. Split input
        x_static = x[:, :self.static_feature_size]   # [B, 17]
        x_dynamic = x[:, self.static_feature_size:]  # [B, 17]

        # 2. Process static branch
        static_out = self.static_branch(x_static) # -> [B, 32]

        # 3. Process dynamic branch (embedding)
        x_dynamic_unsqueezed = x_dynamic.unsqueeze(-1) # -> [B, 17, 1]
        dynamic_embedded = self.dynamic_embedder(x_dynamic_unsqueezed) # -> [B, 17, 128]
        
        # --- [NEW] Strategy 3: Early Fusion (Pre-Attention) ---
        # Broadcast static features to match dynamic sequence length
        seq_len = x_dynamic.size(1)
        static_broadcasted = static_out.unsqueeze(1).repeat(1, seq_len, 1) # -> [B, 17, 32]

        # Concatenate along the feature dimension
        fused_sequence = torch.cat((dynamic_embedded, static_broadcasted), dim=2) # -> [B, 17, 160]
        
        # 4. Pass FUSED sequence through Transformer
        fused_pos_encoded = self.pos_encoder(fused_sequence)
        fused_transformed = self.transformer_encoder(fused_pos_encoded)
        
        # Aggregate sequence info (mean pooling)
        fused_pooled = fused_transformed.mean(dim=1) # -> [B, 160]

        # 5. --- [NEW] Strategy 2: Pass through Convolutional Decoder ---
        latent = self.decoder_input(fused_pooled) # -> [B, 4096]
        latent_reshaped = latent.view(-1, 256, 4, 4) # -> [B, 256, 4, 4]
        
        output_image = self.decoder_upsample(latent_reshaped) # -> [B, 1, 64, 64]
        
        # Flatten the final image to match the target vector shape
        output = output_image.view(-1, OUTPUT_FEATURES) # -> [B, 4096]
        return output
# --- [END MODIFIED] ---


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
            # --- [MODIFIED] Save to new model name ---
            model_path = os.path.join(OUTPUT_DIR, "best_stress_model_v2.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model (V2) saved to {model_path}")

    print("\nTraining finished.")
    avg_epoch_time = total_training_time / NUM_EPOCHS if NUM_EPOCHS > 0 else 0
    print(f"Average epoch time: {avg_epoch_time:.2f}s")


# --- 4. Main Execution Block ---

def main():
    """Main function to set up and run the training process."""
    print("========================================================")
    print("      Stress-Based Surrogate Model Training (V2)        ") # <-- _MODIFIED_
    print("========================================================")

    # Adjust path to be relative to the script's location in dl_model/
    dataset_full_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)

    if not os.path.isdir(dataset_full_path):
        print(f"FATAL: Dataset directory not found at '{dataset_full_path}'")
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
    
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Input features: {total_input_features} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_features})")

    # --- Data Normalization Setup ---
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats from training set...")
    # Use the same stats file as the original stress model
    stats_file_path = os.path.join(os.path.dirname(__file__), "normalization_stats_stress.npz")
    
    if os.path.exists(stats_file_path):
        print(f"Loading pre-calculated stats from {stats_file_path}")
        stats = np.load(stats_file_path)
        mean = torch.from_numpy(stats['mean'])
        std = torch.from_numpy(stats['std'])
    else:
        print("Calculating from scratch...")
        stats_dataset = FractureDataset(train_files, transform=None)
        stats_loader = DataLoader(stats_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        all_x = []
        for x, _ in tqdm(stats_loader, desc="Loading data for stats"):
            all_x.append(x)
        
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0)
        std = x_tensor.std(dim=0)
        np.savez(stats_file_path, mean=mean.numpy(), std=std.numpy())
        print(f"Saved new normalization stats to {stats_file_path}")

    transform = NormalizeTransform(mean, std)
    print("Normalization stats loaded.")

    # --- Create Datasets and DataLoaders ---
    train_dataset = FractureDataset(train_files, transform=transform)
    val_dataset = FractureDataset(val_files, transform=transform)

    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Advanced Dual-Branch model (V2)...") # <-- _MODIFIED_
    model = DualBranchModel_v2(
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