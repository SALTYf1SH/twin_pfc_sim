# -*- coding: utf-8 -*-
"""
Main script for training the surrogate model.

This script handles:
1. Loading the dataset created by data_extractor.py.
2. Defining the Transformer model architecture.
3. Training the model.
4. Evaluating the model's performance.
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# --- Configuration & Hyperparameters ---

# Paths
# Note: We are in the 'dl_model' directory, so we go one level up for the dataset.
DATASET_DIR = "final_dataset"
OUTPUT_DIR = "F:/PFCprj/pfc_twin/twin_pfc_sim/dl_model/trained_models"

# Model Hyperparameters
# INPUT_FEATURES is now determined dynamically from the dataset.
OUTPUT_FEATURES = 64 * 64    # 4096, for the flattened 64x64 grid

# Training Hyperparameters
LEARNING_RATE = 1e-5 # Lowered learning rate for stability
BATCH_SIZE = 32 # Reduced batch size for potentially larger model
NUM_EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.9 # 90% for training, 10% for validation
GRADIENT_CLIP_VALUE = 1.0 # Max norm for gradient clipping

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

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
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
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSurrogateModel(nn.Module):
    """
    A Transformer-based surrogate model for tabular data.
    It treats the input features as a sequence and processes them with a Transformer Encoder.
    """
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerSurrogateModel, self).__init__()
        self.d_model = d_model
        
        self.feature_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.output_layer(x)
        return output

# --- 3. Training & Validation Logic ---

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    """Main training loop."""
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
            # Gradient Clipping
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
            model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model saved to {model_path}")

    print("\nTraining finished.")

# --- 4. Main Execution Block ---

def main():
    """Main function to set up and run the training process."""
    print("========================================================")
    print("      Transformer Surrogate Model Training Script     ")
    print("========================================================")

    if not os.path.isdir(DATASET_DIR):
        print(f"FATAL: Dataset directory not found at '{DATASET_DIR}'")
        print("Please run data_extractor.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{DATASET_DIR}'.")
        return
    
    print(f"Found {len(all_files)} total samples.")

    with np.load(all_files[0]) as first_sample:
        input_features = first_sample['x'].shape[0]
    print(f"Dynamically determined input features: {input_features}")

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
    
    # Create a picklable transform object
    transform = NormalizeTransform(mean, std)
    print("Normalization stats calculated.")
    del stats_dataset, stats_loader, all_x, x_tensor

    # --- Create Datasets and DataLoaders ---
    train_dataset = FractureDataset(train_files, transform=transform)
    val_dataset = FractureDataset(val_files, transform=transform)

    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Transformer model...")
    model = TransformerSurrogateModel(input_size=input_features, output_size=OUTPUT_FEATURES).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
