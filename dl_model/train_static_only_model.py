# -*- coding: utf-8 -*-
"""
Script for training a STATIC-ONLY ablation model.

This model uses only the static MLP branch from the original dual-branch model
to predict the fracture field based solely on the 11 geological parameters.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# --- Configuration & Hyperparameters ---

DATASET_DIR = "../final_dataset"
OUTPUT_DIR = "trained_models"

STATIC_FEATURES = 11
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

class StaticDataset(Dataset):
    """Dataset that loads full data but returns only the static part."""
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            x_static = x_full[:STATIC_FEATURES]
            y = torch.from_numpy(data['y'].astype(np.float32))
        
        if self.transform:
            x_static = self.transform(x_static)
            
        return x_static, y

# --- 2. Model Architecture ---

class StaticOnlyModel(nn.Module):
    """
    An ablation model using only the static MLP branch.
    """
    def __init__(self, static_size, output_size, dropout=0.1):
        super(StaticOnlyModel, self).__init__()

        # Static MLP branch (from original DualBranchModel)
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # New prediction head for this ablation model
        self.prediction_head = nn.Sequential(
            nn.Linear(32, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        static_out = self.static_branch(x)
        output = self.prediction_head(static_out)
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
            model_path = os.path.join(OUTPUT_DIR, "best_model_static_only.pth")
            torch.save(model.state_dict(), model_path)
            print(f" -> New best model saved to {model_path}")

    print("\nTraining finished.")

# --- 4. Main Execution Block ---

def main():
    print("========================================================")
    print("      Static-Only Ablation Model Training Script      ")
    print("========================================================")

    dataset_full_path = os.path.join(os.path.dirname(__file__), DATASET_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    print(f"Found {len(all_files)} total samples.")

    # --- Data Normalization Setup ---
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats from training set (static data only)...")
    stats_dataset = StaticDataset(train_files)
    stats_loader = DataLoader(stats_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_x_static = []
    for x, _ in tqdm(stats_loader, desc="Loading data for stats"):
        all_x_static.append(x)
    
    x_tensor = torch.cat(all_x_static, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    
    transform = NormalizeTransform(mean, std)
    print("Normalization stats calculated.")

    # --- Create Datasets and DataLoaders ---
    train_dataset = StaticDataset(train_files, transform=transform)
    val_dataset = StaticDataset(val_files, transform=transform)

    print(f"Splitting data: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Static-Only model...")
    model = StaticOnlyModel(static_size=STATIC_FEATURES, output_size=OUTPUT_FEATURES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
