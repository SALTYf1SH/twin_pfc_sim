# -*- coding: utf-8 -*-
"""
Model Evaluation Script

This script loads a trained surrogate model and evaluates its performance on the
validation set. It provides both quantitative metrics and visual comparisons.

Evaluation includes:
1.  Quantitative Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE),
    and Structural Similarity Index (SSIM).
2.  Visual Comparison: A 3-panel plot showing the Ground Truth, Model Prediction,
    and the Difference Map for a few sample cases.

How to run:
# For a few visual samples with corrected orientation
python dl_model/evaluate_model.py --model_path trained_models/best_dual_branch_model.pth --num_samples 5

# For a full quantitative evaluation on the entire validation set
python dl_model/evaluate_model.py --model_path trained_models/best_dual_branch_model.pth --all
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Attempt to import scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# --- Configuration ---

# Paths (relative to the project root)
DATASET_DIR = "final_dataset"
DEFAULT_MODEL_PATH = "dl_model/trained_models/best_dual_branch_model.pth"
DEFAULT_OUTPUT_DIR = "dl_model/evaluation_results"

# Data handling
TRAIN_VAL_SPLIT_RATIO = 0.9 # Must be the same as used in training
BATCH_SIZE = 1 # Process one sample at a time for evaluation
RANDOM_SEED = 42 # Use a fixed seed for reproducibility of train/val split

# --- Model Definitions (Copied from training script to make script self-contained) ---

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
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
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

# --- Data Loading & Normalization Classes ---

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
        return x, y, os.path.basename(filepath)

# --- Main Evaluation Logic ---

def evaluate(args):
    """Main function to run the evaluation."""
    print("======================================================")
    print("            Model Evaluation Script                 ")
    print("======================================================")

    if not SKIMAGE_AVAILABLE:
        print("\nWARNING: scikit-image not found. SSIM metric will be skipped.")
        print("Please install it: pip install scikit-image\n")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # --- Load Data and Recreate Train/Val Split ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{DATASET_DIR}'.")
        return

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print(f"Recreated dataset split: {len(train_files)} train, {len(val_files)} val.")

    # --- Calculate Normalization Stats from Training Set ---
    print("Calculating normalization stats from training set...")
    if os.path.exists("dl_model/normalization_stats.npz"):
        print("Loading pre-calculated normalization stats.")
        stats = np.load("dl_model/normalization_stats.npz")
        mean = torch.from_numpy(stats['mean'])
        std = torch.from_numpy(stats['std'])
    else:
        print("Calculating from scratch... (This may take a moment)")
        stats_dataset = FractureDataset(train_files)
        stats_loader = DataLoader(stats_dataset, batch_size=64, shuffle=False)
        all_x = [x for x, _, _ in tqdm(stats_loader, desc="Loading data for stats")]
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0)
        std = x_tensor.std(dim=0)
        np.savez("dl_model/normalization_stats.npz", mean=mean.numpy(), std=std.numpy())
        print("Saved normalization stats to dl_model/normalization_stats.npz")

    transform = NormalizeTransform(mean, std)

    # --- Load Model ---
    with np.load(val_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    
    static_features = 11
    dynamic_features = total_input_features - static_features

    model = DualBranchModel(
        static_size=static_features,
        dynamic_size=dynamic_features,
        output_size=64*64
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Perform Evaluation on Validation Set ---
    val_dataset = FractureDataset(val_files, transform=transform)
    # Use a larger batch size for the full evaluation run
    eval_batch_size = 64 if args.all else 1
    # No need to shuffle if we are evaluating all samples
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=not args.all)

    total_mse, total_mae, total_ssim = 0, 0, 0
    samples_processed = 0
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    if args.all:
        print(f"\n--- Evaluating all {len(val_dataset)} samples in validation set ---")
        iterator = tqdm(val_loader, desc="Evaluating all samples")
    else:
        print(f"\n--- Evaluating {args.num_samples} random samples from validation set ---")
        iterator = val_loader

    with torch.no_grad():
        for i, (inputs, targets, fnames) in enumerate(iterator):
            if not args.all and i >= args.num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # --- Quantitative Evaluation ---
            mse = criterion_mse(outputs, targets).item()
            mae = criterion_mae(outputs, targets).item()
            total_mse += mse * inputs.size(0)
            total_mae += mae * inputs.size(0)

            # Reshape for image-based metrics and visualization
            target_grids = targets.cpu().numpy().reshape(-1, 64, 64)
            output_grids = outputs.cpu().numpy().reshape(-1, 64, 64)

            if SKIMAGE_AVAILABLE:
                for j in range(target_grids.shape[0]):
                    target_grid = target_grids[j]
                    output_grid = output_grids[j]
                    data_range = max(target_grid.max() - target_grid.min(), output_grid.max() - output_grid.min())
                    if data_range == 0: data_range = 1.0
                    total_ssim += ssim(target_grid, output_grid, data_range=data_range)
            
            samples_processed += inputs.size(0)

            # --- Visualization (only if not in --all mode) ---
            if not args.all:
                target_grid = np.rot90(target_grids[0])
                output_grid = np.rot90(output_grids[0])
                fname = fnames[0]

                print(f"\nSample {i+1}/{args.num_samples} (File: {fname}):")
                print(f"  - MSE : {mse:.6f}")
                print(f"  - MAE : {mae:.6f}")
                if SKIMAGE_AVAILABLE: print(f"  - SSIM: {total_ssim / samples_processed:.6f}")

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Sample: {fname}\n MSE: {mse:.4f} | MAE: {mae:.4f} | SSIM: {total_ssim / samples_processed:.4f}', fontsize=16)

                im1 = axes[0].imshow(target_grid, cmap='viridis', interpolation='nearest')
                axes[0].set_title('Ground Truth (Rotated)')
                fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                im2 = axes[1].imshow(output_grid, cmap='viridis', interpolation='nearest')
                axes[1].set_title('Model Prediction (Rotated)')
                fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                diff_map = np.abs(target_grid - output_grid)
                im3 = axes[2].imshow(diff_map, cmap='plasma', interpolation='nearest')
                axes[2].set_title('Absolute Difference')
                fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                for ax in axes: ax.axis('off')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                save_path = os.path.join(args.output_dir, f"evaluation_sample_{i+1}.png")
                plt.savefig(save_path)
                plt.close(fig)
                print(f"  - Saved visualization to {save_path}")

    # --- Final Summary ---
    avg_mse = total_mse / samples_processed
    avg_mae = total_mae / samples_processed
    avg_ssim = total_ssim / samples_processed if SKIMAGE_AVAILABLE else -1

    print("======================================================")
    print("               Evaluation Summary                   ")
    print("======================================================")
    print(f"Average over {samples_processed} samples:")
    print(f"  - Average MSE : {avg_mse:.6f}")
    print(f"  - Average MAE : {avg_mae:.6f}")
    if SKIMAGE_AVAILABLE: print(f"  - Average SSIM: {avg_ssim:.6f}")
    print("======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained surrogate model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model .pth file.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize (ignored if --all is used).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation plots.")
    parser.add_argument("--all", action="store_true", help="Evaluate on all samples in the validation set (no plots will be saved).")
    
    parsed_args = parser.parse_args()
    evaluate(parsed_args)
