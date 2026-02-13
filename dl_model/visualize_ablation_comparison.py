# -*- coding: utf-8 -*-
"""
Script to VISUALLY compare the performance of the three STRESS-BASED
ablation models (Full V4, Static-Only, Dynamic-Only) on specific samples.

V5: 
- Fixed plot orientation. Replaced np.rot90() with .T (transpose)
  to match the user's reference script (compare_prediction.py).
- Kept 'jet' colormap and 'origin=lower'.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics
import argparse

# --- Configuration ---
DATASET_DIR = "../final_dataset_stress"
SCRIPT_DIR = os.path.dirname(__file__)

TRAINED_MODELS_DIR = "trained_models_stress" 
DUAL_BRANCH_V4_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_v4_hybrid_loss.pth")
DYNAMIC_ONLY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_dynamic_only.pth")
STATIC_ONLY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_static_only.pth")

STATS_FULL_PATH = os.path.join(SCRIPT_DIR, "normalization_stats_stress.npz")
STATS_STATIC_PATH = os.path.join(SCRIPT_DIR, "normalization_stats_static_stress.npz")
STATS_DYNAMIC_PATH = os.path.join(SCRIPT_DIR, "normalization_stats_dynamic_stress.npz")

OUTPUT_DIR = "evaluation_results_stress" 

STATIC_FEATURES = 17 
GRID_RESOLUTION = (64, 64)
OUTPUT_FEATURES = GRID_RESOLUTION[0] * GRID_RESOLUTION[1]
TRAIN_VAL_SPLIT_RATIO = 0.9 
RANDOM_SEED = 42


# --- 1. Model & Dataset Definitions ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class AblationDataset(Dataset):
    def __init__(self, npz_file_list, full_transform, static_transform, dynamic_transform):
        self.file_list, self.full_transform, self.static_transform, self.dynamic_transform = \
            npz_file_list, full_transform, static_transform, dynamic_transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        x_static, x_dynamic = x_full[:STATIC_FEATURES], x_full[STATIC_FEATURES:]
        x_full_transformed = self.full_transform(x_full)
        x_static_transformed = self.static_transform(x_static)
        x_dynamic_transformed = self.dynamic_transform(x_dynamic)
        return x_full_transformed, x_static_transformed, x_dynamic_transformed, y, os.path.basename(filepath)

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

class DynamicOnlyModel(nn.Module):
    def __init__(self, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DynamicOnlyModel, self).__init__()
        self.d_model = d_model
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )
    def forward(self, x):
        x_dynamic = x.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        dynamic_out = dynamic_transformed.mean(dim=1)
        output = self.prediction_head(dynamic_out)
        return output

class StaticOnlyModel(nn.Module):
    def __init__(self, static_size, output_size, dropout=0.1):
        super(StaticOnlyModel, self).__init__()
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU()
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(32, 1024), nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )
    def forward(self, x):
        static_out = self.static_branch(x)
        output = self.prediction_head(static_out)
        return output

# --- 2. Main Execution Block ---

def main(args):
    """Main function to load models and run comparison visualization."""
    print("==========================================================")
    print("      Ablation Model Visual Comparison (Stress V4)      ")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Paths ---
    dataset_full_path = os.path.join(SCRIPT_DIR, DATASET_DIR)
    output_dir = os.path.join(SCRIPT_DIR, "..", OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load data files ---
    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    print(f"Found {len(all_files)} total samples. Using {len(val_files)} for validation.")

    # --- Load all 3 Normalization Stats ---
    print("\nLoading pre-calculated normalization stats...")
    try:
        stats_full = np.load(STATS_FULL_PATH)
        full_transform = NormalizeTransform(torch.from_numpy(stats_full['mean']), torch.from_numpy(stats_full['std']))
        
        stats_static = np.load(STATS_STATIC_PATH)
        static_transform = NormalizeTransform(torch.from_numpy(stats_static['mean']), torch.from_numpy(stats_static['std']))
        
        stats_dynamic = np.load(STATS_DYNAMIC_PATH)
        dynamic_transform = NormalizeTransform(torch.from_numpy(stats_dynamic['mean']), torch.from_numpy(stats_dynamic['std']))
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load normalization stats file. {e}")
        return
    print("All 3 normalization transforms loaded.")

    # --- Calculate y data range for SSIM ---
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")

    # --- Create Evaluation Dataset ---
    val_dataset = AblationDataset(val_files, full_transform, static_transform, dynamic_transform)

    # --- Load all 3 Models ---
    print("\nLoading all 3 models...")
    with np.load(all_files[0]) as f:
        dynamic_size = f['x'][STATIC_FEATURES:].shape[0]
    
    try:
        full_model = DualBranchModel(STATIC_FEATURES, dynamic_size, OUTPUT_FEATURES).to(device)
        full_model.load_state_dict(torch.load(DUAL_BRANCH_V4_MODEL_PATH, map_location=device))
        full_model.eval()
        print(f"Loaded Full V4 Model: {DUAL_BRANCH_V4_MODEL_PATH}")

        dynamic_model = DynamicOnlyModel(dynamic_size, OUTPUT_FEATURES).to(device)
        dynamic_model.load_state_dict(torch.load(DYNAMIC_ONLY_MODEL_PATH, map_location=device))
        dynamic_model.eval()
        print(f"Loaded Dynamic-Only Model: {DYNAMIC_ONLY_MODEL_PATH}")

        static_model = StaticOnlyModel(STATIC_FEATURES, OUTPUT_FEATURES).to(device)
        static_model.load_state_dict(torch.load(STATIC_ONLY_MODEL_PATH, map_location=device))
        static_model.eval()
        print(f"Loaded Static-Only Model: {STATIC_ONLY_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not load a model file. {e}")
        return
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Model architecture mismatch. {e}")
        return

    # --- Initialize Metrics (for plotting) ---
    mse_metric = torchmetrics.MeanSquaredError().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)

    # --- Loop and Plot ---
    num_samples = args.num_samples
    if len(val_dataset) < num_samples:
        num_samples = len(val_dataset)
    
    indices_to_plot = np.random.choice(len(val_dataset), num_samples, replace=False)
    print(f"\nGenerating {num_samples} comparison plots...")

    with torch.no_grad():
        for idx in tqdm(indices_to_plot, desc="Generating comparison plots"):
            x_full, x_static, x_dynamic, y_truth, fname = val_dataset[idx]
            
            x_full, x_static, x_dynamic, y_truth = \
                x_full.unsqueeze(0).to(device), \
                x_static.unsqueeze(0).to(device), \
                x_dynamic.unsqueeze(0).to(device), \
                y_truth.unsqueeze(0).to(device)

            pred_full = full_model(x_full)
            pred_static = static_model(x_static)
            pred_dynamic = dynamic_model(x_dynamic)
            
            y_truth_grid = y_truth.reshape(1, 1, *GRID_RESOLUTION)
            pred_full_grid = pred_full.reshape(1, 1, *GRID_RESOLUTION)
            pred_static_grid = pred_static.reshape(1, 1, *GRID_RESOLUTION)
            pred_dynamic_grid = pred_dynamic.reshape(1, 1, *GRID_RESOLUTION)
            
            ssim_full = ssim_metric(pred_full_grid, y_truth_grid).item()
            ssim_static = ssim_metric(pred_static_grid, y_truth_grid).item()
            ssim_dynamic = ssim_metric(pred_dynamic_grid, y_truth_grid).item()
            
            mse_full = mse_metric(pred_full, y_truth).item()
            mse_static = mse_metric(pred_static, y_truth).item()
            mse_dynamic = mse_metric(pred_dynamic, y_truth).item()

            # --- [MODIFIED] Plotting Block ---
            
            fig, axes = plt.subplots(1, 4, figsize=(28, 8)) 
            fig.suptitle(f"Ablation Model Comparison: {fname.replace('.npz', '')}", fontsize=16)

            vmin = y_truth_grid.min().item()
            vmax = y_truth_grid.max().item()

            # Helper function to plot
            def plot_img(ax, data, title, ssim_val, mse_val):
                data_grid = data.cpu().numpy().reshape(GRID_RESOLUTION)
                data_grid = data_grid.T # <-- [FIX V5] Use Transpose
                im = ax.imshow(data_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nSSIM: {ssim_val:.4f} | MSE: {mse_val:.6f}", fontsize=10)
                ax.axis('off')
                return im
            
            # Plot 1: Ground Truth
            data_grid_truth = y_truth_grid.cpu().numpy().reshape(GRID_RESOLUTION)
            data_grid_truth = data_grid_truth.T # <-- [FIX V5] Use Transpose
            im = axes[0].imshow(data_grid_truth, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            axes[0].set_title("Ground Truth", fontsize=10)
            axes[0].axis('off')

            plot_img(axes[1], pred_full_grid, "Full Model (V4)", ssim_full, mse_full)
            plot_img(axes[2], pred_dynamic_grid, "Dynamic-Only", ssim_dynamic, mse_dynamic)
            plot_img(axes[3], pred_static_grid, "Static-Only", ssim_static, mse_static)

            plt.tight_layout(rect=[0, 0.03, 0.95, 0.95]) 
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.04)
            
            save_path = os.path.join(output_dir, f"comparison_plot_{fname.replace('.npz', '.png')}")
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
            # --- [END MODIFIED] ---

    print(f"\nFinished generating {num_samples} comparison plots in '{output_dir}'.")
    print("==========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ablation Model (Stress-Based) comparison.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)