# -*- coding: utf-8 -*-
"""
Script to compare the predictions of all three ABLATION models
for the SUBSIDENCE dataset.

V3 (Final):
- Loads the *correct* (newly trained) DualBranchModel for subsidence.
- Loads all 3 models (Full, Static-Only, Subsidence-Only).
- Correctly calculates all 3 separate normalization statistics.
- Plots all 4 images side-by-side with 'jet' colormap and correct orientation.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics # <-- 导入 torchmetrics
import argparse

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(SCRIPT_DIR, "../final_dataset")
MODELS_DIR = os.path.join(SCRIPT_DIR, "trained_models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../evaluation_results")

# --- [MODIFIED] Paths to all 3 models ---
DUAL_BRANCH_MODEL_PATH = os.path.join(MODELS_DIR, "best_subsidence_model_dual_branch.pth") # <-- 新的V4模型
SUBSIDENCE_ONLY_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_subsidence_only.pth")
STATIC_ONLY_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_static_only.pth")
# --- [END MODIFIED] ---

NUM_PARAMS = 11 # <-- 沉降模型的静态参数数量
GRID_RESOLUTION = (64, 64)
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42

# --- 1. Model & Dataset Definitions ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class AblationDataset(Dataset):
    """Dataset modified to return all three input types + filename."""
    def __init__(self, npz_file_list, full_transform, static_transform, subsidence_transform):
        self.file_list = npz_file_list
        self.full_transform = full_transform
        self.static_transform = static_transform
        self.subsidence_transform = subsidence_transform

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        
        x_static = x_full[:NUM_PARAMS]
        x_subsidence = x_full[NUM_PARAMS:]

        x_full_transformed = self.full_transform(x_full)
        x_static_transformed = self.static_transform(x_static)
        x_subsidence_transformed = self.subsidence_transform(x_subsidence)
            
        return x_full_transformed, x_static_transformed, x_subsidence_transformed, y, os.path.basename(filepath)

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
    """ (V1/V4 Full Model Arch) """
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

class SubsidenceOnlyModel(nn.Module):
    """ (Dynamic-Only Model) """
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(SubsidenceOnlyModel, self).__init__()
        self.d_model = d_model
        self.feature_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_size) # <-- Note: This differs from stress model
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.output_layer(x)
        return output

class StaticOnlyModel(nn.Module):
    """ (Static-Only Model) """
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

# --- 2. Plotting Function ---

def create_comparison_plot(sample, models, metrics, device, index, fname):
    """Generates and saves a comparison plot for a single sample."""
    
    full_model, static_model, subsidence_only_model = models
    mse_metric, ssim_metric = metrics
    x_full_transformed, x_static_transformed, x_subsidence_transformed, y_truth = sample

    # Add batch dim and send to device
    x_full_transformed = x_full_transformed.unsqueeze(0).to(device)
    x_static_transformed = x_static_transformed.unsqueeze(0).to(device)
    x_subsidence_transformed = x_subsidence_transformed.unsqueeze(0).to(device)
    y_truth = y_truth.unsqueeze(0).to(device) # Also send y_truth to device

    # --- Generate Predictions ---
    with torch.no_grad():
        pred_full_model = full_model(x_full_transformed)
        pred_static_model = static_model(x_static_transformed)
        pred_subsidence_only = subsidence_only_model(x_subsidence_transformed)

    # --- Reshape for metrics and plotting ---
    y_truth_grid_img = y_truth.reshape(1, 1, *GRID_RESOLUTION)
    pred_full_grid_img = pred_full_model.reshape(1, 1, *GRID_RESOLUTION)
    pred_static_grid_img = pred_static_model.reshape(1, 1, *GRID_RESOLUTION)
    pred_subsidence_grid_img = pred_subsidence_only.reshape(1, 1, *GRID_RESOLUTION)
    
    # --- Calculate metrics for this sample ---
    ssim_full = ssim_metric(pred_full_grid_img, y_truth_grid_img).item()
    ssim_static = ssim_metric(pred_static_grid_img, y_truth_grid_img).item()
    ssim_dynamic = ssim_metric(pred_subsidence_grid_img, y_truth_grid_img).item()
    
    mse_full = mse_metric(pred_full_model, y_truth).item()
    mse_static = mse_metric(pred_static_model, y_truth).item()
    mse_dynamic = mse_metric(pred_subsidence_only, y_truth).item()

    # --- Get Grids for Plotting (and apply .T transpose) ---
    y_truth_grid = y_truth_grid_img.cpu().numpy().reshape(GRID_RESOLUTION).T
    pred_full_grid = pred_full_grid_img.cpu().numpy().reshape(GRID_RESOLUTION).T
    pred_static_grid = pred_static_grid_img.cpu().numpy().reshape(GRID_RESOLUTION).T
    pred_subsidence_grid = pred_subsidence_grid_img.cpu().numpy().reshape(GRID_RESOLUTION).T

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 8))
    
    vmin = y_truth_grid.min()
    vmax = y_truth_grid.max()
    
    # Plot 1: Ground Truth
    im1 = axes[0].imshow(y_truth_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth', fontsize=16)
    
    # Plot 2: Full Model Prediction
    axes[1].imshow(pred_full_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Full Model (V4)\nSSIM: {ssim_full:.4f} | MSE: {mse_full:.6f}", fontsize=10)

    # Plot 3: Static-Only Model Prediction
    axes[2].imshow(pred_static_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Static-Only\nSSIM: {ssim_static:.4f} | MSE: {mse_static:.6f}", fontsize=10)

    # Plot 4: Subsidence-Only Model Prediction
    axes[3].imshow(pred_subsidence_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    axes[3].set_title(f"Subsidence-Only\nSSIM: {ssim_dynamic:.4f} | MSE: {mse_dynamic:.6f}", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 0.95, 0.9])
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7, pad=0.04)
    fig.suptitle(f'Model Prediction Comparison (Subsidence) for {fname}', fontsize=20)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'prediction_comparison_subsidence_{index}.png')
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f" -> Comparison plot saved to {save_path}")

# --- 3. Main Execution Block ---

def main(args):
    """Main function to load models and run comparison for multiple samples."""
    print("======================================================")
    print("     Ablation Model Visual Comparison (Subsidence)    ")
    print("======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data files ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print(f"FATAL ERROR: No .npz files found in '{DATASET_DIR}'")
        return
        
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    
    TRAIN_VAL_SPLIT_RATIO = 0.9
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    print(f"Found {len(all_files)} files. Using {len(val_files)} for validation.")
    print(f"Plotting {args.num_samples} random validation samples.")

    # --- [MODIFIED] Create ALL 3 Normalization Transforms ---
    print("\nCalculating normalization stats from training set...")
    
    # 1. Full stats
    stats_full_path = os.path.join(SCRIPT_DIR, "normalization_stats_subsidence_full.npz")
    try:
        stats = np.load(stats_full_path)
        mean_full, std_full = torch.from_numpy(stats['mean']), torch.from_numpy(stats['std'])
        print(f"Loaded full stats from {stats_full_path}")
    except FileNotFoundError:
        print(f"Calculating full stats...")
        all_x_full_train = [torch.from_numpy(np.load(f)['x']) for f in train_files]
        x_full_tensor_train = torch.stack(all_x_full_train, dim=0).float()
        mean_full, std_full = x_full_tensor_train.mean(dim=0), x_full_tensor_train.std(dim=0)
        np.savez(stats_full_path, mean=mean_full.numpy(), std=std_full.numpy())
        print(f"Saved full stats to {stats_full_path}")
    full_transform = NormalizeTransform(mean_full, std_full)
    
    # 2. Static stats
    # (These stats are small, we can recalculate or load if they exist)
    x_static_tensor_train = torch.stack([torch.from_numpy(np.load(f)['x'])[:NUM_PARAMS] for f in train_files]).float()
    mean_static, std_static = x_static_tensor_train.mean(dim=0), x_static_tensor_train.std(dim=0)
    static_transform = NormalizeTransform(mean_static, std_static)

    # 3. Subsidence stats
    x_dynamic_tensor_train = torch.stack([torch.from_numpy(np.load(f)['x'])[NUM_PARAMS:] for f in train_files]).float()
    mean_dynamic, std_dynamic = x_dynamic_tensor_train.mean(dim=0), x_dynamic_tensor_train.std(dim=0)
    subsidence_transform = NormalizeTransform(mean_dynamic, std_dynamic)
    print("Normalization stats calculated for all 3 data types.")
    # --- [END MODIFIED] ---

    # --- Calculate y data range for SSIM ---
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")

    # --- Create Ablation Dataset ---
    val_dataset = AblationDataset(val_files, full_transform, static_transform, subsidence_transform)

    # --- [MODIFIED] Load ALL 3 Models ---
    print("\nLoading all 3 models...")
    dynamic_size = x_dynamic_tensor_train.shape[1]
    output_size = GRID_RESOLUTION[0]*GRID_RESOLUTION[1]

    try:
        # 1. Full Model (DualBranch)
        full_model = DualBranchModel(static_size=NUM_PARAMS, dynamic_size=dynamic_size, output_size=output_size).to(device)
        full_model.load_state_dict(torch.load(DUAL_BRANCH_MODEL_PATH, map_location=device))
        full_model.eval()
        print(f"Full model loaded: {DUAL_BRANCH_MODEL_PATH}")

        # 2. Static-Only Model
        static_model = StaticOnlyModel(static_size=NUM_PARAMS, output_size=output_size).to(device)
        static_model.load_state_dict(torch.load(STATIC_ONLY_MODEL_PATH, map_location=device))
        static_model.eval()
        print(f"Static-only model loaded: {STATIC_ONLY_MODEL_PATH}")

        # 3. Subsidence-only Model
        subsidence_only_model = SubsidenceOnlyModel(input_size=dynamic_size, output_size=output_size).to(device)
        subsidence_only_model.load_state_dict(torch.load(SUBSIDENCE_ONLY_MODEL_PATH, map_location=device))
        subsidence_only_model.eval()
        print(f"Subsidence-only model loaded: {SUBSIDENCE_ONLY_MODEL_PATH}")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not load a model file. {e}")
        return
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Model architecture mismatch. {e}")
        return
    # --- [END MODIFIED] ---
    
    models = (full_model, static_model, subsidence_only_model)
    metrics = (torchmetrics.MeanSquaredError().to(device),
               torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device))
    
    # --- Generate Plots for Multiple Samples ---
    print(f"\nGenerating {args.num_samples} comparison plots...")
    
    if len(val_dataset) < args.num_samples:
        print(f"Warning: Requested {args.num_samples}, but only {len(val_dataset)} in validation set. Plotting all.")
        args.num_samples = len(val_dataset)

    indices_to_plot = np.random.choice(len(val_dataset), args.num_samples, replace=False)
    
    for i, idx in enumerate(tqdm(indices_to_plot, desc="Generating Plots")):
        x_full_t, x_static_t, x_sub_t, y_truth, fname = val_dataset[idx]
        sample_data = (x_full_t, x_static_t, x_sub_t, y_truth)
        
        create_comparison_plot(sample_data, models, metrics, device, i + 1, fname)

    print("\n======================================================")
    print("                       Done!                        ")
    print("======================================================")

if __name__ == "__main__":
    # --- [MODIFIED] Added argparse ---
    parser = argparse.ArgumentParser(description="Visualize Ablation Model (Subsidence-Based) comparison.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
    # --- [END MODIFIED] ---