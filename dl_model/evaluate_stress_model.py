# -*- coding: utf-8 -*-
"""
Model Evaluation Script (Stress-Based V4 Model)

This script loads the single trained (V4) surrogate model and evaluates 
its performance on the validation set.

V6: 
- Reverted script to SINGLE model evaluation (V4 Full Model).
- Corrected default model path to 'best_stress_model_v4_hybrid_loss.pth'.
- Corrected all file paths to be relative to the script location.
- Corrected plotting to use 'jet' cmap, 'origin=lower', and .T transpose.
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
import torchmetrics # 导入 torchmetrics

# Attempt to import scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Attempt to import scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# --- Configuration ---

SCRIPT_DIR = os.path.dirname(__file__)

# Paths (relative to the project root, constructed from script dir)
DATASET_DIR = os.path.join(SCRIPT_DIR, "../final_dataset_stress")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../evaluation_results_stress")
STATS_PATH = os.path.join(SCRIPT_DIR, "normalization_stats_stress.npz") # V1/V4 模型的归一化文件

# --- [MODIFIED] Default model path is now V4 ---
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "../trained_models_stress/best_stress_model_v4_hybrid_loss.pth") 
# --- [END MODIFIED] ---

# Data handling
STATIC_FEATURES = 17
GRID_RESOLUTION = (64, 64)
TRAIN_VAL_SPLIT_RATIO = 0.9 
RANDOM_SEED = 42

# --- Model Definitions (V1/V4 Arch) ---

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
    print("        Stress Model V4 Evaluation Script           ")
    print("======================================================")

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
    
    stats_file_path = STATS_PATH 
    if os.path.exists(stats_file_path):
        print(f"Loading pre-calculated normalization stats from {stats_file_path}.")
        stats_file = np.load(stats_file_path)
        mean = torch.from_numpy(stats_file['mean'])
        std = torch.from_numpy(stats_file['std'])
    else:
        print("Calculating from scratch... (This may take a moment)")
        stats_dataset = FractureDataset(train_files, transform=None)
        stats_loader = DataLoader(stats_dataset, batch_size=args.batch_size, shuffle=False)
        all_x = [x for x, _, _ in tqdm(stats_loader, desc="Loading data for stats")]
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0)
        std = x_tensor.std(dim=0)
        np.savez(stats_file_path, mean=mean.numpy(), std=std.numpy())
        print(f"Saved normalization stats to {stats_file_path}")

    transform = NormalizeTransform(mean, std)

    # --- Calculate y data range for SSIM ---
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")
    
    # --- Load Model ---
    with np.load(val_files[0]) as first_sample:
        total_input_features = first_sample['x'].shape[0]
    
    dynamic_features = total_input_features - STATIC_FEATURES
    print(f"Model Features: {total_input_features} (Static: {STATIC_FEATURES}, Dynamic: {dynamic_features})")

    model = DualBranchModel(
        static_size=STATIC_FEATURES,
        dynamic_size=dynamic_features,
        output_size=64*64
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{args.model_path}'")
        return
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model. {e}")
        return
        
    model.eval()
    print("Model loaded successfully.")

    # --- Perform Evaluation on Validation Set ---
    val_dataset = FractureDataset(val_files, transform=transform)

    if args.all:
        # --- MODE 1: Full Quantitative Evaluation & Hypothesis Testing ---
        print(f"\n--- Evaluating all {len(val_dataset)} samples for statistical analysis ---")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # --- Use torchmetrics for consistency ---
        mse_metric = torchmetrics.MeanSquaredError().to(device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(device)
        ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
        pcc_metric = torchmetrics.PearsonCorrCoef().to(device)
        # ----------------------------------------
        
        all_sample_mses = [] # For histogram
        
        with torch.no_grad():
            for inputs, targets, fnames in tqdm(val_loader, desc="Evaluating all samples"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # For histogram
                all_sample_mses.extend(nn.MSELoss(reduction='none')(outputs, targets).mean(dim=1).cpu().numpy())
                
                # Reshape for metrics
                targets_img = targets.reshape(-1, 1, *GRID_RESOLUTION)
                outputs_img = outputs.reshape(-1, 1, *GRID_RESOLUTION)

                # Update metrics
                mse_metric.update(outputs, targets)
                mae_metric.update(outputs, targets)
                ssim_metric.update(outputs_img, targets_img)
                pcc_metric.update(outputs.flatten(), targets.flatten())
        
        # --- Compute final metrics ---
        avg_mse = mse_metric.compute().item()
        avg_mae = mae_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        avg_pcc = pcc_metric.compute().item()
        
        mse_array = np.array(all_sample_mses)
        samples_processed = len(mse_array)

        print("======================================================")
        print("                Evaluation Summary                  ")
        print("======================================================")
        print(f"Average over {samples_processed} samples:")
        print(f"   - Average MSE : {avg_mse:.6f}")
        print(f"   - Average MAE : {avg_mae:.6f}")
        print(f"   - Average SSIM: {avg_ssim:.6f}")
        print(f"   - Average PCC : {avg_pcc:.6f}")
        
        # --- Hypothesis Verification Section ---
        print("\n======================================================")
        print("        Hypothesis Verification (MSE Dist.)         ")
        print("======================================================")
        print("\n--- [Hypothesis 1] Log-Normal Distribution Test ---")
        
        plt.rcParams['font.sans-serif'] = ['SimSun'] 
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 14
        plt.rcParams['mathtext.fontset'] = 'cm' 
        fig, ax1 = plt.subplots(figsize=(12, 6))
        n_bins = 100 
        ax1.hist(mse_array, bins=n_bins, alpha=0.7, label="MSE 频率", color='#1f77b4')
        ax1.set_xlabel("均方误差 (MSE)", fontproperties='SimSun', fontsize=16)
        ax1.set_ylabel("频率", fontproperties='SimSun', fontsize=16)
        ax1.set_title(f"MSE 分布 (样本数={samples_processed})", fontproperties='SimSun', fontsize=18)
        ax1.grid(True, linestyle='--', alpha=0.5, color='lightgray')
        ax2 = ax1.twiny()
        min_log = np.log10(np.maximum(mse_array.min(), 1e-10))
        max_log = np.log10(mse_array.max())
        log_bins = np.logspace(min_log, max_log, n_bins) 
        ax2.hist(mse_array, bins=log_bins, alpha=0.5, color='#ff7f0e', label="MSE 频率 (对数轴)")
        ax2.set_xscale('log')
        ax2.set_xlabel("均方误差 (MSE) 对数尺度", fontproperties='SimSun', fontsize=16)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily('Times New Roman'); label.set_fontsize(12)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontfamily('Times New Roman'); label.set_fontsize(12)
        ax1.tick_params(axis='x', labelsize=12, labelcolor='black', length=4)
        ax1.tick_params(axis='y', labelsize=12, labelcolor='black', length=4)
        ax2.tick_params(axis='x', labelsize=12, labelcolor='black', length=4)
        legend1 = ax1.legend(loc='upper left', prop={'family': 'SimSun', 'size': 12})
        for text in legend1.get_texts(): text.set_fontfamily('SimSun')
        legend2 = ax2.legend(loc='upper right', prop={'family': 'SimSun', 'size': 12})
        for text in legend2.get_texts(): text.set_fontfamily('SimSun')
        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, "mse_distribution_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close(fig)
        print(f"  Saved MSE distribution histogram to {hist_path}")

        if SCIPY_AVAILABLE:
            log_mse = np.log(mse_array + 1e-20)
            shapiro_stat, shapiro_p = stats.shapiro(log_mse)
            print(f"  Shapiro-Wilk Test (on log(MSE)):")
            print(f"     Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4g}")
            if shapiro_p > 0.05:
                print("  Conclusion: The log-transformed MSE data *is* normally distributed (p > 0.05).")
            else:
                print("  Conclusion: The log-transformed MSE data is *not* normally distributed (p <= 0.05).")
        
        threshold = 0.004
        count_below_threshold = np.sum(mse_array < threshold)
        percent_below_threshold = (count_below_threshold / samples_processed) * 100
        print(f"\n--- Outlier Observation (Threshold MSE < {threshold}) ---")
        print(f"  {count_below_threshold} / {samples_processed} samples ({percent_below_threshold:.2f}%) have MSE < {threshold}.")
        inliers_mse = mse_array[mse_array < threshold]
        if len(inliers_mse) > 0:
            print(f"  Inlier Mean MSE: {np.mean(inliers_mse):.6f}")
            print(f"  Inlier Std. Dev. (σ): {np.std(inliers_mse):.6f}")
        print("======================================================")

    else:
        # --- MODE 2: Visual Sample Evaluation ---
        print(f"\n--- Evaluating {args.num_samples} random samples from validation set ---")
        
        # --- [MODIFIED] Use torchmetrics for plotting ---
        criterion_mse = torchmetrics.MeanSquaredError().to(device)
        criterion_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
        # --- [END MODIFIED] ---
        
        if len(val_dataset) < args.num_samples:
            args.num_samples = len(val_dataset)
        
        indices_to_plot = np.random.choice(len(val_dataset), args.num_samples, replace=False)

        with torch.no_grad():
            for i, idx in enumerate(tqdm(indices_to_plot, desc="Generating visual samples")):
                
                inputs, targets, fname = val_dataset[idx]
                
                inputs_batch = inputs.unsqueeze(0).to(device)
                targets_batch = targets.unsqueeze(0).to(device)
                
                outputs_batch = model(inputs_batch)

                mse = criterion_mse(outputs_batch, targets_batch).item()
                
                # Reshape for SSIM
                targets_img = targets_batch.reshape(1, 1, *GRID_RESOLUTION)
                outputs_img = outputs_batch.reshape(1, 1, *GRID_RESOLUTION)
                ssim_val = criterion_ssim(outputs_img, targets_img).item()
                
                # --- [MODIFIED] Plotting with 'jet' and '.T' ---
                target_grid = targets.cpu().numpy().reshape(GRID_RESOLUTION).T
                output_grid = outputs_batch.cpu().numpy().reshape(GRID_RESOLUTION).T
                
                print(f"\nSample {i+1}/{args.num_samples} (File: {fname}):")
                print(f"   - MSE : {mse:.6f}")
                print(f"   - SSIM: {ssim_val:.6f}")
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1x3 plot
                fig.suptitle(f'Sample: {fname}\n MSE: {mse:.4f} | SSIM: {ssim_val:.4f}', fontsize=16)

                vmin = target_grid.min()
                vmax = target_grid.max()

                im1 = axes[0].imshow(target_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                axes[0].set_title('Ground Truth')
                
                im2 = axes[1].imshow(output_grid, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                axes[1].set_title('Model Prediction (V4)')
                
                diff_map = np.abs(target_grid - output_grid)
                im3 = axes[2].imshow(diff_map, cmap='plasma', origin='lower')
                axes[2].set_title('Absolute Difference')
                
                for ax in axes: ax.axis('off')
                
                # Adjust layout and colorbars
                plt.tight_layout(rect=[0, 0.03, 0.95, 0.94])
                fig.colorbar(im1, ax=axes[0:2], shrink=0.7, pad=0.04)
                fig.colorbar(im3, ax=axes[2], shrink=0.7, pad=0.04)
                # --- [END MODIFIED] ---
                
                save_path = os.path.join(args.output_dir, f"evaluation_sample_v4_{i+1}.png")
                plt.savefig(save_path, dpi=200)
                plt.close(fig)
                print(f"   - Saved visualization to {save_path}")

        print("======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained surrogate model (Stress-Based V4).")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model .pth file.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to visualize (ignored if --all is used).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation plots.")
    parser.add_argument("--all", action="store_true", help="Evaluate on all samples in the validation set (this enables statistical analysis).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for --all evaluation.")
    
    parsed_args = parser.parse_args()
    
    # --- Fix path relativity ---
    parsed_args.output_dir = os.path.abspath(parsed_args.output_dir)
    parsed_args.model_path = os.path.abspath(parsed_args.model_path)
    # --- End fix ---
    
    evaluate(parsed_args)