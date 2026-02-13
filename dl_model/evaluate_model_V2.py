# -*- coding: utf-8 -*-
"""
Model Evaluation Script (Enhanced for Statistical Analysis)

This script loads a trained surrogate model and evaluates its performance on the
validation set. It provides both quantitative metrics and visual comparisons.

Evaluation includes:
1.  Quantitative Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE),
    and Structural Similarity Index (SSIM).
2.  Visual Comparison: A 3-panel plot showing the Ground Truth, Model Prediction,
    and the Difference Map for a few sample cases.
3.  Statistical Hypothesis Testing (when using --all):
    -   Calculates and plots the MSE distribution.
    -   Performs a log-normality test on the MSE distribution.
    -   Calculates statistics on "inlier" data vs "outlier" data.

How to run:
# For a few visual samples with corrected orientation
python dl_model/evaluate_model.py --model_path trained_models/best_dual_branch_model.pth --num_samples 5

# For a full quantitative evaluation AND hypothesis testing on the entire validation set
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
    print("\nWARNING: scikit-image not found. SSIM metric will be skipped.")
    print("Please install it: pip install scikit-image\n")

# Attempt to import scipy
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("\nWARNING: scipy not found. Statistical tests (log-normality) will be skipped.")
    print("Please install it: pip install scipy\n")


# --- Configuration ---

# Paths (relative to the project root)
DATASET_DIR = "final_dataset"
DEFAULT_MODEL_PATH = "trained_models/best_dual_branch_model.pth"
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
    print("               Model Evaluation Script                ")
    print("======================================================")

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
        stats_file = np.load("dl_model/normalization_stats.npz")
        mean = torch.from_numpy(stats_file['mean'])
        std = torch.from_numpy(stats_file['std'])
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

    if args.all:
        # --- MODE 1: Full Quantitative Evaluation & Hypothesis Testing ---
        print(f"\n--- Evaluating all {len(val_dataset)} samples for statistical analysis ---")
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        all_sample_mses = []
        all_sample_maes = []
        all_sample_ssims = []
        
        # 使用 'none' 来获取每个样本的损失
        criterion_mse_per_sample = nn.MSELoss(reduction='none').to(device)
        criterion_mae_per_sample = nn.L1Loss(reduction='none').to(device)
        
        with torch.no_grad():
            for inputs, targets, fnames in tqdm(val_loader, desc="Evaluating all samples"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # --- Quantitative Evaluation (per-sample) ---
                # 1. MSE (per-sample)
                # (B, 4096) -> (B,)
                mse_samples = criterion_mse_per_sample(outputs, targets).mean(dim=1)
                all_sample_mses.extend(mse_samples.cpu().numpy())

                # 2. MAE (per-sample)
                # (B, 4096) -> (B,)
                mae_samples = criterion_mae_per_sample(outputs, targets).mean(dim=1)
                all_sample_maes.extend(mae_samples.cpu().numpy())

                # 3. SSIM (per-sample)
                if SKIMAGE_AVAILABLE:
                    target_grids = targets.cpu().numpy().reshape(-1, 64, 64)
                    output_grids = outputs.cpu().numpy().reshape(-1, 64, 64)
                    for j in range(target_grids.shape[0]):
                        target_grid = target_grids[j]
                        output_grid = output_grids[j]
                        data_range = max(target_grid.max() - target_grid.min(), 1e-8) # 避免除以零
                        ssim_val = ssim(target_grid, output_grid, data_range=data_range)
                        all_sample_ssims.append(ssim_val)
        
        # --- Post-loop Analysis ---
        mse_array = np.array(all_sample_mses)
        mae_array = np.array(all_sample_maes)
        samples_processed = len(mse_array)

        # --- Original Summary ---
        avg_mse = mse_array.mean()
        avg_mae = mae_array.mean()
        print("======================================================")
        print("                 Evaluation Summary                   ")
        print("======================================================")
        print(f"Average over {samples_processed} samples:")
        print(f"   - Average MSE : {avg_mse:.6f}")
        print(f"   - Average MAE : {avg_mae:.6f}")
        if SKIMAGE_AVAILABLE:
            avg_ssim = np.array(all_sample_ssims).mean()
            print(f"   - Average SSIM: {avg_ssim:.6f}")
        
# --- NEW: Hypothesis Verification Section ---
        print("\n======================================================")
        print("         Hypothesis Verification (MSE Dist.)          ")
        print("======================================================")

        # --- [Hypothesis 1] Log-Normal Distribution ---
        print("\n--- [Hypothesis 1] Log-Normal Distribution Test ---")
        
        # 1a. Plot Histogram
        # 设置 Matplotlib 支持中文和英文/数字字体
        plt.rcParams['font.sans-serif'] = ['SimSun'] # 中文字体设置为宋体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        plt.rcParams['font.size'] = 14 # 默认字体大小稍微调大
        plt.rcParams['mathtext.fontset'] = 'cm' # 使用Computer Modern字体，更接近Times New Roman风格

        fig, ax1 = plt.subplots(figsize=(12, 6)) # 稍微加大图幅

        # 绘制主要直方图 (MSE Frequency)
        n_bins = 100 # 增加一些 bins 以更平滑显示
        ax1.hist(mse_array, bins=n_bins, alpha=0.7, label="MSE 频率", color='#1f77b4') # 蓝色
        
        # 设置主X轴（MSE）
        ax1.set_xlabel("均方误差 (MSE)", fontproperties='SimSun', fontsize=16)
        ax1.set_ylabel("频率", fontproperties='SimSun', fontsize=16)
        ax1.set_title(f"MSE 分布 (样本数={samples_processed})", fontproperties='SimSun', fontsize=18)
        
        # 网格线
        ax1.grid(True, linestyle='--', alpha=0.5, color='lightgray')

        # 设置次X轴（对数尺度MSE）
        ax2 = ax1.twiny()
        # 计算对数区间的 bin，用于在对数轴上绘制直方图，以展现其近似正态的形态
        # 避免 log(0) 或 log(负数)
        min_log = np.log10(np.maximum(mse_array.min(), 1e-10)) # 用np.maximum确保最小值为正
        max_log = np.log10(mse_array.max())
        # 在对数空间均匀分布的 bins，然后在原始尺度上查看其效果
        log_bins = np.logspace(min_log, max_log, n_bins) 
        
        # 绘制第二个直方图 (Log-Scale Bins)，使用相同的 MSE 数据，但在对数尺度的X轴上绘制
        # 这里实际上是在对数轴上重新分箱，以便观察对数正态的形态
        ax2.hist(mse_array, bins=log_bins, alpha=0.5, color='#ff7f0e', label="MSE 频率 (对数轴)") # 橙色
        ax2.set_xscale('log') # 设置次X轴为对数尺度
        ax2.set_xlabel("均方误差 (MSE) 对数尺度", fontproperties='SimSun', fontsize=16)
        
        # 设置X轴刻度标签的字体为 Times New Roman
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(12)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(12)
        
        # 设置主、次X轴的刻度线样式（Times New Roman）
        ax1.tick_params(axis='x', labelsize=12, labelcolor='black', length=4)
        ax1.tick_params(axis='y', labelsize=12, labelcolor='black', length=4)
        ax2.tick_params(axis='x', labelsize=12, labelcolor='black', length=4)


        # 设置图例
        # 为确保图例中文字体是宋体，数字和英文是 Times New Roman
        legend1 = ax1.legend(loc='upper left', prop={'family': 'SimSun', 'size': 12})
        for text in legend1.get_texts():
            text.set_fontfamily('SimSun') # 确保标签文本是宋体
        
        legend2 = ax2.legend(loc='upper right', prop={'family': 'SimSun', 'size': 12})
        for text in legend2.get_texts():
            text.set_fontfamily('SimSun') # 确保标签文本是宋体


        plt.tight_layout() # 调整布局，防止重叠
        
        hist_path = os.path.join(args.output_dir, "mse_distribution_histogram.png")
        plt.savefig(hist_path, dpi=300) # 保存更高分辨率
        plt.close(fig)
        print(f"  Saved MSE distribution histogram to {hist_path}")

        # 1b. Log-Normality Test (Shapiro-Wilk on log(MSE))
        if SCIPY_AVAILABLE:
            log_mse = np.log(mse_array + 1e-20) # 添加小epsilon避免 log(0)
            shapiro_stat, shapiro_p = stats.shapiro(log_mse)
            print(f"  Shapiro-Wilk Test (on log(MSE)):")
            print(f"    Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4g}")
            if shapiro_p > 0.05:
                print("  Conclusion: The log-transformed MSE data *is* normally distributed (p > 0.05).")
                print("  >> Hypothesis 1 (Log-Normal distribution) is plausible.")
            else:
                print("  Conclusion: The log-transformed MSE data is *not* normally distributed (p <= 0.05).")
                print("  >> Hypothesis 1 (Log-Normal distribution) is rejected by this test.")
        else:
            print("  Skipped log-normality test (scipy not found).")
        
        # --- [Hypothesis 2] Outlier Observation ---
        threshold = 0.004
        count_below_threshold = np.sum(mse_array < threshold)
        percent_below_threshold = (count_below_threshold / samples_processed) * 100
        print(f"\n--- [Hypothesis 2] Outlier Observation (Threshold MSE < {threshold}) ---")
        print(f"  {count_below_threshold} / {samples_processed} samples ({percent_below_threshold:.2f}%) have MSE < {threshold}.")
        print(f"  >> Hypothesis 2 (obs): ~87% of samples < 0.004. (Observed: {percent_below_threshold:.2f}%)")

        # --- [Hypothesis 3] Statistics After Removing Outliers ---
        inliers_mse = mse_array[mse_array < threshold]
        print(f"\n--- [Hypothesis 3] Statistics After Removing Outliers (MSE >= {threshold}) ---")
        if len(inliers_mse) > 0:
            inlier_std_dev = np.std(inliers_mse)
            inlier_mean = np.mean(inliers_mse)
            print(f"  Samples remaining (inliers): {len(inliers_mse)}")
            print(f"  Inlier Mean MSE: {inlier_mean:.6f}")
            print(f"  Inlier Std. Dev. (σ): {inlier_std_dev:.6f}")
            print(f"  >> Hypothesis 3: σ ≈ 0.0009. (Observed: {inlier_std_dev:.6f})")
        else:
            print(f"  No samples found below threshold {threshold}, cannot calculate inlier stats.")

        print("======================================================")

    else:
        # --- MODE 2: Visual Sample Evaluation ---
        print(f"\n--- Evaluating {args.num_samples} random samples from validation set ---")
        # 确保使用 shuffle=True 和 batch_size=1 来获取随机样本
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        # 使用标准的 'mean' reduction 进行单一样本评估
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()
        
        total_mse, total_mae, total_ssim = 0, 0, 0
        samples_processed = 0

        with torch.no_grad():
            for i, (inputs, targets, fnames) in enumerate(val_loader):
                if i >= args.num_samples:
                    break
                
                samples_processed += 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                mse = criterion_mse(outputs, targets).item()
                mae = criterion_mae(outputs, targets).item()
                total_mse += mse
                total_mae += mae
                
                # Reshape (B, 4096) -> (64, 64) since B=1
                target_grid = targets.cpu().numpy().reshape(64, 64)
                output_grid = outputs.cpu().numpy().reshape(64, 64)
                fname = fnames[0]
                
                ssim_val = 0
                if SKIMAGE_AVAILABLE:
                    data_range = max(target_grid.max() - target_grid.min(), 1e-8)
                    ssim_val = ssim(target_grid, output_grid, data_range=data_range)
                    total_ssim += ssim_val

                # --- Visualization (Rotated) ---
                target_grid = np.rot90(target_grid)
                output_grid = np.rot90(output_grid)

                print(f"\nSample {i+1}/{args.num_samples} (File: {fname}):")
                print(f"   - MSE : {mse:.6f}")
                print(f"   - MAE : {mae:.6f}")
                if SKIMAGE_AVAILABLE: print(f"   - SSIM: {ssim_val:.6f}")
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Sample: {fname}\n MSE: {mse:.4f} | MAE: {mae:.4f} | SSIM: {ssim_val:.4f}', fontsize=16)

                im1 = axes[0].imshow(target_grid, cmap='jet', interpolation='nearest')
                axes[0].set_title('Ground Truth (Rotated)')
                fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                im2 = axes[1].imshow(output_grid, cmap='jet', interpolation='nearest')
                axes[1].set_title('Model Prediction (Rotated)')
                # ********************
                # * 代码修正点   *
                # ********************
                # 将 ax=axes[0] 修改为 ax=axes[1]
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
                print(f"   - Saved visualization to {save_path}")

        # --- Final Summary (for sample mode) ---
        avg_mse = total_mse / samples_processed
        avg_mae = total_mae / samples_processed
        
        print("======================================================")
        print(f"       Evaluation Summary ({samples_processed} Samples)        ")
        print("======================================================")
        print(f"   - Average MSE : {avg_mse:.6f}")
        print(f"   - Average MAE : {avg_mae:.6f}")
        if SKIMAGE_AVAILABLE:
            avg_ssim = total_ssim / samples_processed
            print(f"   - Average SSIM: {avg_ssim:.6f}")
        print("======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained surrogate model with statistical analysis.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model .pth file.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to visualize (ignored if --all is used).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation plots.")
    parser.add_argument("--all", action="store_true", help="Evaluate on all samples in the validation set (this enables statistical analysis).")
    
    parsed_args = parser.parse_args()
    evaluate(parsed_args)