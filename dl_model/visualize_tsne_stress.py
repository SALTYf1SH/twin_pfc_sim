# -*- coding: utf-8 -*-
"""
Script to visualize the t-SNE of the FULL V4 MODEL's 
conditional vector space (Stress-Based Model).

This script analyzes the internal representation of the 'static_branch'
from the 'best_stress_model_v4_hybrid_loss.pth' model.
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

# Attempt to import sklearn
try:
    from sklearn.manifold import TSNE
except ImportError:
    print("\nFATAL ERROR: scikit-learn not found.")
    print("Please install it: pip install scikit-learn\n")
    exit()


# --- Configuration ---
DATASET_DIR = "../final_dataset_stress"
MODEL_PATH = "trained_models_stress/best_stress_model_v4_hybrid_loss.pth" # <-- 1. 加载V4完整模型
STATS_PATH = "normalization_stats_stress.npz" # <-- 2. 加载V4的 *完整* 归一化文件
OUTPUT_DIR = "evaluation_results_stress"
STATIC_FEATURES = 17 
RANDOM_SEED = 42

# (Corresponds to sandstone_emod in parameter_sampler_hqh.py)
PARAM_TO_COLOR_IDX = 0 
PARAM_TO_COLOR_NAME = "Sandstone Elastic Modulus (emod)"

# --- 1. Dataset and Model Definitions ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class TSNEDataset(Dataset):
    """
    Modified dataset:
    Returns: transformed_full_x (for model), original_static_x (for coloring)
    """
    def __init__(self, npz_file_list, transform=None):
        self.file_list, self.transform = npz_file_list, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            
        x_static_original = x_full[:STATIC_FEATURES]
        x_full_transformed = x_full.clone()
        
        if self.transform:
            x_full_transformed = self.transform(x_full_transformed)
            
        return x_full_transformed, x_static_original

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

# --- [MODIFIED] 3. 加载 V1/V4 完整模型架构 ---
class DualBranchModel(nn.Module):
    """ (This is the V1/V4 Model Arch: Late Fusion + MLP Decoder) """
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
        # The full forward pass (not used by this script)
        x_static, x_dynamic = x[:, :self.static_feature_size], x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        # ... (rest of the pass) ...
        pass

    # --- [NEW] Method to get the conditional vector ---
    def get_conditional_vector(self, x_full):
        """
        Gets the output of the static branch (the 'c' vector)
        by processing *only* the static part of the *full* input vector.
        """
        # We must extract the static part *from the normalized full vector*
        x_static_transformed = x_full[:, :self.static_feature_size]
        return self.static_branch(x_static_transformed)

# --- 2. Main Execution Block ---

def main():
    """Main function to load model, run t-SNE, and plot."""
    print("==========================================================")
    print("      t-SNE Visualization (V4 Full Model Static Branch)   ")
    print("==========================================================")
    
    script_dir = os.path.dirname(__file__)
    model_full_path = os.path.join(script_dir, "..", MODEL_PATH)
    stats_full_path = os.path.join(script_dir, "..", STATS_PATH) # <-- 4. 使用 V4 完整 stats
    dataset_full_path = os.path.join(script_dir, DATASET_DIR)
    output_dir = os.path.join(script_dir, "..", OUTPUT_DIR)
    
    if not os.path.exists(model_full_path):
        print(f"FATAL: Model file not found at '{model_full_path}'")
        return
    if not os.path.exists(stats_full_path):
        print(f"FATAL: Stats file not found at '{stats_full_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    print(f"Found {len(all_files)} total samples.")

    # --- Data Normalization & Splitting (use Validation Set) ---
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    TRAIN_VAL_SPLIT_RATIO = 0.9
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Loading normalization stats...")
    stats = np.load(stats_full_path) # <-- 5. 加载 V4 完整 stats
    mean = torch.from_numpy(stats['mean'])
    std = torch.from_numpy(stats['std'])
    transform = NormalizeTransform(mean, std)
    print(f"Normalization stats loaded from '{stats_full_path}'.")

    val_dataset = TSNEDataset(val_files, transform=transform) # <-- 6. 使用新 Dataset
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"Created validation set with {len(val_dataset)} samples.")

    # --- Model Loading ---
    print("Initializing and loading the V4 FULL model...")
    with np.load(all_files[0]) as f:
        dynamic_size = f['x'][STATIC_FEATURES:].shape[0]
    output_features = 64 * 64
    
    model = DualBranchModel( # <-- 7. 实例化 V4 完整模型
        static_size=STATIC_FEATURES, 
        dynamic_size=dynamic_size,
        output_size=output_features
    ).to(device)
    
    model.load_state_dict(torch.load(model_full_path, map_location=device))
    model.eval()
    print(f"Model state loaded from '{model_full_path}'.")

    # --- Feature Extraction Loop ---
    print("Extracting conditional vectors from V4 model's static branch...")
    all_conditional_vectors = []
    all_color_params = []

    with torch.no_grad():
        for x_full_transformed, x_static_original in tqdm(val_loader, desc="Extracting features"):
            x_full_transformed = x_full_transformed.to(device)
            
            # 8. 从 V4 完整模型中获取条件向量
            conditional_vectors = model.get_conditional_vector(x_full_transformed) 
            
            all_conditional_vectors.append(conditional_vectors.cpu())
            
            # Get the un-normalized parameter for coloring
            color_param = x_static_original[:, PARAM_TO_COLOR_IDX]
            all_color_params.append(color_param)

    # Combine all batches
    conditional_vectors_np = torch.cat(all_conditional_vectors, dim=0).numpy()
    color_params_np = torch.cat(all_color_params, dim=0).numpy()
    print(f"Extracted {conditional_vectors_np.shape[0]} conditional vectors.")

    # --- t-SNE Calculation ---
    print("Running t-SNE... (This may take a few minutes)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=RANDOM_SEED)
    tsne_results = tsne.fit_transform(conditional_vectors_np)
    print("t-SNE calculation complete.")

    # --- Plotting ---
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=color_params_np, 
        cmap='viridis', 
        alpha=0.7
    )
    
    plt.colorbar(label=PARAM_TO_COLOR_NAME)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of V4 Full Model (Static Branch)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tsne_visualization_stress_V4_FULL_MODEL.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nSuccessfully saved t-SNE plot to '{save_path}'.")
    print("==========================================================")


if __name__ == "__main__":
    main()