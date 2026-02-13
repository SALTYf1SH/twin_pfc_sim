# -*- coding: utf-8 -*-
"""
Final script to compare the predictions of all three models for the ablation study:
1. Full Dual-Branch Model
2. Dynamic-Only (Subsidence) Model
3. Static-Only Model

Generates a 1x4 plot comparing the Ground Truth with the three model predictions.
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

# --- Configuration ---

DATASET_DIR = "../final_dataset"
MODELS_DIR = "trained_models"
DUAL_BRANCH_MODEL_PATH = "../trained_models/best_dual_branch_model.pth"
SUBSIDENCE_ONLY_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_subsidence_only.pth")
STATIC_ONLY_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_static_only.pth")
OUTPUT_DIR = "evaluation_results"

NUM_PARAMS = 11
GRID_RESOLUTION = (64, 64)
OUTPUT_FEATURES = GRID_RESOLUTION[0] * GRID_RESOLUTION[1]
NUM_SAMPLES_TO_COMPARE = 5

# --- 1. Model & Dataset Definitions (Copied and adapted) ---

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class AblationDataset(Dataset):
    """Dataset that returns all three required input formats."""
    def __init__(self, npz_file_list, full_transform, static_transform, subsidence_transform):
        self.file_list = npz_file_list
        self.full_transform = full_transform
        self.static_transform = static_transform
        self.subsidence_transform = subsidence_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        
        x_static = x_full[:NUM_PARAMS]
        x_subsidence = x_full[NUM_PARAMS:]

        # Apply the correct transform to each part
        x_full_transformed = self.full_transform(x_full)
        x_static_transformed = self.static_transform(x_static)
        x_subsidence_transformed = self.subsidence_transform(x_subsidence)
            
        return x_full_transformed, x_static_transformed, x_subsidence_transformed, y

# Model definitions need to be here to load the state_dicts
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
        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

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

class SubsidenceOnlyModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(SubsidenceOnlyModel, self).__init__()
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

class StaticOnlyModel(nn.Module):
    def __init__(self, static_size, output_size, dropout=0.1):
        super(StaticOnlyModel, self).__init__()
        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.prediction_head = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

    def forward(self, x):
        static_out = self.static_branch(x)
        output = self.prediction_head(static_out)
        return output

# --- 2. Plotting Function ---

def create_comparison_plot(sample_file, transforms, models, device, index):
    """Generates and saves a comparison plot for a single sample."""
    full_transform, static_transform, subsidence_transform = transforms
    full_model, static_model, dynamic_model = models

    # --- Load and Prepare Data ---
    dataset = AblationDataset([sample_file], full_transform, static_transform, subsidence_transform)
    x_full, x_static, x_dynamic, y_truth = dataset[0]

    # Add batch dimension
    x_full = x_full.unsqueeze(0).to(device)
    x_static = x_static.unsqueeze(0).to(device)
    x_dynamic = x_dynamic.unsqueeze(0).to(device)

    # --- Generate Predictions ---
    with torch.no_grad():
        pred_full = full_model(x_full)
        pred_static = static_model(x_static)
        pred_dynamic = dynamic_model(x_dynamic)

    # --- Reshape and Plot ---
    y_truth_grid = y_truth.reshape(GRID_RESOLUTION).T
    pred_full_grid = pred_full.cpu().reshape(GRID_RESOLUTION).T
    pred_static_grid = pred_static.cpu().reshape(GRID_RESOLUTION).T
    pred_dynamic_grid = pred_dynamic.cpu().reshape(GRID_RESOLUTION).T

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    
    # Plot Ground Truth
    im = axes[0].imshow(y_truth_grid, cmap='jet', origin='lower', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth', fontsize=16)
    
    # Plot Full Model Prediction
    axes[1].imshow(pred_full_grid, cmap='jet', origin='lower', vmin=0, vmax=1)
    axes[1].set_title('Prediction (Full Model)', fontsize=16)

    # Plot Dynamic-Only Model Prediction
    axes[2].imshow(pred_dynamic_grid, cmap='jet', origin='lower', vmin=0, vmax=1)
    axes[2].set_title('Prediction (Dynamic Only)', fontsize=16)

    # Plot Static-Only Model Prediction
    axes[3].imshow(pred_static_grid, cmap='jet', origin='lower', vmin=0, vmax=1)
    axes[3].set_title('Prediction (Static Only)', fontsize=16)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    sample_name = os.path.basename(sample_file).replace('.npz', '')
    fig.suptitle(f'Ablation Study Comparison for {sample_name}', fontsize=20)
    
    save_path = os.path.join(OUTPUT_DIR, f'ablation_comparison_{index}.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Comparison plot saved to {save_path}")

# --- 3. Main Execution Block ---

def main():
    print("======================================================")
    print("          Ablation Study Comparison Script        ")
    print("======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    np.random.shuffle(all_files)
    sample_files = all_files[:NUM_SAMPLES_TO_COMPARE]

    # --- Create Normalization Transforms ---
    print("\nCalculating normalization stats...")
    all_x_full = [torch.from_numpy(np.load(f)['x']) for f in all_files]
    x_full_tensor = torch.stack(all_x_full, dim=0).float()
    
    # Full transform
    mean_full, std_full = x_full_tensor.mean(dim=0), x_full_tensor.std(dim=0)
    full_transform = NormalizeTransform(mean_full, std_full)

    # Static transform
    x_static_tensor = x_full_tensor[:, :NUM_PARAMS]
    mean_static, std_static = x_static_tensor.mean(dim=0), x_static_tensor.std(dim=0)
    static_transform = NormalizeTransform(mean_static, std_static)

    # Dynamic transform
    x_dynamic_tensor = x_full_tensor[:, NUM_PARAMS:]
    mean_dynamic, std_dynamic = x_dynamic_tensor.mean(dim=0), x_dynamic_tensor.std(dim=0)
    subsidence_transform = NormalizeTransform(mean_dynamic, std_dynamic)
    print("Normalization stats calculated for all data types.")

    # --- Load Models ---
    print("\nLoading models...")
    dynamic_size = x_dynamic_tensor.shape[1]

    # 1. Full Model
    full_model = DualBranchModel(static_size=NUM_PARAMS, dynamic_size=dynamic_size, output_size=OUTPUT_FEATURES).to(device)
    full_model.load_state_dict(torch.load(DUAL_BRANCH_MODEL_PATH, map_location=device))
    full_model.eval()
    print(f"Full model loaded from {DUAL_BRANCH_MODEL_PATH}")

    # 2. Dynamic-Only Model
    dynamic_model = SubsidenceOnlyModel(input_size=dynamic_size, output_size=OUTPUT_FEATURES).to(device)
    dynamic_model.load_state_dict(torch.load(SUBSIDENCE_ONLY_MODEL_PATH, map_location=device))
    dynamic_model.eval()
    print(f"Dynamic-only model loaded from {SUBSIDENCE_ONLY_MODEL_PATH}")

    # 3. Static-Only Model
    static_model = StaticOnlyModel(static_size=NUM_PARAMS, output_size=OUTPUT_FEATURES).to(device)
    static_model.load_state_dict(torch.load(STATIC_ONLY_MODEL_PATH, map_location=device))
    static_model.eval()
    print(f"Static-only model loaded from {STATIC_ONLY_MODEL_PATH}")

    # --- Generate Plots ---
    models = (full_model, static_model, dynamic_model)
    transforms = (full_transform, static_transform, subsidence_transform)

    print(f"\nGenerating {NUM_SAMPLES_TO_COMPARE} comparison plots...")
    for i, sample_file in enumerate(tqdm(sample_files, desc="Generating Plots")):
        create_comparison_plot(sample_file, transforms, models, device, i + 1)

    print("\n======================================================")
    print("                       Done!                        ")
    print("======================================================")

if __name__ == "__main__":
    main()
