# -*- coding: utf-8 -*-
"""
Script to quantitatively evaluate the performance of the three STRESS-BASED
ablation models (Full V4-HybridLoss, Static-Only, Dynamic-Only).

This script is updated to load the new 'DualBranchModel' (V1 architecture)
trained with the V4 Hybrid Loss.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics

# --- Configuration ---
DATASET_DIR = "../final_dataset_stress"
SCRIPT_DIR = os.path.dirname(__file__)
TRAINED_MODELS_DIR = "trained_models_stress" 

# --- _MODIFIED_ (Path to the new V4 model) ---
DUAL_BRANCH_V4_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_v4_hybrid_loss.pth")
DYNAMIC_ONLY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_dynamic_only.pth")
STATIC_ONLY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "best_stress_model_static_only.pth")
# --- _END_MODIFIED_ ---

STATIC_FEATURES = 17 
GRID_RESOLUTION = (64, 64)
OUTPUT_FEATURES = GRID_RESOLUTION[0] * GRID_RESOLUTION[1]
TRAIN_VAL_SPLIT_RATIO = 0.9 
BATCH_SIZE = 32
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
        with np.load(self.file_list[idx]) as data:
            x_full = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        x_static, x_dynamic = x_full[:STATIC_FEATURES], x_full[STATIC_FEATURES:]
        x_full_transformed = self.full_transform(x_full)
        x_static_transformed = self.static_transform(x_static)
        x_dynamic_transformed = self.dynamic_transform(x_dynamic)
        return x_full_transformed, x_static_transformed, x_dynamic_transformed, y

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

# --- [MODIFIED] Model 1: The V1 Full Model Architecture ---
class DualBranchModel(nn.Module):
    """ (This is the V1 Model from train_stress_model.py) """
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
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
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

# --- Model 2: The Dynamic-Only Ablation Model ---
class DynamicOnlyModel(nn.Module):
    def __init__(self, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DynamicOnlyModel, self).__init__()
        self.d_model = d_model
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
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

# --- Model 3: The Static-Only Ablation Model ---
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

def smape_loss_func(pred, target, epsilon=1e-8):
    numerator = 2 * torch.abs(pred - target)
    denominator = torch.abs(pred) + torch.abs(target) + epsilon
    loss = torch.mean(numerator / denominator)
    return loss

def main():
    print("======================================================")
    print("     Ablation Models Evaluation Script (Stress V4)    ")
    print("======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_full_path = os.path.join(SCRIPT_DIR, DATASET_DIR)
    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print(f"Found {len(all_files)} total samples. Using {len(val_files)} for evaluation.")

    print("\nLoading pre-calculated normalization stats...")
    try:
        stats_full_path = os.path.join(SCRIPT_DIR, "normalization_stats_stress.npz")
        stats_full = np.load(stats_full_path)
        full_transform = NormalizeTransform(torch.from_numpy(stats_full['mean']), torch.from_numpy(stats_full['std']))
        print(f"Loaded full stats from {stats_full_path}")
        stats_static_path = os.path.join(SCRIPT_DIR, "normalization_stats_static_stress.npz")
        stats_static = np.load(stats_static_path)
        static_transform = NormalizeTransform(torch.from_numpy(stats_static['mean']), torch.from_numpy(stats_static['std']))
        print(f"Loaded static stats from {stats_static_path}")
        stats_dynamic_path = os.path.join(SCRIPT_DIR, "normalization_stats_dynamic_stress.npz")
        stats_dynamic = np.load(stats_dynamic_path)
        dynamic_transform = NormalizeTransform(torch.from_numpy(stats_dynamic['mean']), torch.from_numpy(stats_dynamic['std']))
        print(f"Loaded dynamic stats from {stats_dynamic_path}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load normalization stats file. {e}")
        return

    print("Calculating y stats for SSIM data_range...")
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    ssim_data_range = (y_train_tensor.max() - y_train_tensor.min()).item()
    print(f"y data range for SSIM: {ssim_data_range:.4f}")

    eval_dataset = AblationDataset(val_files, full_transform, static_transform, dynamic_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nLoading models...")
    with np.load(all_files[0]) as f:
        dynamic_size = f['x'][STATIC_FEATURES:].shape[0]
    print(f"Detected features: Static={STATIC_FEATURES}, Dynamic={dynamic_size}")

    try:
        # --- [MODIFIED] Load V4 model ---
        full_model = DualBranchModel(static_size=STATIC_FEATURES, dynamic_size=dynamic_size, output_size=OUTPUT_FEATURES).to(device)
        full_model.load_state_dict(torch.load(DUAL_BRANCH_V4_MODEL_PATH, map_location=device))
        full_model.eval()
        print(f"Full model (V4 - Hybrid Loss) loaded from {DUAL_BRANCH_V4_MODEL_PATH}")
        # --- [END MODIFIED] ---

        dynamic_model = DynamicOnlyModel(dynamic_size=dynamic_size, output_size=OUTPUT_FEATURES).to(device)
        dynamic_model.load_state_dict(torch.load(DYNAMIC_ONLY_MODEL_PATH, map_location=device))
        dynamic_model.eval()
        print(f"Dynamic-only model loaded from {DYNAMIC_ONLY_MODEL_PATH}")

        static_model = StaticOnlyModel(static_size=STATIC_FEATURES, output_size=OUTPUT_FEATURES).to(device)
        static_model.load_state_dict(torch.load(STATIC_ONLY_MODEL_PATH, map_location=device))
        static_model.eval()
        print(f"Static-only model loaded from {STATIC_ONLY_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not load a model file. {e}")
        return
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Model architecture mismatch. {e}")
        return

    print("\nEvaluating models on the test set...")
    
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    ssim_full = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    ssim_static = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    ssim_dynamic = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)

    pcc_full = torchmetrics.PearsonCorrCoef().to(device)
    pcc_static = torchmetrics.PearsonCorrCoef().to(device)
    pcc_dynamic = torchmetrics.PearsonCorrCoef().to(device)

    total_mse_full, total_mse_static, total_mse_dynamic = 0.0, 0.0, 0.0
    total_mae_full, total_mae_static, total_mae_dynamic = 0.0, 0.0, 0.0
    total_smape_full, total_smape_static, total_smape_dynamic = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x_full, x_static, x_dynamic, y_truth in tqdm(eval_loader, desc="Evaluating Models"):
            x_full, x_static, x_dynamic, y_truth = x_full.to(device), x_static.to(device), x_dynamic.to(device), y_truth.to(device)
            pred_full = full_model(x_full)
            pred_static = static_model(x_static)
            pred_dynamic = dynamic_model(x_dynamic)
            B = y_truth.shape[0]
            y_truth_img = y_truth.reshape(B, 1, *GRID_RESOLUTION)
            pred_full_img = pred_full.reshape(B, 1, *GRID_RESOLUTION)
            pred_static_img = pred_static.reshape(B, 1, *GRID_RESOLUTION)
            pred_dynamic_img = pred_dynamic.reshape(B, 1, *GRID_RESOLUTION)
            
            total_mse_full += mse_criterion(pred_full, y_truth).item()
            total_mae_full += mae_criterion(pred_full, y_truth).item()
            total_smape_full += smape_loss_func(pred_full, y_truth).item()
            ssim_full.update(pred_full_img, y_truth_img)
            pcc_full.update(pred_full.flatten(), y_truth.flatten())
            
            total_mse_static += mse_criterion(pred_static, y_truth).item()
            total_mae_static += mae_criterion(pred_static, y_truth).item()
            total_smape_static += smape_loss_func(pred_static, y_truth).item()
            ssim_static.update(pred_static_img, y_truth_img)
            pcc_static.update(pred_static.flatten(), y_truth.flatten())
            
            total_mse_dynamic += mse_criterion(pred_dynamic, y_truth).item()
            total_mae_dynamic += mae_criterion(pred_dynamic, y_truth).item()
            total_smape_dynamic += smape_loss_func(pred_dynamic, y_truth).item()
            ssim_dynamic.update(pred_dynamic_img, y_truth_img)
            pcc_dynamic.update(pred_dynamic.flatten(), y_truth.flatten())

    num_batches = len(eval_loader)
    avg_mse_full, avg_mae_full, avg_smape_full = total_mse_full / num_batches, total_mae_full / num_batches, total_smape_full / num_batches
    avg_ssim_full, avg_pcc_full = ssim_full.compute().item(), pcc_full.compute().item()
    
    avg_mse_static, avg_mae_static, avg_smape_static = total_mse_static / num_batches, total_mae_static / num_batches, total_smape_static / num_batches
    avg_ssim_static, avg_pcc_static = ssim_static.compute().item(), pcc_static.compute().item()
    
    avg_mse_dynamic, avg_mae_dynamic, avg_smape_dynamic = total_mse_dynamic / num_batches, total_mae_dynamic / num_batches, total_smape_dynamic / num_batches
    avg_ssim_dynamic, avg_pcc_dynamic = ssim_dynamic.compute().item(), pcc_dynamic.compute().item()

    print("\n======================================================")
    print("        Ablation Study Results (Stress Model V4)      ")
    print("======================================================")
    
    print("--- Full Model (V4 - V1 Arch + Hybrid Loss) ---") # <-- _MODIFIED_
    print(f"   Average MSE:   {avg_mse_full:.6f}")
    print(f"   Average MAE:   {avg_mae_full:.6f}")
    print(f"   Average SMAPE: {avg_smape_full:.6f}  (Range: 0-2)")
    print(f"   Average SSIM:  {avg_ssim_full:.6f}  (Range: 0-1)")
    print(f"   Average PCC:   {avg_pcc_full:.6f}  (Range: -1 to 1)")
    
    print("\n--- Dynamic-Only Model (Transformer + MLP Head) ---")
    print(f"   Average MSE:   {avg_mse_dynamic:.6f}")
    print(f"   Average MAE:   {avg_mae_dynamic:.6f}")
    print(f"   Average SMAPE: {avg_smape_dynamic:.6f}  (Range: 0-2)")
    print(f"   Average SSIM:  {avg_ssim_dynamic:.6f}  (Range: 0-1)")
    print(f"   Average PCC:   {avg_pcc_dynamic:.6f}  (Range: -1 to 1)")

    print("\n--- Static-Only Model (MLP + MLP Head) ---")
    print(f"   Average MSE:   {avg_mse_static:.6f}")
    print(f"   Average MAE:   {avg_mae_static:.6f}")
    print(f"   Average SMAPE: {avg_smape_static:.6f}  (Range: 0-2)")
    print(f"   Average SSIM:  {avg_ssim_static:.6f}  (Range: 0-1)")
    print(f"   Average PCC:   {avg_pcc_static:.6f}  (Range: -1 to 1)")
    
    print("======================================================")

if __name__ == "__main__":
    main()