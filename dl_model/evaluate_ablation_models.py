# -*- coding: utf-8 -*-
"""
Script to quantitatively evaluate the performance of the three ablation models
(Full, Static-Only, Dynamic-Only) on the entire test set.

Calculates and reports Mean Squared Error (MSE), Mean Absolute Error (MAE),
Symmetric Mean Absolute Percentage Error (SMAPE),
Structural Similarity Index (SSIM), and Pearson Correlation Coefficient (PCC)
for each model.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchmetrics # <-- 导入 torchmetrics

# --- Configuration ---

DATASET_DIR = "../final_dataset"

# Define paths relative to the current script's directory
SCRIPT_DIR = os.path.dirname(__file__)
ROOT_TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, "../trained_models")
DL_MODEL_TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, "trained_models")

DUAL_BRANCH_MODEL_PATH = os.path.join(ROOT_TRAINED_MODELS_DIR, "best_dual_branch_model.pth")
SUBSIDENCE_ONLY_MODEL_PATH = os.path.join(DL_MODEL_TRAINED_MODELS_DIR, "best_model_subsidence_only.pth")
STATIC_ONLY_MODEL_PATH = os.path.join(DL_MODEL_TRAINED_MODELS_DIR, "best_model_static_only.pth")

OUTPUT_DIR = "evaluation_results"

NUM_PARAMS = 11
GRID_RESOLUTION = (64, 64)
OUTPUT_FEATURES = GRID_RESOLUTION[0] * GRID_RESOLUTION[1]
TRAIN_VAL_SPLIT_RATIO = 0.9 # Same as in training scripts
BATCH_SIZE = 32 # Use same batch size as training

# --- 1. Model & Dataset Definitions (Copied from compare_all_models.py) ---

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

# --- 2. 自定义 SMAPE 评估函数 ---
def smape_loss_func(pred, target, epsilon=1e-8):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) loss.
    Formula: mean(2 * |pred - target| / (|pred| + |target| + epsilon))
    Range: [0, 2]
    """
    numerator = 2 * torch.abs(pred - target)
    denominator = torch.abs(pred) + torch.abs(target) + epsilon
    loss = torch.mean(numerator / denominator)
    return loss


# --- 3. Main Execution Block ---

def main():
    print("======================================================")
    print("           Ablation Models Evaluation Script          ")
    print("======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data files ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    # Note: For reproducible results, use a fixed seed
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print(f"Found {len(all_files)} total samples. Using {len(val_files)} for evaluation.")

    # --- Create Normalization Transforms (from training set) ---
    print("\nCalculating normalization stats from training set...")
    all_x_full_train = [torch.from_numpy(np.load(f)['x']) for f in train_files]
    x_full_tensor_train = torch.stack(all_x_full_train, dim=0).float()
    
    mean_full, std_full = x_full_tensor_train.mean(dim=0), x_full_tensor_train.std(dim=0)
    full_transform = NormalizeTransform(mean_full, std_full)

    x_static_tensor_train = x_full_tensor_train[:, :NUM_PARAMS]
    mean_static, std_static = x_static_tensor_train.mean(dim=0), x_static_tensor_train.std(dim=0)
    static_transform = NormalizeTransform(mean_static, std_static)

    x_dynamic_tensor_train = x_full_tensor_train[:, NUM_PARAMS:]
    mean_dynamic, std_dynamic = x_dynamic_tensor_train.mean(dim=0), x_dynamic_tensor_train.std(dim=0)
    subsidence_transform = NormalizeTransform(mean_dynamic, std_dynamic)

    # --- 3. 计算 y 的统计数据用于 SSIM ---
    print("Calculating y stats for SSIM data_range...")
    all_y_train = [torch.from_numpy(np.load(f)['y'].astype(np.float32)) for f in train_files]
    y_train_tensor = torch.stack(all_y_train, dim=0)
    y_min = y_train_tensor.min()
    y_max = y_train_tensor.max()
    ssim_data_range = (y_max - y_min).item() # 获取 y 的动态范围
    print(f"y data range for SSIM: {ssim_data_range:.4f} (min: {y_min:.4f}, max: {y_max:.4f})")
    
    print("Normalization stats calculated for all data types.")


    # --- Create Evaluation Dataset and DataLoader ---
    eval_dataset = AblationDataset(val_files, full_transform, static_transform, subsidence_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Load Models ---
    print("\nLoading models...")
    dynamic_size = x_dynamic_tensor_train.shape[1]

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

    # --- 4. Evaluate Models ---
    print("\nEvaluating models on the test set...")
    
    # --- 实例化所有评估标准 ---
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss() # MAE
    
    # 实例化 SSIM 指标 (使用 torchmetrics) - <--- 修复 FutureWarning
    ssim_full = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    ssim_static = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)
    ssim_dynamic = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=ssim_data_range).to(device)

    # 实例化 PCC 指标 (torchmetrics)
    pcc_full = torchmetrics.PearsonCorrCoef().to(device)
    pcc_static = torchmetrics.PearsonCorrCoef().to(device)
    pcc_dynamic = torchmetrics.PearsonCorrCoef().to(device)

    # --- 初始化指标累加器 ---
    total_mse_full = 0.0
    total_mse_static = 0.0
    total_mse_dynamic = 0.0

    total_mae_full = 0.0
    total_mae_static = 0.0
    total_mae_dynamic = 0.0

    total_smape_full = 0.0
    total_smape_static = 0.0
    total_smape_dynamic = 0.0

    with torch.no_grad():
        for x_full, x_static, x_dynamic, y_truth in tqdm(eval_loader, desc="Evaluating Models"):
            x_full, x_static, x_dynamic, y_truth = x_full.to(device), x_static.to(device), x_dynamic.to(device), y_truth.to(device)

            pred_full = full_model(x_full)
            pred_static = static_model(x_static)
            pred_dynamic = dynamic_model(x_dynamic)

            # --- 5. 重塑为图像格式 (B, C, H, W) 以便 SSIM 计算 ---
            B = y_truth.shape[0] # 获取当前批次大小
            y_truth_img = y_truth.reshape(B, 1, *GRID_RESOLUTION)
            pred_full_img = pred_full.reshape(B, 1, *GRID_RESOLUTION)
            pred_static_img = pred_static.reshape(B, 1, *GRID_RESOLUTION)
            pred_dynamic_img = pred_dynamic.reshape(B, 1, *GRID_RESOLUTION)

            # --- 计算并累积所有指标 ---

            # 1. Full Model
            total_mse_full += mse_criterion(pred_full, y_truth).item()
            total_mae_full += mae_criterion(pred_full, y_truth).item()
            total_smape_full += smape_loss_func(pred_full, y_truth).item()
            ssim_full.update(pred_full_img, y_truth_img) # 更新 SSIM 状态
            pcc_full.update(pred_full.flatten(), y_truth.flatten()) # 更新 PCC 状态 - <--- 修复 ValueError
            
            # 2. Static-Only Model
            total_mse_static += mse_criterion(pred_static, y_truth).item()
            total_mae_static += mae_criterion(pred_static, y_truth).item()
            total_smape_static += smape_loss_func(pred_static, y_truth).item()
            ssim_static.update(pred_static_img, y_truth_img)
            pcc_static.update(pred_static.flatten(), y_truth.flatten()) # 更新 PCC 状态 - <--- 修复 ValueError
            
            # 3. Dynamic-Only Model
            total_mse_dynamic += mse_criterion(pred_dynamic, y_truth).item()
            total_mae_dynamic += mae_criterion(pred_dynamic, y_truth).item()
            total_smape_dynamic += smape_loss_func(pred_dynamic, y_truth).item()
            ssim_dynamic.update(pred_dynamic_img, y_truth_img)
            pcc_dynamic.update(pred_dynamic.flatten(), y_truth.flatten()) # 更新 PCC 状态 - <--- 修复 ValueError

    # --- 6. 计算最终平均值并报告 ---
    
    num_batches = len(eval_loader)

    # Full Model Averages
    avg_mse_full = total_mse_full / num_batches
    avg_mae_full = total_mae_full / num_batches
    avg_smape_full = total_smape_full / num_batches
    avg_ssim_full = ssim_full.compute().item() # 从累积状态计算最终 SSIM
    avg_pcc_full = pcc_full.compute().item()   # 计算最终 PCC

    # Static-Only Model Averages
    avg_mse_static = total_mse_static / num_batches
    avg_mae_static = total_mae_static / num_batches
    avg_smape_static = total_smape_static / num_batches
    avg_ssim_static = ssim_static.compute().item()
    avg_pcc_static = pcc_static.compute().item()   # 计算最终 PCC

    # Dynamic-Only Model Averages
    avg_mse_dynamic = total_mse_dynamic / num_batches
    avg_mae_dynamic = total_mae_dynamic / num_batches
    avg_smape_dynamic = total_smape_dynamic / num_batches
    avg_ssim_dynamic = ssim_dynamic.compute().item()
    avg_pcc_dynamic = pcc_dynamic.compute().item() # 计算最终 PCC


    print("\n======================================================")
    print("                 Ablation Study Results               ")
    print("======================================================")
    
    print("--- Full Dual-Branch Model ---")
    print(f"  Average MSE:   {avg_mse_full:.6f}")
    print(f"  Average MAE:   {avg_mae_full:.6f}")
    print(f"  Average SMAPE: {avg_smape_full:.6f}  (Range: 0-2)")
    print(f"  Average SSIM:  {avg_ssim_full:.6f}  (Range: 0-1)")
    print(f"  Average PCC:   {avg_pcc_full:.6f}  (Range: -1 to 1)")
    
    print("\n--- Dynamic-Only Model ---")
    print(f"  Average MSE:   {avg_mse_dynamic:.6f}")
    print(f"  Average MAE:   {avg_mae_dynamic:.6f}")
    print(f"  Average SMAPE: {avg_smape_dynamic:.6f}  (Range: 0-2)")
    print(f"  Average SSIM:  {avg_ssim_dynamic:.6f}  (Range: 0-1)")
    print(f"  Average PCC:   {avg_pcc_dynamic:.6f}  (Range: -1 to 1)")

    print("\n--- Static-Only Model ---")
    print(f"  Average MSE:   {avg_mse_static:.6f}")
    print(f"  Average MAE:   {avg_mae_static:.6f}")
    print(f"  Average SMAPE: {avg_smape_static:.6f}  (Range: 0-2)")
    print(f"  Average SSIM:  {avg_ssim_static:.6f}  (Range: 0-1)")
    print(f"  Average PCC:   {avg_pcc_static:.6f}  (Range: -1 to 1)")
    
    print("======================================================")

if __name__ == "__main__":
    main()