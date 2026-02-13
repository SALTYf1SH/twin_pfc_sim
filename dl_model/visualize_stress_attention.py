# -*- coding: utf-8 -*-
"""
Script to visualize the attention mechanism of the trained DUAL-BRANCH
Stress-Based model (V1/V4 architecture).

V3: Corrected dual-Y-axis plotting logic.
    - Left Y-Axis (ax1): Stress and Stress Gradient (physical values)
    - Right Y-Axis (ax2): Attention Score (normalized value)
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer # 明确导入

# --- Configuration ---
DATASET_DIR = "../final_dataset_stress"
MODEL_PATH = "trained_models_stress/best_stress_model_v4_hybrid_loss.pth"
STATS_PATH = "normalization_stats_stress.npz" # V1/V4 模型的归一化文件
OUTPUT_DIR = "evaluation_results_stress" # 保存到应力评估文件夹
STATIC_FEATURES = 17 # HQH工况的17个静态参数

# --- 1. Dataset and Model Definitions ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class FractureDataset(Dataset):
    """Modified to return original_x, transformed_x, y, and filename."""
    def __init__(self, npz_file_list, transform=None):
        self.file_list, self.transform = npz_file_list, transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            original_x = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        transformed_x = original_x
        if self.transform:
            transformed_x = self.transform(original_x)
        # 返回原始索引以便在 val_dataset 中查找
        return original_x, transformed_x, y, os.path.basename(filepath), idx


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

# --- V1/V4 Dual Branch Model Architecture ---
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

    def forward_with_attention(self, x_full):
        x_static = x_full[:, :self.static_feature_size]
        x_dynamic = x_full[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_dynamic_unsqueezed = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic_unsqueezed) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        attention_maps = []
        dynamic_output = dynamic_pos_encoded
        for layer in self.transformer_encoder.layers:
            attn_output, attn_weights = layer.self_attn(
                dynamic_output, dynamic_output, dynamic_output, 
                need_weights=True, average_attn_weights=False
            )
            attention_maps.append(attn_weights)
            dynamic_output = dynamic_output + layer.dropout1(attn_output)
            dynamic_output = layer.norm1(dynamic_output)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(dynamic_output))))
            dynamic_output = dynamic_output + layer.dropout2(ff_output)
            dynamic_output = layer.norm2(dynamic_output)
        dynamic_out_pooled = dynamic_output.mean(dim=1)
        fused = torch.cat((static_out, dynamic_out_pooled), dim=1)
        final_prediction = self.fusion_head(fused)
        return final_prediction, attention_maps

# --- 2. Visualization Function ---

def visualize_attention(model, dataset, device, sample_idx=0, save_name_suffix=None):
    
    # Get a single sample using the index
    original_x, transformed_x, target_y, fname, _ = dataset[sample_idx]
    
    original_x_batch = original_x.unsqueeze(0).to(device)
    transformed_x_batch = transformed_x.unsqueeze(0).to(device)
    
    model.eval()
    
    with torch.no_grad():
        prediction, attention_maps = model.forward_with_attention(transformed_x_batch)

    original_x_np = original_x.squeeze(0).cpu().numpy()
    stress_curve = original_x_np[STATIC_FEATURES:]
    last_layer_attention = attention_maps[-1].squeeze(0).cpu().numpy()
    avg_head_attention = np.mean(last_layer_attention, axis=0)
    attention_score = np.sum(avg_head_attention, axis=0)
    
    min_score, max_score = np.min(attention_score), np.max(attention_score)
    if max_score > min_score:
        attention_score = (attention_score - min_score) / (max_score - min_score)
    else:
        attention_score = np.zeros_like(attention_score)

    gradient = np.gradient(stress_curve)
    
    # --- [MODIFIED] Dual-Y-Axis Plot Logic ---
    fig, ax1 = plt.subplots(figsize=(18, 8))
    x_axis = np.arange(len(stress_curve))
    
    # --- AXIS 1 (Left): Stress and Gradient ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Workface Position Index')
    ax1.set_ylabel('Floor Stress / Gradient', color=color1) # <-- 1. Y-label
    # Plot Stress
    l1 = ax1.plot(x_axis, stress_curve, color=color1, label='Floor Stress', lw=2)
    
    color2 = 'tab:green'
    # Plot Gradient
    l2 = ax1.plot(x_axis, gradient, color=color2, label='Stress Gradient', linestyle='--', lw=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- AXIS 2 (Right): Attention ---
    ax2 = ax1.twinx()
    color3 = 'tab:red'
    ax2.set_ylabel('Attention Score', color=color3) # <-- 2. Y-label
    # Plot Attention
    l3 = ax2.fill_between(x_axis, attention_score, color=color3, alpha=0.4, label='Attention Score')
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(0, 1.05) # <-- 3. Fixed scale for attention
    # --- [END MODIFIED] ---

    if save_name_suffix is None:
        save_name_suffix = f"sample_{sample_idx}"
    
    fig.suptitle(f'Transformer Attention Analysis (Stress Model)\n{save_name_suffix}', fontsize=16)
    
    # --- [MODIFIED] Combined Legend ---
    all_lines = l1 + l2
    all_labels = [l.get_label() for l in all_lines]
    # Manually add the fill patch to the legend
    all_lines.append(l3)
    all_labels.append(l3.get_label())
    
    fig.legend(all_lines, all_labels, loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # --- [END MODIFIED] ---
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'attention_visualization_stress_{save_name_suffix}.png')
    plt.savefig(save_path)
    
    plt.close(fig)

# --- 3. Main Execution Block ---

def main():
    # --- [FIX] 'global' 声明必须在任何使用之前 ---
    global OUTPUT_DIR
    
    """Main function to load model and run visualization."""
    print("==========================================================")
    print("       Stress Model Attention Visualization Script        ")
    print("==========================================================")
    
    script_dir = os.path.dirname(__file__)
    model_full_path = os.path.join(script_dir, "..", MODEL_PATH)
    stats_full_path = os.path.join(script_dir, STATS_PATH)
    dataset_full_path = os.path.join(script_dir, DATASET_DIR)
    
    # 正确地读取和重新分配全局变量
    output_full_dir = os.path.join(script_dir, "..", OUTPUT_DIR)
    OUTPUT_DIR = output_full_dir 

    if not os.path.exists(model_full_path):
        print(f"FATAL: Model file not found at '{model_full_path}'")
        return

    if not os.path.exists(stats_full_path):
        print(f"FATAL: Stats file not found at '{stats_full_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(dataset_full_path, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{dataset_full_path}'.")
        return
    
    print(f"Found {len(all_files)} total samples.")

    np.random.seed(42)
    np.random.shuffle(all_files)
    TRAIN_VAL_SPLIT_RATIO = 0.9
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Loading normalization stats...")
    stats = np.load(stats_full_path)
    mean = torch.from_numpy(stats['mean'])
    std = torch.from_numpy(stats['std'])
    transform = NormalizeTransform(mean, std)
    print(f"Normalization stats loaded from '{stats_full_path}'.")

    val_dataset = FractureDataset(val_files, transform=transform)
    print(f"Created validation set with {len(val_dataset)} samples.")

    print("Initializing and loading the model...")
    temp_x = val_dataset[0][0]
    dynamic_features = temp_x.shape[0] - STATIC_FEATURES
    output_features = 64 * 64

    model = DualBranchModel(
        static_size=STATIC_FEATURES, 
        dynamic_size=dynamic_features,
        output_size=output_features
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_full_path, map_location=device))
    except RuntimeError as e:
        print(f"FATAL: Error loading model state_dict. Architecture mismatch? {e}")
        return
        
    print(f"Model state loaded from '{model_full_path}'.")

    num_samples_to_visualize = 100
    if len(val_dataset) < num_samples_to_visualize:
        num_samples_to_visualize = len(val_dataset)

    random_indices = np.random.choice(len(val_dataset), num_samples_to_visualize, replace=False)

    print(f"\nStarting visualization for {num_samples_to_visualize} random samples...")
    for idx in tqdm(random_indices, desc="Generating attention plots"):
        # We need the index *within* the val_dataset
        # val_dataset[idx] returns (..., original_idx_in_all_files)
        # We actually just need the index in val_dataset, which is 'idx'
        
        fname = val_dataset[idx][3] 
        filename_suffix = fname.replace('.npz', '')
        
        visualize_attention(model, val_dataset, device, sample_idx=idx, save_name_suffix=filename_suffix)

    print(f"\nFinished generating {num_samples_to_visualize} plots in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()