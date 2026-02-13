# -*- coding: utf-8 -*-
"""
Batch Ablation Evaluation Script for SUBSIDENCE (Strict Alignment).

Alignment with Stress Evaluation Logic:
1. GT-PCR Filtering: Filters samples where GT violates physics (Threshold=0.5).
2. Metrics: Adds MAE, PCC. Uses True Evo Error (Delta Model vs Delta GT).
3. Shape Fix: Flattens tensors for vector metrics; Reshapes for image metrics.
4. Dataset: Applies y.T (Transpose) for correct physical orientation.
"""

import os
import glob
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import math
import argparse
# [Mamba] Import
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it to use Mamba architecture.")
    Mamba = None

# --- 依赖检查 ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM will be 0.")

# --- 0. 基础配置 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset") # Subsidence Data
MODEL_DIR = os.path.join(BASE_DIR, "../trained_models_subsidence_ablation_mamba")
OUTPUT_DIR = os.path.join(BASE_DIR, "../evaluation_results_subsidence_mamba")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_summary_mamba.csv")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

STATIC_FEATURES = 11      # Subsidence specific
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42
STEP_DISTANCE_M = 10.0
BATCH_SIZE = 64
MODEL_LENGTH_M = 500.0    
MODEL_HEIGHT_M = 150.0    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 实验列表 ---
EXPERIMENTS = [
    {"name": "Full Dual-Branch", "ablation": "full",         "branch": "dual"},
    {"name": "Static Only",      "ablation": "full",         "branch": "static_only"},
    {"name": "Dynamic Only",     "ablation": "full",         "branch": "dynamic_only"},
    {"name": "Baseline (MSE)",   "ablation": "baseline",     "branch": "dual"},
    {"name": "No SSIM",          "ablation": "no_ssim",      "branch": "dual"},
    {"name": "No Arch Prior",    "ablation": "no_arch",      "branch": "dual"},
    {"name": "No Evolution",     "ablation": "no_evo",       "branch": "dual"},
    {"name": "No TV",            "ablation": "no_tv",        "branch": "dual"},
]

# --- 2. 物理掩膜生成器 (Subsidence Arch Model) ---

class ActivityArchMaskGenerator(nn.Module):
    def __init__(self, output_size=64):
        super(ActivityArchMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, self.H), 
            torch.linspace(0, MODEL_LENGTH_M, self.W),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def forward(self, mining_distances, phys_params):
        batch_size = mining_distances.size(0)
        masks = []
        for i in range(batch_size):
            d = mining_distances[i]
            h_max = phys_params[i, 0]; w_arch = phys_params[i, 1]
            beta = phys_params[i, 2]; lag = phys_params[i, 3]
            
            xc = d - lag 
            curr_H = h_max * torch.tanh(d / 100.0)
            
            x_term = (self.x_grid - xc) / (w_arch + 1e-6)
            # Physical boundary definition
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * (x_term.abs() <= 1.0).float()
            
            # Binary Mask (Valid zone is BELOW the arch curve)
            spatial_mask = (self.y_grid <= y_boundary).float()
            masks.append(spatial_mask)
            
        return torch.stack(masks).unsqueeze(1)

# --- 3. Dataset (关键修正: Transpose) ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None, global_index_map=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.global_index_map = global_index_map
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); s_start = s_pos + 7; s_id = int(filename[s_start : s_start + 4])
        st_pos = filename.rfind("step"); st_start = st_pos + 5; st_id = int(filename[st_start : st_start + 3])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M
        
        # Load Current T
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            
            # [FIX] Transpose Logic (Physical Alignment)
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T 
        
        if st_id > 1:
            prev_path = None
            if self.global_index_map is not None:
                prev_path = self.global_index_map.get((s_id, st_id - 1))
            
            if prev_path is None:
                prev_idx = self.index_map.get((s_id, st_id - 1))
                if prev_idx is not None: prev_path = self.file_list[prev_idx]

            if prev_path is not None:
                with np.load(prev_path) as data:
                    x_prev = torch.from_numpy(data['x'].astype(np.float32))
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
                    if y_prev.ndim == 1: y_prev = y_prev.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev = y_prev.T
            else: 
                y_prev = y_t.clone()
                x_prev = x_t.clone()
        else: 
            y_prev = torch.zeros_like(y_t)
            x_prev = x_t.clone()

        # Params
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)
        
        if self.transform:
            x_t = self.transform(x_t)
            x_prev = self.transform(x_prev) # Normalize prev state too
        
        return x_t, x_prev, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 4. Model (DualBranch) ---

# --- 2b. Baseline Models ---

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128; self.init_size = 8
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        x = self.fc(x).view(-1, self.init_channels, self.init_size, self.init_size)
        return self.conv_blocks(x).view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len; self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        # No dropout in Subsidence LSTM baseline as per inspection
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class TransformerDualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerDualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=500):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)

        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos = self.pos_encoder(dynamic_embedded)
        dynamic_out = self.transformer_encoder(dynamic_pos).mean(dim=1)
        fused = torch.cat((static_out, dynamic_out), dim=1)
        return self.fusion_head(fused)

# --- 2b. Baseline Models ---

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128; self.init_size = 8
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        x = self.fc(x).view(-1, self.init_channels, self.init_size, self.init_size)
        return self.conv_blocks(x).view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len; self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        # No dropout in Subsidence LSTM baseline as per inspection
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class TransformerDualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerDualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=500):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)

        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos = self.pos_encoder(dynamic_embedded)
        dynamic_out = self.transformer_encoder(dynamic_pos).mean(dim=1)
        fused = torch.cat((static_out, dynamic_out), dim=1)
        return self.fusion_head(fused)

# --- 3. Model Definition (DualBranchMamba) ---

class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, n_layers=2, dropout=0.1): 
        super(DualBranchMambaModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode
        
        # 1. Static Branch (MLP)
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(64, 32), 
            nn.ReLU()
        )
        
        # 2. Dynamic Branch (Mamba)
        self.dynamic_embedder = nn.Linear(1, d_model)
        
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=2,    # Block expansion factor
                ) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        else:
            if branch_mode in ['dual', 'dynamic_only']:
                 # Only strict if we need this branch
                 pass # Warning handled at import time

        # Fusion Head
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
        # x shape: [Batch, Static+Dynamic]
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:] # [Batch, Seq_Len]
        
        # Branch 1: Static
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            static_out = torch.zeros(x.size(0), 32, device=x.device)
        
        # Branch 2: Dynamic (Mamba)
        if self.branch_mode in ['dual', 'dynamic_only']:
            # Reshape for Mamba: [Batch, Seq_Len, 1] -> Embedding -> [Batch, Seq_Len, d_model]
            x_dynamic_seq = x_dynamic.unsqueeze(-1)
            x_mamba = self.dynamic_embedder(x_dynamic_seq)
            
            if hasattr(self, 'mamba_layers'):
                # Mamba Forward
                for layer, norm in zip(self.mamba_layers, self.norms):
                    residual = x_mamba
                    x_mamba = norm(x_mamba)
                    x_mamba = layer(x_mamba)
                    x_mamba = residual + x_mamba # Residual connection
                
                # Pooling
                dynamic_out = x_mamba.mean(dim=1)
            else:
                 # Fallback if mamba not installed (should not happen in testing env)
                 dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        # Fusion
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

def calculate_pcc(pred, target):
    vx = pred - torch.mean(pred, dim=(1,2,3), keepdim=True)
    vy = target - torch.mean(target, dim=(1,2,3), keepdim=True)
    cost = torch.sum(vx * vy, dim=(1,2,3))
    den = torch.sqrt(torch.sum(vx ** 2, dim=(1,2,3)) * torch.sum(vy ** 2, dim=(1,2,3))) + 1e-8
    return (cost / den).cpu().numpy()

# --- 5. Evaluation (With Filtering) ---

def evaluate_single_experiment(exp_config, val_loader, device, feature_dims, min_gt_pcr):
    ablation, branch = exp_config['ablation'], exp_config['branch']
    model_name = f"best_subsidence_{ablation}_{branch}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    print(f"Evaluating: {exp_config['name']} ...")
    if not os.path.exists(model_path):
        print(f"  [Skip] Model not found: {model_name}"); return None

    model = DualBranchMambaModel(
        static_size=feature_dims['static'], dynamic_size=feature_dims['dynamic'],
        output_size=OUTPUT_FEATURES, branch_mode=branch
    ).to(device)
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except: print("  [Error] Load failed"); return None
    model.eval()
    
    metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'Evo': [], 'PCC': [], 'PCR': []}
    mask_generator = ActivityArchMaskGenerator(output_size=64).to(device)
    
    total_samples = 0
    kept_samples = 0
    
    with torch.no_grad():
        for inputs, targets, targets_prev, dists, phys_params in tqdm(val_loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets_prev, dists, phys_params = targets_prev.to(device), dists.to(device), phys_params.to(device)
            
            # --- [Step 1] GT-PCR Filtering ---
            arch_masks = mask_generator(dists, phys_params) # [B, 1, 64, 64]
            tgt_imgs = targets.view(-1, 1, 64, 64)
            
            # Binarize GT (Subsidence Threshold > 0.05 to ignore noise)
            gt_bin = (tgt_imgs > 0.05).float()
            gt_intersection = (gt_bin * arch_masks).sum(dim=(1,2,3))
            gt_total = gt_bin.sum(dim=(1,2,3)) + 1e-6
            gt_pcr = gt_intersection / gt_total
            
            keep_indices = gt_pcr >= min_gt_pcr
            
            total_samples += inputs.size(0)
            kept_samples += keep_indices.sum().item()
            
            if not keep_indices.any(): continue
            
            # Filter Data
            inputs = inputs[keep_indices]
            targets = targets[keep_indices]
            targets_prev = targets_prev[keep_indices]
            arch_masks = arch_masks[keep_indices]
            tgt_imgs = tgt_imgs[keep_indices]
            
            # --- [Step 2] Inference ---
            outputs = model(inputs)
            pred_imgs = outputs.view(-1, 1, 64, 64)
            pred_prev = model(targets_prev.view(targets_prev.size(0), -1)) # Wait, need x_prev not y_prev for model input!
            # Correct Logic: We need x_prev from loader.
            # ERROR in Data Loader: Evaluate loop expects x_prev. 
            # I must fix the evaluate loop to use x_prev from loader. 
            # See fix below in Main and Loader.
            
    # [Correction]: The loop logic above was incomplete regarding x_prev.
    # Re-writing loop with x_prev correctly.
    return None # Placeholder, real logic in main loop below

# Redefining evaluate function correctly
def evaluate_loop(exp_config, val_loader, device, feature_dims, min_gt_pcr):
    ablation, branch = exp_config['ablation'], exp_config['branch']
    model_name = f"best_subsidence_{ablation}_{branch}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    print(f"Evaluating: {exp_config['name']} ...")
    if not os.path.exists(model_path):
        print(f"  [Skip] Model not found: {model_name}"); return None
        
    model = DualBranchMambaModel(feature_dims['static'], feature_dims['dynamic'], OUTPUT_FEATURES, branch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
    mask_generator = ActivityArchMaskGenerator(output_size=64).to(device)
    
    total_cnt, kept_cnt = 0, 0
    # Global Accums
    total_evo_error_sq = 0.0
    total_evo_energy_sq = 0.0
    criterion_mse = nn.MSELoss(reduction='none')
    criterion_mae = nn.L1Loss(reduction='none')
    
    with torch.no_grad():
        for x_t, x_prev, y_t, y_prev, dists, phys_params in tqdm(val_loader, leave=False):
            x_t, x_prev = x_t.to(device), x_prev.to(device)
            y_t, y_prev = y_t.to(device), y_prev.to(device)
            dists, phys_params = dists.to(device), phys_params.to(device)
            
            # Filtering
            arch_masks = mask_generator(dists, phys_params)
            tgt_imgs = y_t.view(-1, 1, 64, 64)
            gt_bin = (tgt_imgs > 0.05).float()
            gt_pcr = (gt_bin * arch_masks).sum(dim=(1,2,3)) / (gt_bin.sum(dim=(1,2,3)) + 1e-6)
            
            keep = gt_pcr >= min_gt_pcr
            total_cnt += len(keep); kept_cnt += keep.sum().item()
            if not keep.any(): continue
            
            x_t, x_prev = x_t[keep], x_prev[keep]
            y_t, y_prev = y_t[keep], y_prev[keep]
            arch_masks, tgt_imgs = arch_masks[keep], tgt_imgs[keep]
            
            # Inference
            pred_t = model(x_t)
            # pred_prev = model(x_prev) # Not needed for Pred-GT Evo Logic
            
            # Metrics
            y_t_flat = y_t.view(y_t.size(0), -1)
            y_prev_flat = y_prev.view(y_prev.size(0), -1)
            
            metrics['MSE'].extend(criterion_mse(pred_t, y_t_flat).mean(dim=1).cpu().numpy())
            metrics['MAE'].extend(criterion_mae(pred_t, y_t_flat).mean(dim=1).cpu().numpy())
            
            # True Evo: Global accumulation
            # Metric = sqrt(sum((Delta_pred - Delta_gt)^2) / (sum(Delta_gt^2) + epsilon))
            model_delta = pred_t - y_prev_flat
            true_delta = y_t_flat - y_prev_flat
            
            cur_diff_sq = torch.sum((model_delta - true_delta) ** 2).item()
            cur_norm_sq = torch.sum(true_delta ** 2).item()
            
            total_evo_error_sq += cur_diff_sq
            total_evo_energy_sq += cur_norm_sq
            
            pred_imgs = pred_t.view(-1, 1, 64, 64)
            metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
            
            # PCR (Model)
            pred_clean = torch.clamp(pred_imgs, 0.0, 1.0)
            pred_clean[pred_clean < 0.05] = 0.0 # Subsidence Noise Threshold
            masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
            total = pred_clean.sum(dim=(1,2,3)) + 1e-6
            metrics['PCR'].extend((masked/total).cpu().numpy())
            
            if SKIMAGE_AVAILABLE:
                np_pred = torch.clamp(pred_imgs, 0, 1).cpu().numpy()
                np_tgt = tgt_imgs.cpu().numpy()
                for p, t in zip(np_pred, np_tgt):
                    metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

    if kept_cnt == 0: return None
    # Calculate Global Evo
    global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
    
    res = {k: np.mean(v) for k, v in metrics.items()}
    res['Evo'] = global_evo
    res['Model'] = exp_config['name']
    print(f"  [Kept {kept_cnt}/{total_cnt}] MSE:{res['MSE']:.5f} | Global_Evo:{res['Evo']:.4f} | PCR:{res['PCR']:.4f}")
    return res

# --- 6. Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_gt_pcr", type=float, default=0.5, help="Filter Threshold")
    args = parser.parse_args()
    
    print("========================================================")
    print("  Subsidence Ablation Eval (Strict Alignment & Filtered)")
    print(f"  Condition: GT-PCR >= {args.min_gt_pcr}")
    print("========================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data."); return
    
    np.random.seed(RANDOM_SEED); np.random.shuffle(all_files)
    val_files = all_files[int(TRAIN_VAL_SPLIT_RATIO * len(all_files)):]
    
    # [FIX] Build Global Index Map
    print("Building Global File Index for Sequential Lookup...")
    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample"); s_start = s_pos + 7; s_id = int(fn[s_start : s_start + 4])
            st_pos = fn.rfind("step"); st_start = st_pos + 5; st_id = int(fn[st_start : st_start + 3])
            global_index_map[(s_id, st_id)] = fp
        except: continue
        
    # Stats (Full Training Set)
    print("Calculating/Loading Stats...")
    transform = None
    stats_path = os.path.join(MODEL_DIR, "subsidence_stats_ablation.pt")
    
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        transform = NormalizeTransform(stats['mean'], stats['std'])
    else:
        # Fallback Calculation
        train_files = all_files[:int(TRAIN_VAL_SPLIT_RATIO * len(all_files))]
        temp_loader = DataLoader(SequentialFractureDataset(train_files, "", global_index_map=global_index_map), batch_size=128)
        all_x = []
        for x, _, _, _, _, _ in tqdm(temp_loader, desc="Calc Stats"): all_x.append(x)
        x_tensor = torch.cat(all_x, dim=0)
        transform = NormalizeTransform(x_tensor.mean(dim=0), x_tensor.std(dim=0))

    # Loader with Global Map
    val_loader = DataLoader(SequentialFractureDataset(val_files, PARAMS_JSON_PATH, transform, global_index_map=global_index_map), batch_size=BATCH_SIZE, shuffle=False)
    
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    feature_dims = {'static': STATIC_FEATURES, 'dynamic': total_feats - STATIC_FEATURES}
    
    # Define Models to Evaluate
    
    # 1. Our Proposed Mamba Model
    mamba_model_path = os.path.join(BASE_DIR, "../trained_models_subsidence_physics_mamba", "best_subsidence_full_dual.pth")
    
    # 2. Transformer Baseline (Previous SOTA)
    transformer_dir = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_subsidence_ablation")
    transformer_path = os.path.join(transformer_dir, "best_subsidence_full_dual.pth") # Check filename match
    
    # 3. Standard Baselines (CNN, LSTM)
    baseline_dir = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_baselines_subsidence")
    cnn_path = os.path.join(baseline_dir, "best_baseline_subsidence_CNN.pth")
    lstm_path = os.path.join(baseline_dir, "best_baseline_subsidence_LSTM.pth")
    
    models_to_eval = [
        {"name": "Proposed (Mamba)", "path": mamba_model_path, "type": "Mamba", "branch": "dual"},
        {"name": "Transformer (Dual)", "path": transformer_path, "type": "Transformer", "branch": "dual"},
        {"name": "Start Baseline (CNN)", "path": cnn_path, "type": "CNN", "branch": None},
        {"name": "Seq Baseline (LSTM)", "path": lstm_path, "type": "LSTM", "branch": None},
    ]

    all_results = []
    
    for model_cfg in models_to_eval:
        print(f"Evaluating: {model_cfg['name']}...")
        model_path = model_cfg['path']
        
        # Check if Mamba model exists, if not warn
        if not os.path.exists(model_path):
            # Try alternative filename for Transformer if not found (Physics version)
            if model_cfg['type'] == 'Transformer':
                 alt_path = os.path.join(BASE_DIR, "../new_dl_model/trained_models_subsidence_physics/best_subsidence_physics_model.pth")
                 # Also try ablation name pattern
                 alt_path_2 = os.path.join(BASE_DIR, "../new_dl_model/trained_models_subsidence_ablation/best_subsidence_full_dual.pth")
                 
                 if os.path.exists(alt_path):
                     print(f"  [Info] Switched to Physics Model: {alt_path}")
                     model_path = alt_path
                 elif os.path.exists(alt_path_2):
                     print(f"  [Info] Switched to Ablation Model: {alt_path_2}")
                     model_path = alt_path_2
                 else:
                     print(f"  [Warning] Model file not found: {model_cfg['path']}")
                     continue
            else:
                print(f"  [Warning] Model file not found: {model_cfg['path']}")
                continue
            
        # Init Model
        if model_cfg['type'] == 'Mamba':
             model = DualBranchMambaModel(feature_dims['static'], feature_dims['dynamic'], OUTPUT_FEATURES, model_cfg['branch']).to(DEVICE)
        elif model_cfg['type'] == 'Transformer':
             model = TransformerDualBranchModel(feature_dims['static'], feature_dims['dynamic'], OUTPUT_FEATURES).to(DEVICE)
        elif model_cfg['type'] == 'CNN':
             model = DeepCNNBaseline(input_size=feature_dims['static'] + feature_dims['dynamic']).to(DEVICE)
        elif model_cfg['type'] == 'LSTM':
             model = BiLSTMBaseline(dynamic_len=feature_dims['dynamic'], static_len=feature_dims['static']).to(DEVICE)
        
        try: 
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"  [Error] Failed to load model: {e}")
            continue

        # Evaluation Loop with Filtering
        mask_generator = ActivityArchMaskGenerator(output_size=64).to(DEVICE)
        metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
        
        total_samples = 0
        kept_samples = 0
        
        total_evo_error_sq = 0.0
        total_evo_energy_sq = 0.0
        
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_mae = nn.L1Loss(reduction='none')
        
        with torch.no_grad():
            for x_t, x_prev, y_t, y_prev, dists, phys_params in tqdm(val_loader, leave=False):
                # Move to device
                x_t, x_prev = x_t.to(DEVICE), x_prev.to(DEVICE)
                y_t, y_prev = y_t.to(DEVICE), y_prev.to(DEVICE)
                dists, phys_params = dists.to(DEVICE), phys_params.to(DEVICE)
                
                # --- Step 1: Pre-calculate Mask and GT-PCR ---
                arch_masks = mask_generator(dists, phys_params) # [B, 1, 64, 64]
                tgt_imgs = y_t.view(-1, 1, 64, 64)
                
                # Binarize GT (Subsidence Threshold 0.05)
                gt_bin = (tgt_imgs > 0.05).float()
                
                gt_intersection = (gt_bin * arch_masks).sum(dim=(1,2,3))
                gt_total = gt_bin.sum(dim=(1,2,3)) + 1e-6
                gt_pcr = gt_intersection / gt_total
                
                # Filtering
                keep_indices = gt_pcr >= args.min_gt_pcr
                
                total_samples += x_t.size(0)
                kept_samples += keep_indices.sum().item()
                
                if not keep_indices.any(): continue
                
                # Apply filter
                x_t = x_t[keep_indices]
                x_prev = x_prev[keep_indices] # Need for Evo
                y_t = y_t[keep_indices]
                y_prev = y_prev[keep_indices]
                arch_masks = arch_masks[keep_indices]
                tgt_imgs = tgt_imgs[keep_indices]
                
                # --- Step 2: Inference ---
                outputs = model(x_t)
                
                targets_flat = y_t.view(y_t.size(0), -1)
                targets_prev_flat = y_prev.view(y_prev.size(0), -1)
                pred_imgs = outputs.view(-1, 1, 64, 64)
                
                # --- Step 3: Metrics ---
                metrics['MSE'].extend(criterion_mse(outputs, targets_flat).mean(dim=1).cpu().numpy())
                metrics['MAE'].extend(criterion_mae(outputs, targets_flat).mean(dim=1).cpu().numpy())
                
                # Evo
                pred_delta = outputs - targets_prev_flat
                gt_delta = targets_flat - targets_prev_flat
                cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
                cur_norm_sq = torch.sum(gt_delta ** 2).item()
                total_evo_error_sq += cur_diff_sq
                total_evo_energy_sq += cur_norm_sq
                
                metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
                
                if SKIMAGE_AVAILABLE:
                    np_pred = torch.clamp(pred_imgs, 0, 1).cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                    for p, t in zip(np_pred, np_tgt):
                        metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

                # PCR (Threshold 0.05 for Subsidence)
                pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
                pred_clean = pred_clamped.clone()
                pred_clean[pred_clean < 0.05] = 0.0
                
                masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
                total = pred_clean.sum(dim=(1,2,3)) + 1e-6
                
                valid_idx = total > 1e-3
                if valid_idx.any():
                    metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())

        if kept_samples == 0:
            res = {k: 0.0 for k in metrics}
            res['Evo'] = 0.0
        else:
            res = {k: np.mean(v) for k, v in metrics.items()}
            res["Evo"] = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
            
        res["Model"] = model_cfg['name']
        print(f"  [Samples: {kept_samples}/{total_samples}] MSE: {res['MSE']:.5f} | PCR: {res['PCR']:.4f}")
        all_results.append(res)

    print("\nWriting summary to CSV...")
    baseline_csv = os.path.join(OUTPUT_DIR, "baseline_comparison_summary.csv")
    fieldnames = ["Model", "MSE", "MAE", "SSIM", "PCC", "Evo", "PCR"]
    with open(baseline_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print("Done.")

if __name__ == "__main__":
    main()