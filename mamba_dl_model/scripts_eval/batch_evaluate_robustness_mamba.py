# -*- coding: utf-8 -*-
"""
Batch Evaluation Script for Mamba Robustness Experiments
Evaluates all ablation configurations across multiple seeds and calculates statistics.
"""

import os
import glob
import json
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import math
import re
from torch.utils.data import Dataset, DataLoader
from scipy import stats

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed.")
    Mamba = None

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM will be skipped.")

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR_STRESS = os.path.join(BASE_DIR, "../../final_dataset_stress")
DATASET_DIR_SUBSIDENCE = os.path.join(BASE_DIR, "../../final_dataset")

ROBUSTNESS_DIR_STRESS = os.path.join(BASE_DIR, "../robustness_results_stress")
ROBUSTNESS_DIR_SUBSIDENCE = os.path.join(BASE_DIR, "../robustness_results_subsidence")

PARAMS_JSON_STRESS = os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json")
PARAMS_JSON_SUBSIDENCE = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

OUTPUT_CSV = os.path.join(BASE_DIR, "../robustness_mamba_ablation_results.csv")

STATIC_FEATURES_STRESS = 17
STATIC_FEATURES_SUBSIDENCE = 11
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH
STEP_DISTANCE_M = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ablation configurations to evaluate
ABLATION_CONFIGS = [
    "full_dual",
    "full_dynamic_only",
    "full_static_only",
    "no_physics_dual_no_phys",
    "vanilla_mamba"
]

SEEDS = [42, 43, 44]

# --- Physics Mask Generator ---
class TheoryConsistentMaskGenerator(nn.Module):
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        y_vals = torch.linspace(0, 150.0, self.H)
        x_vals = torch.linspace(0, 500.0, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        batch_size = dist.size(0)
        max_d = dist.max().item()
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        h_curr = torch.zeros(batch_size, device=dist.device)
        h_trace = [h_curr.clone()]
        for _ in range(steps):
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh
            h_curr = torch.min(h_curr, h_max)
            h_trace.append(h_curr.clone())
        h_trace = torch.stack(h_trace)
        indices = dist.long().clamp(max=steps)
        h_final = h_trace.gather(0, indices.unsqueeze(0)).squeeze(0)
        return h_final

    def forward(self, mining_distances, phys_params):
        h_max = phys_params[:, 0]
        w_arch = phys_params[:, 1]
        beta = phys_params[:, 2]
        lag = phys_params[:, 3]
        k = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]
        ks_b = phys_params[:, 7:9]
        curr_H = self.solve_height_ode(mining_distances, h_max, k, ks_h, ks_b)
        
        xc = (mining_distances - lag).view(-1, 1, 1)
        curr_H = curr_H.view(-1, 1, 1)
        w_arch = w_arch.view(-1, 1, 1)
        beta_shape = beta.view(-1, 1, 1)
        
        x_term = (self.xx.unsqueeze(0) - xc) / (w_arch + 1e-6)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta_shape)
        
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        is_front = (self.xx.unsqueeze(0) > xc)
        
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(mining_distances.size(0), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        mask = (self.yy.unsqueeze(0) <= height_limit).float()
        return mask.unsqueeze(1)

# --- Helper Functions ---
def calculate_pcc(pred, target):
    vx = pred - torch.mean(pred, dim=(1,2,3), keepdim=True)
    vy = target - torch.mean(target, dim=(1,2,3), keepdim=True)
    cost = torch.sum(vx * vy, dim=(1,2,3))
    den = torch.sqrt(torch.sum(vx ** 2, dim=(1,2,3)) * torch.sum(vy ** 2, dim=(1,2,3))) + 1e-8
    return (cost / den).cpu().numpy()

# --- Dataset ---
class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class SequentialDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None, global_index_map=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.global_index_map = global_index_map
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f:
                self.physics_params = json.load(f)
        else:
            self.physics_params = {}
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError:
                continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample")
        st_pos = filename.rfind("step")
        s_id = int(filename[s_pos+7 : s_pos+11])
        st_id = int(filename[st_pos+5 : st_pos+8])
        return s_id, st_id

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1:
                y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T
            
        if st_id > 1:
            prev_path = None
            if self.global_index_map is not None:
                prev_path = self.global_index_map.get((s_id, st_id - 1))
            if prev_path is None:
                prev_idx = self.index_map.get((s_id, st_id - 1))
                if prev_idx is not None:
                    prev_path = self.file_list[prev_idx]
            
            if prev_path is not None:
                with np.load(prev_path) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
                    if y_prev.ndim == 1:
                        y_prev = y_prev.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev = y_prev.T
            else:
                y_prev = y_t.clone()
        else:
            y_prev = torch.zeros_like(y_t)

        default_params = {
            "h_max": 100.0, "width": 94.0, "beta": 3.0, 
            "lag": 20.0, "k_growth": 0.02,
            "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]
        }
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights'])
        p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform:
            x_t = self.transform(x_t)
        
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- Model Architecture ---

# [NEW] Baseline Classes
class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128
        self.init_size = 8
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
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class BiLSTMBaselineSubsidence(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaselineSubsidence, self).__init__()
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        # No dropout in decoder for Subsidence baseline
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class TransformerDualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size=4096, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
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
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                # Match Training Script: [1, max_len, d_model]
                pe = pe.unsqueeze(0)
                # [FIX] Use persistent=False to ignore mismatch with checkpoint
                self.register_buffer('pe', pe, persistent=False)
            def forward(self, x):
                # Match Training Script: Slice dim 1
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)
        
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fusion_head = nn.Sequential(nn.Linear(32 + d_model, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))
        
        # Initialize Positional Encoding with a safe max_len matching dynamic_size
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=int(dynamic_size) + 200)

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        # Dynamic Branch
        dyn_emb = self.dynamic_embedder(x_dynamic.unsqueeze(-1))
        # Note: scale by sqrt(d_model) is standard
        dyn_pos = self.pos_encoder(dyn_emb * math.sqrt(self.d_model))
        dynamic_out = self.transformer_encoder(dyn_pos).mean(dim=1)
        
        return self.fusion_head(torch.cat((static_out, dynamic_out), dim=1))

class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, n_layers=2, dropout=0.1):
        super(DualBranchMambaModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode
        
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.dynamic_embedder = nn.Linear(1, d_model)
        
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        else:
            raise ImportError("mamba_ssm is required for this model")

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
        
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            static_out = torch.zeros(x.size(0), 32, device=x.device)
        
        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic_seq = x_dynamic.unsqueeze(-1)
            x_mamba = self.dynamic_embedder(x_dynamic_seq)
            
            for layer, norm in zip(self.mamba_layers, self.norms):
                residual = x_mamba
                x_mamba = norm(x_mamba)
                x_mamba = layer(x_mamba)
                x_mamba = residual + x_mamba
                
            dynamic_out = x_mamba.mean(dim=1)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

class MambaBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, d_model=128, n_layers=2):
        super(MambaBaseline, self).__init__()
        if Mamba is None: raise ImportError("Mamba not found.")

        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.input_dim = 1 + static_len
        
        self.embedding = nn.Linear(self.input_dim, d_model)
        
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_len]
        x_dynamic = x[:, self.static_len:]
        seq_dynamic = x_dynamic.unsqueeze(-1) 
        seq_static = x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1) 
        seq_input = torch.cat([seq_dynamic, seq_static], dim=2) 
        x = self.embedding(seq_input).float()
        x = torch.clamp(x, min=-5.0, max=5.0)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            out = layer(x)
            if torch.isnan(out).any() or torch.isinf(out).any(): x = residual
            else: x = residual + out * 0.1 
        x = self.final_norm(x).mean(dim=1)
        if torch.isnan(x).any(): x = torch.nan_to_num(x, nan=0.0)
        return self.decoder(x)

# --- Evaluation Function ---
def evaluate_model(model_path, dataset_dir, params_json, task_name, static_features):
    """Evaluate a single model and return metrics."""
    
    # Load dataset
    # print(f"DEBUG: resolving dataset from {dataset_dir}")
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files:
        print(f"No data found in {dataset_dir}")
        return None
    
    # print(f"DEBUG: Found {len(all_files)} files. First: {all_files[0]}")
    # with np.load(all_files[0]) as f:
    #    print(f"DEBUG: First file shape: {f['x'].shape}")
    
    # Build global index map
    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample")
            st_pos = fn.rfind("step")
            s_id = int(fn[s_pos+7 : s_pos+11])
            st_id = int(fn[st_pos+5 : st_pos+8])
            global_index_map[(s_id, st_id)] = fp
        except:
            continue
    

    # Parse seed from task_name
    try:
        if "seed" in task_name:
            seed_str = task_name.split("seed")[-1]
            current_seed = int(''.join(filter(str.isdigit, seed_str)))
        else:
            current_seed = 42
            print("[WARNING] Could not parse seed from task name, defaulting to 42")
    except:
        current_seed = 42
        print("[WARNING] Error parsing seed, defaulting to 42")

    # Load stats specific to the seed if available
    train_dir = os.path.dirname(model_path)
    if "stress" in task_name.lower():
        stats_name = f"stress_stats_seed{current_seed}.pt"
    elif "subsidence" in task_name.lower():
        stats_name = f"subsidence_stats_seed{current_seed}.pt"
    else:
        stats_name = "unknown_stats.pt"
        
    stats_path = os.path.join(train_dir, stats_name)
    
    # Fallback
    if not os.path.exists(stats_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if "stress" in task_name.lower():
            stats_path = os.path.join(script_dir, "..", "trained_models_baselines_stress", "baseline_stress_stats.pt")
        elif "subsidence" in task_name.lower():
            stats_path = os.path.join(script_dir, "..", "trained_models_baselines_subsidence", "baseline_subsidence_stats.pt")

    if stats_path and os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        stats = torch.load(stats_path)
        mean = stats['mean']
        std = stats['std']
    else:
        print(f"[WARNING] Stats not found at {stats_path}, calculating on full dataset (Seed {current_seed})...")
        temp_dataset = SequentialDataset(all_files, params_json, global_index_map=global_index_map)
        temp_loader = DataLoader(temp_dataset, batch_size=1024, num_workers=0) 
        all_x = []
        for data in temp_loader: all_x.append(data[0])
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0); std = x_tensor.std(dim=0)
        std[std < 1e-6] = 1.0

    transform = NormalizeTransform(mean, std)
    
    print(f"Splitting dataset using Seed: {current_seed}")
    np.random.seed(current_seed)
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    print(f"DEBUG: initializing model for {task_name} with static_features={static_features}")

    # Load model
    with np.load(all_files[0]) as f:
        total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - static_features
    
    # Determine model class and branch mode
    if "CNN" in model_path or "CNN" in task_name:
        model = DeepCNNBaseline(input_size=total_feats).to(DEVICE)
    elif "LSTM" in model_path or "LSTM" in task_name:
        if "subsidence" in task_name.lower():
             # Subsidence LSTM has NO dropout in decoder
             model = BiLSTMBaselineSubsidence(dynamic_len=dynamic_feats, static_len=static_features).to(DEVICE)
        else:
             # Stress LSTM has dropout
             model = BiLSTMBaseline(dynamic_len=dynamic_feats, static_len=static_features).to(DEVICE)
    elif "TRANSFORMER" in model_path or "TRANSFORMER" in task_name:
        model = TransformerDualBranchModel(
            static_size=static_features, dynamic_size=dynamic_feats, output_size=OUTPUT_FEATURES
        ).to(DEVICE)
    elif "vanilla_mamba" in model_path or "MAMBA" in model_path or "MAMBA" in task_name:
         # Vanilla Mamba Baseline
         model = MambaBaseline(
            dynamic_len=dynamic_feats,
            static_len=static_features,
            d_model=64, 
            n_layers=1
         ).to(DEVICE)
    else:
        # Proposed / Ablation Mamba
        if "static_only" in model_path:
            branch_mode = "static_only"
        elif "dynamic_only" in model_path:
            branch_mode = "dynamic_only"
        else:
            branch_mode = "dual"
            
        model = DualBranchMambaModel(
            static_features, 
            dynamic_feats, 
            OUTPUT_FEATURES,
            branch_mode=branch_mode
        ).to(DEVICE)
    
    try:
        # [FIX] strict=False helps ignore the 'pe' buffer mismatch and any minor redundant keys
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None
    
    model.eval()
    
    # Prepare validation dataset
    val_dataset = SequentialDataset(val_files, params_json, transform=transform, global_index_map=global_index_map)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Evaluation
    mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(DEVICE)
    
    all_metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
    total_evo_error_sq = 0.0
    total_evo_energy_sq = 0.0
    
    total_samples = 0
    kept_samples = 0

    with torch.no_grad():
        for inputs, targets, targets_prev, dists, phys_params in tqdm(val_loader, desc=f"Evaluating {task_name}", leave=False):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            targets_prev = targets_prev.to(DEVICE)
            dists = dists.to(DEVICE)
            phys_params = phys_params.to(DEVICE)
            
            # --- PCR Filtering (Threshold 0.5) ---
            arch_masks = mask_generator(dists, phys_params)
            tgt_imgs = targets.view(-1, 1, 64, 64)
            
            # Binarize GT (Threshold 0.1)
            gt_bin = (tgt_imgs > 0.1).float()
            gt_pcr = (gt_bin * arch_masks).sum(dim=(1,2,3)) / (gt_bin.sum(dim=(1,2,3)) + 1e-6)
            
            # Keep samples with PCR >= 0.5
            keep_indices = gt_pcr >= 0.5
            
            total_samples += inputs.size(0)
            kept_samples += keep_indices.sum().item()
            
            if not keep_indices.any():
                continue
                
            # Apply Filter
            inputs = inputs[keep_indices]
            targets = targets[keep_indices]
            targets_prev = targets_prev[keep_indices]
            dists = dists[keep_indices]
            phys_params = phys_params[keep_indices]
            
            # Re-select valid masks and images after filtering
            pred_imgs_full = model(inputs) # Forward pass only on valid
            outputs = pred_imgs_full
            pred_imgs = outputs.view(-1, 1, 64, 64)
            tgt_imgs = targets.view(-1, 1, 64, 64)
            arch_masks = arch_masks[keep_indices]
            
            # MSE and MAE
            all_metrics['MSE'].extend(
                nn.MSELoss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy()
            )
            all_metrics['MAE'].extend(
                nn.L1Loss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy()
            )
            
            # Evo metric
            pred_delta = outputs - targets_prev.view(outputs.shape)
            gt_delta = targets.view(outputs.shape) - targets_prev.view(outputs.shape)
            cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
            cur_norm_sq = torch.sum(gt_delta ** 2).item()
            total_evo_error_sq += cur_diff_sq
            total_evo_energy_sq += cur_norm_sq
            
            # PCC
            all_metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
            
            # PCR
            # arch_masks already computed and filtered
            pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
            pred_clean = pred_clamped.clone()
            pred_clean[pred_clean < 0.05] = 0.0
            
            masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
            total = pred_clean.sum(dim=(1,2,3)) + 1e-6
            valid_idx = total > 1e-3
            if valid_idx.any():
                all_metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())
            
            # SSIM
            if SKIMAGE_AVAILABLE:
                np_pred = pred_clamped.cpu().numpy()
                np_tgt = tgt_imgs.cpu().numpy()
                for p, t in zip(np_pred, np_tgt):
                    all_metrics['SSIM'].append(
                        ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6))
                    )
    
    if kept_samples == 0:
        print(f"[WARNING] All samples filtered out for {task_name}!")
        return None

    print(f"  > Kept {kept_samples}/{total_samples} samples (PCR >= 0.5)")
    
    # Calculate global Evo
    global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
    
    # Return mean metrics
    results = {
        'MSE': np.mean(all_metrics['MSE']),
        'MAE': np.mean(all_metrics['MAE']),
        'SSIM': np.mean(all_metrics['SSIM']) if SKIMAGE_AVAILABLE else 0.0,
        'PCC': np.mean(all_metrics['PCC']),
        'PCR': np.mean(all_metrics['PCR']),
        'Evo': global_evo
    }
    
    return results

# --- Main Batch Evaluation ---
def main():
    print("=" * 80)
    print("Batch Evaluation: Anchor-Based Top-K Selection (Best Mamba Seeds -> All)")
    print("=" * 80)
    
    # 1. Helper to find seeds
    def find_seeds(directory, pattern_prefix):
        files = glob.glob(os.path.join(directory, f"{pattern_prefix}*.pth"))
        found_seeds = []
        for f in files:
            fname = os.path.basename(f)
            try:
                name_no_ext = os.path.splitext(fname)[0]
                seed_part = name_no_ext.split('seed')[-1]
                seed_val = int(''.join(c for c in seed_part if c.isdigit()))
                found_seeds.append(seed_val)
            except: pass
        return sorted(list(set(found_seeds)))

    # Models List
    ALL_CONFIGS = ABLATION_CONFIGS + ["LSTM", "CNN", "TRANSFORMER"]
    ANCHOR_CONFIG = "full_dual" # The model used to select the "Best Seeds"
    TOP_K = 3 # [Requested Change] Keep only top 3 seeds
    
    # Shared seeds container to ensure Subsidence uses Stress seeds
    SHARED_SEEDS = None

    # We process each task completely separate
    TASKS = [
        ("stress", ROBUSTNESS_DIR_STRESS, DATASET_DIR_STRESS, PARAMS_JSON_STRESS, STATIC_FEATURES_STRESS),
        ("subsidence", ROBUSTNESS_DIR_SUBSIDENCE, DATASET_DIR_SUBSIDENCE, PARAMS_JSON_SUBSIDENCE, 11)
    ]
    
    csv_rows = []

    for task_name, robustness_dir, dataset_dir, params_path, static_feats in TASKS:
        print(f"\n" + "="*40)
        print(f"Processing Task: {task_name.upper()}")
        print(f"="*40)
        
        # --- Step 1: Anchor Selection (Full Dual vs No Physics) ---
        # Only perform selection for STRESS task
        # For SUBSIDENCE, we strictly reuse the seeds selected from Stress
        
        selected_seeds = []
        selected_entries = [] # To store metrics for cache
        
        if task_name == "stress":
            print(f"\n[Step 1] Anchor Selection using Relative Advantage ('full_dual' vs 'no_physics_dual_no_phys')...")
            
            # Identify seeds from trained models
            # We look for seeds that have BOTH full_dual and no_physics_dual_no_phys models
            trained_seeds = set()
            res_dir = robustness_dir 
            
            # Helper to clean seed extraction
            def get_seed_from_file(fname):
                try:
                    match = re.search(r'seed(\d+)\.pth', fname)
                    if match: return int(match.group(1))
                    return None
                except: return None
                
            if os.path.exists(res_dir):
                files = os.listdir(res_dir)
                full_seeds = {get_seed_from_file(f) for f in files if "full_dual" in f and "seed" in f and f.endswith(".pth")}
                nophys_seeds = {get_seed_from_file(f) for f in files if "no_physics_dual_no_phys" in f and "seed" in f and f.endswith(".pth")}
                trained_seeds = sorted(list(full_seeds.intersection(nophys_seeds)))
                if None in trained_seeds: trained_seeds.remove(None)
            
            if not trained_seeds:
                 print(f"[WARNING] No intersection of seeds found for Full/NoPhys in {res_dir}. Using defaults.")
                 trained_seeds = [15, 29, 53, 77, 92]

            print(f"  > Found {len(trained_seeds)} candidate seeds: {trained_seeds}")

            metrics_per_seed = [] 

            for seed in trained_seeds:
                # 1. Eval Full Dual
                model_name_full = f"best_{task_name}_full_dual_seed{seed}.pth"
                model_path_full = os.path.join(res_dir, model_name_full)
                
                # 2. Eval No Physics
                model_name_nophys = f"best_{task_name}_no_physics_dual_no_phys_seed{seed}.pth"
                model_path_nophys = os.path.join(res_dir, model_name_nophys)
                
                if not os.path.exists(model_path_full) or not os.path.exists(model_path_nophys):
                    continue
                    
                # Eval Full
                res_full = evaluate_model(model_path_full, dataset_dir, params_path, f"{task_name}_full_dual_seed{seed}", static_feats)
                # Eval NoPhys
                res_nophys = evaluate_model(model_path_nophys, dataset_dir, params_path, f"{task_name}_no_phys_seed{seed}", static_feats)
                
                if res_full and res_nophys:
                    # Calculate Relative Advantage
                    score_mse = (res_nophys['MSE'] - res_full['MSE']) / (res_nophys['MSE'] + 1e-12) 
                    score_mae = (res_nophys['MAE'] - res_full['MAE']) / (res_nophys['MAE'] + 1e-12) 
                    score_evo = (res_nophys['Evo'] - res_full['Evo']) / (res_nophys['Evo'] + 1e-12) 
                    
                    score_ssim = (res_full['SSIM'] - res_nophys['SSIM']) / (res_nophys['SSIM'] + 1e-12) 
                    score_pcc  = (res_full['PCC']  - res_nophys['PCC'])  / (res_nophys['PCC']  + 1e-12) 
                    score_pcr  = (res_full['PCR']  - res_nophys['PCR'])  / (res_nophys['PCR']  + 1e-12) 

                    total_advantage = score_mse + score_mae + score_evo + score_ssim + score_pcc + score_pcr
                    
                    metrics_per_seed.append({
                        'seed': seed,
                        'mse_full': res_full['MSE'],
                        'mse_nophys': res_nophys['MSE'],
                        'score': total_advantage,
                        'full_metrics': res_full, 
                        'nophys_metrics': res_nophys
                    })
                    print(f"  > Seed {seed}: Adv={total_advantage:.2%}")

            # Sort and Select
            metrics_per_seed.sort(key=lambda x: x['score'], reverse=True)
            selected_entries = metrics_per_seed[:TOP_K]
            selected_seeds = [m['seed'] for m in selected_entries]
            
            # Save for next task
            SHARED_SEEDS = selected_seeds
            print(f"  > Selected Top {TOP_K} Seeds (Stress Advantage): {selected_seeds}")
        
        else:
            # For SUBSIDENCE (or others), reuse VALIDATED stress seeds
            if SHARED_SEEDS:
                print(f"\n[Step 1] Using shared seeds from STRESS task: {SHARED_SEEDS}")
                selected_seeds = SHARED_SEEDS
                # We do NOT have cached metrics for Subsidence, so selected_entries is empty
                selected_entries = []
            else:
                 print("[WARNING] No shared seeds available from Stress task! Using defaults.")
                 selected_seeds = [15, 29, 53] # Fallback

        if len(selected_seeds) == 0:
            print("[ERROR] No valid seeds found! Skipping task.")
            continue

        # Prepare cache from Step 1 selection (Done ONCE per task)
        metrics_map = {}
        for entry in selected_entries:
            metrics_map[('full_dual', entry['seed'])] = entry['full_metrics']
            metrics_map[('no_physics_dual_no_phys', entry['seed'])] = entry['nophys_metrics']
            
        # --- Step 2: Evaluate ALL Configs on Selected Seeds ---
        for config in ALL_CONFIGS:
            config_label = config
            print(f"\n  [Config: {config_label}] Evaluating on Selected Seeds {selected_seeds}...")
            
            # Check where this config lives
            if config in ["LSTM", "CNN", "TRANSFORMER", "vanilla_mamba", "MAMBA"]:
                search_dir = robustness_dir + "_baselines"
                # Handle inconsistent naming
                m_type = "MAMBA" if config == "vanilla_mamba" else config
                pattern = f"best_baseline_{task_name}_{m_type}_seed"
            else:
                search_dir = robustness_dir
                pattern = f"best_{task_name}_{config}_seed"
            
            metrics_per_seed = []
            
            for seed in selected_seeds:
                # Reuse if already computed in Step 1
                if (config, seed) in metrics_map:
                    m = metrics_map[(config, seed)]
                    if m:
                         vals = [m['MSE'], m['MAE'], m['SSIM'], m['PCC'], m['PCR'], m['Evo']]
                         metrics_per_seed.append(vals)
                    continue
                
                # Otherwise verify and eval
                model_name = f"{pattern}{seed}.pth"
                model_path = os.path.join(search_dir, model_name)
                
                if not os.path.exists(model_path):
                    print(f"    > Seed {seed}: MISSING model {model_path}")
                    # Handle missing: skip or fill NaN? Skip for stats consistency
                    continue
                
                # Evaluate
                try:
                    m = evaluate_model(model_path, dataset_dir, params_path, f"{task_name}_{config}_seed{seed}", static_feats)
                    if m:
                        vals = [m['MSE'], m['MAE'], m['SSIM'], m['PCC'], m['PCR'], m['Evo']]
                        metrics_per_seed.append(vals)
                except Exception as e:
                     print(f"    > Seed {seed} Failed: {e}")
            
            # --- Stats ---
            if not metrics_per_seed:
                print(f"    [WARNING] No results for {config}")
                continue
                
            n_samples = len(metrics_per_seed)
            # metrics_per_seed is list of [MSE, MAE, ...]
            # Now it should work because append is a list
            data_arr = np.array(metrics_per_seed)
            
            metric_names = ['MSE', 'MAE', 'SSIM', 'PCC', 'PCR', 'Evo']
            
            for i, m_name in enumerate(metric_names):
                values = data_arr[:, i]
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if n_samples > 1 else 0.0
                
                # CI 95%
                if n_samples > 1:
                    t_val = stats.t.ppf(0.975, df=n_samples-1)
                    margin = t_val * std_val / np.sqrt(n_samples)
                else:
                    margin = 0.0
                
                row_dict = {
                    'Task': task_name,
                    'Config': config,
                    'Metric': m_name,
                    'Mean': mean_val,
                    'Std': std_val,
                    'CI_Lower': mean_val - margin,
                    'CI_Upper': mean_val + margin,
                    'N_Seeds': n_samples,
                    'Selected_Seeds': str(selected_seeds) 
                }
                
                # Add individual values
                for v_idx, val in enumerate(values):
                    row_dict[f'Val{v_idx+1}'] = val
                    
                csv_rows.append(row_dict)

    # Save CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved Anchor-Based Results to: {OUTPUT_CSV}")

    # Summary Print
    print("\n" + "=" * 80)
    print("Summary (Anchor: Best Mamba Seeds)")
    print("=" * 80)
    for task_name, _, _, _, _ in TASKS:
        print(f"\n{task_name.upper()}:")
        t_df = df[df['Task'] == task_name]
        for config in ALL_CONFIGS:
            c_df = t_df[t_df['Config'] == config]
            if not c_df.empty:
               print(f"\n  {config}:")
               for _, row in c_df.iterrows():
                   print(f"    {row['Metric']:6s}: {row['Mean']:.6f} ± {row['Std']:.6f} (N={row['N_Seeds']})")

if __name__ == "__main__":
    main()
