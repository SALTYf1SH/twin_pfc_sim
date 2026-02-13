# -*- coding: utf-8 -*-
"""
Batch Evaluation Script for Baseline Robustness Experiments
Evaluates baseline models (CNN, LSTM, Transformer, Mamba) across multiple seeds.
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
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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

ROBUSTNESS_DIR_STRESS = os.path.join(BASE_DIR, "../robustness_results_stress_baselines")
ROBUSTNESS_DIR_SUBSIDENCE = os.path.join(BASE_DIR, "../robustness_results_subsidence_baselines")

PARAMS_JSON_STRESS = os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json")
PARAMS_JSON_SUBSIDENCE = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

OUTPUT_CSV = os.path.join(BASE_DIR, "../robustness_baseline_results.csv")

STATIC_FEATURES_STRESS = 17
STATIC_FEATURES_SUBSIDENCE = 11
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH
STEP_DISTANCE_M = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline Models to evaluate
BASELINE_MODELS = ["CNN", "LSTM", "TRANSFORMER", "MAMBA"]
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

# --- Baseline Model Architectures ---
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

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, use_dropout=True):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len = dynamic_len
        self.static_len = static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        
        # Subsidence training script did not use dropout in decoder, Stress did.
        if use_dropout:
            self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, output_size))
        else:
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
                pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)
        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fusion_head = nn.Sequential(nn.Linear(32 + d_model, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        dynamic_out = self.transformer_encoder(self.pos_encoder(self.dynamic_embedder(x_dynamic.unsqueeze(-1)) * math.sqrt(self.d_model))).mean(dim=1)
        return self.fusion_head(torch.cat((static_out, dynamic_out), dim=1))

# --- Evaluation Function ---
def evaluate_model(model_path, dataset_dir, params_json, model_type, task_name, static_features):
    """Evaluate a single baseline model."""
    
    # Load dataset
    all_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if not all_files:
        print(f"No data found in {dataset_dir}")
        return None
    
    # Global Index Map
    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample")
            st_pos = fn.rfind("step")
            s_id = int(fn[s_pos+7 : s_pos+11])
            st_id = int(fn[st_pos+5 : st_pos+8])
            global_index_map[(s_id, st_id)] = fp
        except: continue

    # Stats
    # FIX: Try to load saved stats first (from training) to ensure exact match
    # Path relative to script: ../trained_models_baselines_{task}/baseline_{task}_stats.pt
    # But we are in scripts_eval, so ../trained_models_baselines_{task} is correct.
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_dir_name = f"trained_models_baselines_{task_name.split('_')[0]}" # extracts 'stress' or 'subsidence'
    stats_fname = f"baseline_{task_name.split('_')[0]}_stats.pt"
    # stats_path = os.path.join(script_dir, "..", stats_dir_name, stats_fname)  <-- This might be brittle if task_name varies
    
    # Better approach: explicit check based on task
    if "stress" in task_name.lower():
        stats_path = os.path.join(script_dir, "..", "trained_models_baselines_stress", "baseline_stress_stats.pt")
    elif "subsidence" in task_name.lower():
        stats_path = os.path.join(script_dir, "..", "trained_models_baselines_subsidence", "baseline_subsidence_stats.pt")
    else:
        stats_path = None
    
    if stats_path and os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        stats = torch.load(stats_path)
        mean = stats['mean']
        std = stats['std']
    else:
        print(f"Stats not found at {stats_path}, calculating on full dataset...")
        # Fallback: Calculate on ALL files (not subset) to match training logic
        temp_dataset = SequentialDataset(all_files, params_json, global_index_map=global_index_map)
        temp_loader = DataLoader(temp_dataset, batch_size=1024, num_workers=0) # Large batch for speed
        all_x = []
        for data in temp_loader: all_x.append(data[0])
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0); std = x_tensor.std(dim=0)
        std[std < 1e-6] = 1.0

    transform = NormalizeTransform(mean, std)
    
    # Split
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    # Dimensions
    print(f"DEBUG: initializing Baseline {model_type} for {task_name} with static_features={static_features}")
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - static_features
    
    # Initialize Model
    if model_type == "MAMBA":
        # Check for full dual model instantiation
        # The user wants "Full Dual" Mamba to be the baseline representative
        if "full_dual" in model_path or "robustness_results_stress" in model_path or "robustness_results_subsidence" in model_path:
             model = DualBranchMambaModel(static_features, dynamic_feats, OUTPUT_FEATURES).to(DEVICE)
        else:
             # Fallback to vanilla if path doesn't look like full dual (shouldn't happen with new logic)
             model = MambaBaseline(dynamic_len=dynamic_feats, static_len=static_features).to(DEVICE)
    elif model_type == "CNN":
        model = DeepCNNBaseline(input_size=total_feats).to(DEVICE)
    elif model_type == "LSTM":
        # Check if we need to disable dropout (Subsidence models mismatch)
        # static_features == 11 is the identifier for Subsidence here
        use_dropout = True
        if static_features == 11 or "subsidence" in task_name.lower():
            use_dropout = False
        
        model = BiLSTMBaseline(dynamic_len=dynamic_feats, static_len=static_features, use_dropout=use_dropout).to(DEVICE)
    elif model_type == "TRANSFORMER":
        model = TransformerDualBranchModel(static_size=static_features, dynamic_size=dynamic_feats, output_size=OUTPUT_FEATURES).to(DEVICE)
    else:
        print(f"Unknown model type: {model_type}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None
    
    model.eval()
    
    val_dataset = SequentialDataset(val_files, params_json, transform=transform, global_index_map=global_index_map)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(DEVICE)
    all_metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
    total_evo_error_sq = 0.0
    total_evo_energy_sq = 0.0
    
    total_samples = 0
    kept_samples = 0
    
    with torch.no_grad():
        for inputs, targets, targets_prev, dists, phys_params in tqdm(val_loader, desc=f"Eval {task_name}", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            targets_prev, dists, phys_params = targets_prev.to(DEVICE), dists.to(DEVICE), phys_params.to(DEVICE)
            
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
            arch_masks = arch_masks[keep_indices]
            
            # Forward pass only on valid samples
            outputs = model(inputs)
            pred_imgs = outputs.view(-1, 1, 64, 64)
            tgt_imgs = targets.view(-1, 1, 64, 64)
            
            # MSE MAE
            all_metrics['MSE'].extend(nn.MSELoss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy())
            all_metrics['MAE'].extend(nn.L1Loss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy())
            
            # Evo
            pred_delta = outputs - targets_prev.view(outputs.shape)
            gt_delta = targets.view(outputs.shape) - targets_prev.view(outputs.shape)
            cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
            cur_norm_sq = torch.sum(gt_delta ** 2).item()
            total_evo_error_sq += cur_diff_sq
            total_evo_energy_sq += cur_norm_sq
            
            # PCC
            all_metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
            
            # PCR
            # arch_masks already computed and filtered above
            pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
            pred_clean = pred_clamped.clone()
            pred_clean[pred_clean < 0.05] = 0.0 # Standard threshold
            
            masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
            total = pred_clean.sum(dim=(1,2,3)) + 1e-6
            valid_idx = total > 1e-3
            if valid_idx.any():
                all_metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())
                
            # SSIM
            if SKIMAGE_AVAILABLE:
                np_pred = pred_clamped.cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                for p, t in zip(np_pred, np_tgt):
                    all_metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

    if kept_samples == 0:
        print(f"[WARNING] All samples filtered out for {task_name}!")
        return None

    print(f"  > Kept {kept_samples}/{total_samples} samples (PCR >= 0.5)")
    
    global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
    
    return {
        'MSE': np.mean(all_metrics['MSE']),
        'MAE': np.mean(all_metrics['MAE']),
        'SSIM': np.mean(all_metrics['SSIM']) if SKIMAGE_AVAILABLE else 0.0,
        'PCC': np.mean(all_metrics['PCC']),
        'PCR': np.mean(all_metrics['PCR']),
        'Evo': global_evo
    }

def main():
    print("=" * 80)
    print("Batch Evaluation: Baseline Robustness Experiments")
    print("=" * 80)
    
    all_results = []
    
    # Stress
    for model_type in BASELINE_MODELS:
        res = {'Task': 'stress', 'Model': model_type}
        for seed in SEEDS:
            # Baseline naming: best_baseline_{task}_{TYPE}_seed{seed}.pth
            if model_type == "MAMBA":
                # Use Full Dual Mamba from Ablation results
                ablation_dir = ROBUSTNESS_DIR_STRESS.replace("_baselines", "") # Remove _baselines suffix
                fpath = os.path.join(ablation_dir, f"best_stress_full_dual_seed{seed}.pth")
            else:
                fpath = os.path.join(ROBUSTNESS_DIR_STRESS, f"best_baseline_stress_{model_type}_seed{seed}.pth")
            
            if not os.path.exists(fpath):
                print(f"[Skip] {fpath} not found") # Changed fname to fpath
                continue
                
            print(f"\n[Stress] Eval {model_type} Seed {seed}")
            metrics = evaluate_model(fpath, DATASET_DIR_STRESS, PARAMS_JSON_STRESS, model_type, f"stress_{model_type}_{seed}", STATIC_FEATURES_STRESS)
            if metrics: res[f'Seed{seed}'] = metrics
            
        all_results.append(res)
        
    # Subsidence
    for model_type in BASELINE_MODELS:
        res = {'Task': 'subsidence', 'Model': model_type}
        for seed in SEEDS:
            if model_type == "MAMBA":
                # Use Full Dual Mamba from Ablation results
                ablation_dir = ROBUSTNESS_DIR_SUBSIDENCE.replace("_baselines", "")
                fpath = os.path.join(ablation_dir, f"best_subsidence_full_dual_seed{seed}.pth")
            else:
                fpath = os.path.join(ROBUSTNESS_DIR_SUBSIDENCE, f"best_baseline_subsidence_{model_type}_seed{seed}.pth")
            
            if not os.path.exists(fpath):
                print(f"[Skip] {fpath} not found")
                continue
                
            print(f"\n[Subsidence] Eval {model_type} Seed {seed}")
            # HARDCODED FIX: Explicitly passing 11
            metrics = evaluate_model(fpath, DATASET_DIR_SUBSIDENCE, PARAMS_JSON_SUBSIDENCE, model_type, f"subsidence_{model_type}_{seed}", 11)
            if metrics: res[f'Seed{seed}'] = metrics
            
        all_results.append(res)

    print("\n" + "=" * 80)
    print("Calculating Statistics...")
    print("=" * 80)
    
    csv_rows = []
    for r in all_results:
        task = r['Task']
        model = r['Model']
        
        seeds_found = [k for k in r.keys() if k.startswith('Seed')]
        if not seeds_found: continue
        
        metrics_list = ['MSE', 'MAE', 'SSIM', 'PCC', 'PCR', 'Evo']
        
        for metric in metrics_list:
            vals = [r[s][metric] for s in seeds_found]
            
            if len(vals) >= 2:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                t_val = stats.t.ppf(0.975, df=len(vals)-1)
                ci_margin = t_val * std_val / np.sqrt(len(vals))
                ci_low = mean_val - ci_margin
                ci_high = mean_val + ci_margin
            else:
                mean_val = vals[0]
                std_val = 0.0
                ci_low = mean_val
                ci_high = mean_val
                
            row = {
                'Task': task, 'Model': model, 'Metric': metric,
                'Mean': mean_val, 'Std': std_val, 'CI_Lower': ci_low, 'CI_Upper': ci_high
            }
            # Fill individual seeds
            for seed in SEEDS:
                key = f'Seed{seed}'
                row[key] = r[key][metric] if key in r else None
                
            csv_rows.append(row)
            
    df = pd.DataFrame(csv_rows)
    # Reorder columns
    cols = ['Task', 'Model', 'Metric'] + [f'Seed{s}' for s in SEEDS] + ['Mean', 'Std', 'CI_Lower', 'CI_Upper']
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Baseline Result saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
