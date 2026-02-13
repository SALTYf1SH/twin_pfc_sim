# -*- coding: utf-8 -*-
"""
Ablation Study Evaluation Script (Strict Alignment with Main Eval)

Strict Alignments:
1. GT-PCR Filtering: Automatically skips samples where GT violates physics (default threshold 0.5).
2. Mask Generator: Uses TheoryConsistentMaskGenerator (ODE-based).
3. Metric Logic: 
   - Evo: (Pred_t - GT_{t-1}) vs (GT_t - GT_{t-1})
   - PCR: Pred threshold 0.15, GT threshold 0.1
   - MSE/MAE: Flattened calculation.
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

# --- Dependencies Check ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM metrics will be skipped.")

# --- 0. Configuration ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset_stress")
MODEL_DIR = os.path.join(BASE_DIR, "../trained_models_stress_ablation_mamba") # New Dir for Mamba Ablation
OUTPUT_DIR = os.path.join(BASE_DIR, "../evaluation_results_stress_mamba")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_summary_mamba.csv")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json")

STATIC_FEATURES = 17      
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 
STEP_DISTANCE_M = 10.0
TRAIN_VAL_SPLIT_RATIO = 0.9
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ablation Experiments List
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

# --- 1. Physics Mask Generator (ODE-based) ---

class TheoryConsistentMaskGenerator(nn.Module):
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        
        y_vals = torch.linspace(0, 150.0, self.H) 
        x_vals = torch.linspace(0, 500.0, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid); self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        batch_size = dist.size(0); max_d = dist.max().item()
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        h_curr = torch.zeros(batch_size, device=dist.device); h_trace = [h_curr.clone()]
        for _ in range(steps):
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh; h_curr = torch.min(h_curr, h_max)
            h_trace.append(h_curr.clone())
        h_trace = torch.stack(h_trace)
        indices = dist.long().clamp(max=steps)
        h_final = h_trace.gather(0, indices.unsqueeze(0)).squeeze(0)
        return h_final

    def forward(self, mining_distances, phys_params):
        h_max = phys_params[:, 0]; w_arch = phys_params[:, 1]; beta = phys_params[:, 2]
        lag = phys_params[:, 3]; k = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]; ks_b = phys_params[:, 7:9]
        curr_H = self.solve_height_ode(mining_distances, h_max, k, ks_h, ks_b)
        
        xc = (mining_distances - lag).view(-1, 1, 1); curr_H = curr_H.view(-1, 1, 1)
        w_arch = w_arch.view(-1, 1, 1); beta_shape = beta.view(-1, 1, 1)
        
        x_term = (self.xx.unsqueeze(0) - xc) / (w_arch + 1e-6)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta_shape)
        
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        is_front = (self.xx.unsqueeze(0) > xc)
        
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(mining_distances.size(0), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        mask = (self.yy.unsqueeze(0) <= height_limit).float()
        return mask.unsqueeze(1)

# --- 2. Dataset & Utilities ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None, global_index_map=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.global_index_map = global_index_map # [FIX] Allow global lookup
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
        s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
        s_id = int(filename[s_pos+7 : s_pos+11])
        st_id = int(filename[st_pos+5 : st_pos+8])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T # [CRITICAL] Transpose to match main model logic
            
        if st_id > 1:
            # [FIX] Try Global Map first, then Local Map
            prev_path = None
            if self.global_index_map is not None:
                prev_path = self.global_index_map.get((s_id, st_id - 1))
            
            if prev_path is None:
                # Fallback to local index map
                prev_idx = self.index_map.get((s_id, st_id - 1))
                if prev_idx is not None:
                    prev_path = self.file_list[prev_idx]

            if prev_path is not None:
                with np.load(prev_path) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
                    if y_prev.ndim == 1: y_prev = y_prev.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev = y_prev.T
                    x_prev = torch.from_numpy(data['x'].astype(np.float32))
            else: 
                # Only if truly missing
                y_prev = y_t.clone()
                x_prev = x_t.clone()
        else: 
            y_prev = torch.zeros_like(y_t)
            x_prev = torch.zeros_like(x_t)

        default_params = {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]}
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: 
            x_t = self.transform(x_t)
            x_prev = self.transform(x_prev)
            
        return x_t, x_prev, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 3. Model Definition (DualBranch) ---

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
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, output_size))
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
        
        # Positional Encoding Internal
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

# --- 4. Single Experiment Evaluation (With Filtering) ---

def evaluate_single_experiment(exp_config, val_loader, device, feature_dims, min_gt_pcr):
    ablation, branch = exp_config['ablation'], exp_config['branch']
    model_name = f"best_stress_{ablation}_{branch}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    print(f"Evaluating: {exp_config['name']} ...")
    if not os.path.exists(model_path):
        print(f"  [Skip] Model not found: {model_name}"); return None

    # Load Model
    model = DualBranchMambaModel(
        static_size=feature_dims['static'], dynamic_size=feature_dims['dynamic'],
        output_size=OUTPUT_FEATURES, branch_mode=branch
    ).to(device)
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except: print("  [Error] Load failed"); return None
    model.eval()
    
    # Init Metrics & Mask Generator
    mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(device)
    metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
    
    total_samples = 0
    kept_samples = 0
    
    # Global Accums
    total_evo_error_sq = 0.0
    total_evo_energy_sq = 0.0
    
    with torch.no_grad():
        for inputs, inputs_prev, targets, targets_prev, dists, phys_params in tqdm(val_loader, leave=False):
            inputs, inputs_prev = inputs.to(device), inputs_prev.to(device)
            targets, targets_prev = targets.to(device), targets_prev.to(device)
            dists, phys_params = dists.to(device), phys_params.to(device)
            
            # --- [Step 1] GT-PCR Filtering (Strictly matches Main Eval) ---
            arch_masks = mask_generator(dists, phys_params) # [B, 1, 64, 64]
            tgt_imgs = targets.view(-1, 1, 64, 64)
            
            # Binarize GT (Threshold 0.1)
            gt_bin = (tgt_imgs > 0.1).float()
            gt_pcr = (gt_bin * arch_masks).sum(dim=(1,2,3)) / (gt_bin.sum(dim=(1,2,3)) + 1e-6)
            
            keep_indices = gt_pcr >= min_gt_pcr
            
            total_samples += inputs.size(0)
            kept_samples += keep_indices.sum().item()
            
            if not keep_indices.any(): continue
            
            # Apply Filter
            inputs = inputs[keep_indices]
            inputs_prev = inputs_prev[keep_indices]
            targets = targets[keep_indices]
            targets_prev = targets_prev[keep_indices]
            arch_masks = arch_masks[keep_indices]
            tgt_imgs = tgt_imgs[keep_indices]
            
            # --- [Step 2] Model Inference ---
            outputs = model(inputs)
            # outputs_prev is no longer needed for Evo (we use Targets_Prev)

            
            pred_imgs = outputs.view(-1, 1, 64, 64)
            
            # --- [Step 3] Metrics Calculation ---
            
            # Flatten for MSE/MAE/Evo
            targets_flat = targets.view(targets.size(0), -1)
            targets_prev_flat = targets_prev.view(targets_prev.size(0), -1)
            
            # MSE & MAE
            metrics['MSE'].extend(nn.MSELoss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
            metrics['MAE'].extend(nn.L1Loss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
            
            # Evo: Global accumulation
            # Metric = sqrt(sum((Delta_pred - Delta_gt)^2) / (sum(Delta_gt^2) + epsilon))
            pred_delta = outputs - targets_prev_flat
            gt_delta = targets_flat - targets_prev_flat
            
            cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
            cur_norm_sq = torch.sum(gt_delta ** 2).item()
            
            total_evo_error_sq += cur_diff_sq
            total_evo_energy_sq += cur_norm_sq
            
            # PCC
            metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
            
            # PCR (Model Threshold 0.15)
            pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
            pred_clean = pred_clamped.clone()
            pred_clean[pred_clean < 0.15] = 0.0
            
            masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
            total = pred_clean.sum(dim=(1,2,3)) + 1e-6
            valid_idx = total > 1e-3
            if valid_idx.any():
                metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())
            
            # SSIM
            if SKIMAGE_AVAILABLE:
                np_pred = pred_clamped.cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                for p, t in zip(np_pred, np_tgt):
                    metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

    if kept_samples == 0:
        print("  [Warning] No samples passed the filter.")
        return None

    # Calculate Global Evo
    global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
    
    res = {k: np.mean(v) for k, v in metrics.items()}
    res["Evo"] = global_evo
    res["Model"] = exp_config['name']
    
    # Console Output
    print(f"  [Kept {kept_samples}/{total_samples}] MSE:{res['MSE']:.5f} | Global_Evo:{res['Evo']:.4f} | PCR:{res['PCR']:.4f}")
    return res

# --- 5. Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_gt_pcr", type=float, default=0.5, help="GT-PCR Filter Threshold")
    args = parser.parse_args()
    
    print("========================================================")
    print("      Ablation Evaluation (Filtered & Strict)           ")
    print(f"      Threshold: GT-PCR >= {args.min_gt_pcr}          ")
    print("========================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    # Data Split
    np.random.seed(42); np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    print(f"Validation Set: {len(val_files)} samples")
    
    # [FIX] Build Global Index Map
    print("Building Global File Index for Sequential Lookup...")
    global_index_map = {}
    for fp in all_files:
        filename = os.path.basename(fp)
        try:
            s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
            s_id = int(filename[s_pos+7 : s_pos+11])
            st_id = int(filename[st_pos+5 : st_pos+8])
            global_index_map[(s_id, st_id)] = fp
        except: continue
        
    # Stats Calculation (Full Training Set)
    print("Calculating/Loading Stats...")
    transform = None
    stats_path = os.path.join(MODEL_DIR, "stress_stats_ablation.pt")
    
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        transform = NormalizeTransform(stats['mean'], stats['std'])
    else:
        # Fallback Calculation
        temp_loader = DataLoader(SequentialStressDataset(train_files, "", global_index_map=global_index_map), batch_size=128)
        all_x = []
        for x, _, _, _, _, _ in tqdm(temp_loader, desc="Calc Stats"): all_x.append(x)
        x_tensor = torch.cat(all_x, dim=0)
        transform = NormalizeTransform(x_tensor.mean(dim=0), x_tensor.std(dim=0))

    # Loader with Global Map
    val_dataset = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get Dims
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    feature_dims = {'static': STATIC_FEATURES, 'dynamic': total_feats - STATIC_FEATURES}
    
    # Run Experiments
    # Define Models to Evaluate
    # Format: (Name, Path, Type, Config)
    
    # 1. Our Proposed Mamba Model
    mamba_model_path = os.path.join(BASE_DIR, "../trained_models_stress_physics_mamba", "best_stress_full_dual.pth") # Assumes trained
    
    # 2. Transformer Baseline (Our previous SOTA)
    transformer_dir = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_stress_ablation")
    transformer_path = os.path.join(transformer_dir, "best_stress_full_dual.pth")
    
    # 3. Standard Baselines (CNN, LSTM)
    baseline_dir = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_baselines_stress")
    cnn_path = os.path.join(baseline_dir, "best_baseline_stress_CNN.pth")
    lstm_path = os.path.join(baseline_dir, "best_baseline_stress_LSTM.pth")
    
    models_to_eval = [
        {"name": "Proposed (Mamba)", "path": mamba_model_path, "type": "Mamba", "branch": "dual"},
        {"name": "Transformer (Dual)", "path": transformer_path, "type": "Transformer", "branch": "dual"},
        {"name": "Start Baseline (CNN)", "path": cnn_path, "type": "CNN", "branch": None},
        {"name": "Seq Baseline (LSTM)", "path": lstm_path, "type": "LSTM", "branch": None},
    ]

    all_results = []
    
    for model_cfg in models_to_eval:
        print(f"Evaluating: {model_cfg['name']}...")
        if not os.path.exists(model_cfg['path']):
            print(f"  [Warning] Model file not found: {model_cfg['path']}")
            continue
            
        # Init Model
        if model_cfg['type'] == 'Mamba':
             model = DualBranchMambaModel(
                static_size=feature_dims['static'], dynamic_size=feature_dims['dynamic'],
                output_size=OUTPUT_FEATURES, branch_mode=model_cfg['branch']
            ).to(DEVICE)
        elif model_cfg['type'] == 'Transformer':
             model = TransformerDualBranchModel(
                static_size=feature_dims['static'], dynamic_size=feature_dims['dynamic'],
                output_size=OUTPUT_FEATURES
             ).to(DEVICE)
        elif model_cfg['type'] == 'CNN':
             model = DeepCNNBaseline(input_size=feature_dims['static'] + feature_dims['dynamic']).to(DEVICE)
        elif model_cfg['type'] == 'LSTM':
             model = BiLSTMBaseline(dynamic_len=feature_dims['dynamic'], static_len=feature_dims['static']).to(DEVICE)
        
        try: 
            model.load_state_dict(torch.load(model_cfg['path'], map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f"  [Error] Failed to load model: {e}")
            continue

        # Init Metrics & Mask Generator
        mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(DEVICE)
        metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
        
        total_samples = 0
        kept_samples = 0
        
        # Global Accums
        total_evo_error_sq = 0.0
        total_evo_energy_sq = 0.0
        
        with torch.no_grad():
            for inputs, inputs_prev, targets, targets_prev, dists, phys_params in tqdm(val_loader, leave=False):
                inputs, inputs_prev = inputs.to(DEVICE), inputs_prev.to(DEVICE)
                targets, targets_prev = targets.to(DEVICE), targets_prev.to(DEVICE)
                dists, phys_params = dists.to(DEVICE), phys_params.to(DEVICE)
                
                # --- [Step 1] GT-PCR Filtering ---
                arch_masks = mask_generator(dists, phys_params) # [B, 1, 64, 64]
                tgt_imgs = targets.view(-1, 1, 64, 64)
                
                # Binarize GT (Threshold 0.1)
                gt_bin = (tgt_imgs > 0.1).float()
                gt_pcr = (gt_bin * arch_masks).sum(dim=(1,2,3)) / (gt_bin.sum(dim=(1,2,3)) + 1e-6)
                
                keep_indices = gt_pcr >= args.min_gt_pcr
                
                total_samples += inputs.size(0)
                kept_samples += keep_indices.sum().item()
                
                if not keep_indices.any(): continue
                
                # Apply Filter
                inputs = inputs[keep_indices]
                inputs_prev = inputs_prev[keep_indices]
                targets = targets[keep_indices]
                targets_prev = targets_prev[keep_indices]
                arch_masks = arch_masks[keep_indices]
                tgt_imgs = tgt_imgs[keep_indices]
                
                # --- [Step 2] Model Inference ---
                outputs = model(inputs)
                pred_imgs = outputs.view(-1, 1, 64, 64)
                
                # --- [Step 3] Metrics Calculation ---
                
                targets_flat = targets.view(targets.size(0), -1)
                targets_prev_flat = targets_prev.view(targets_prev.size(0), -1)
                
                # MSE & MAE
                metrics['MSE'].extend(nn.MSELoss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
                metrics['MAE'].extend(nn.L1Loss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
                
                # Evo: Global accumulation
                pred_delta = outputs - targets_prev_flat
                gt_delta = targets_flat - targets_prev_flat
                
                cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
                cur_norm_sq = torch.sum(gt_delta ** 2).item()
                
                total_evo_error_sq += cur_diff_sq
                total_evo_energy_sq += cur_norm_sq
                
                # PCC
                metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
                
                # PCR (Model Threshold 0.15)
                pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
                pred_clean = pred_clamped.clone()
                pred_clean[pred_clean < 0.15] = 0.0
                
                masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
                total = pred_clean.sum(dim=(1,2,3)) + 1e-6
                valid_idx = total > 1e-3
                if valid_idx.any():
                    metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())
                
                # SSIM
                if SKIMAGE_AVAILABLE:
                    np_pred = pred_clamped.cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                    for p, t in zip(np_pred, np_tgt):
                        metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

        if kept_samples == 0:
            print("  [Warning] No samples passed the filter.")
            res = {k: 0.0 for k in metrics}
            res['Evo'] = 0.0
        else:
            # Calculate Global Evo
            global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
            res = {k: np.mean(v) for k, v in metrics.items()}
            res["Evo"] = global_evo
            
        res["Model"] = model_cfg['name']
        print(f"  [Samples: {kept_samples}/{total_samples}] MSE: {res['MSE']:.5f} | Evo: {res['Evo']:.4f} | PCR: {res['PCR']:.4f}")
        all_results.append(res)
            
    # Save CSV
    print("\nWriting summary to CSV...")
    
    # Update Output Name to be baseline specific
    baseline_csv = os.path.join(OUTPUT_DIR, "baseline_comparison_summary.csv")
    
    fieldnames = ["Model", "MSE", "MAE", "SSIM", "PCC", "Evo", "PCR"]
    with open(baseline_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

            
    print(f"Done! Summary saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()