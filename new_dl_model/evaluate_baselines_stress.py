# -*- coding: utf-8 -*-
"""
Physics-Consistent Baseline Evaluation for STRESS (Matches Main Model Logic).

Features:
1. GT-PCR Filtering: Skips samples where Ground Truth violates physics masks.
   Controlled by --min_gt_pcr (Default: 0.5).
2. Theory-Consistent Masks: Uses the ODE-based mask generator.
3. Strict Metric Alignment: MSE, MAE, SSIM, PCC, Evo, PCR (Thresholds matched).
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import math

# --- Dependencies Check ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM metrics will be skipped.")

# --- 0. Configuration ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models_baselines_stress")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

STATIC_FEATURES = 17      
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
STEP_DISTANCE_M = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Physics Mask Generator (From Your Script) ---

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
            # [FIX] Transpose Logic
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
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
                    if y_prev.ndim == 1: y_prev = y_prev.reshape(IMG_SIZE, IMG_SIZE)
                    y_prev = y_prev.T
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)

        # Physics Params Loading (Required for Mask Generator)
        default_params = {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]}
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

def calculate_pcc(pred, target):
    vx = pred - torch.mean(pred, dim=(1,2,3), keepdim=True)
    vy = target - torch.mean(target, dim=(1,2,3), keepdim=True)
    cost = torch.sum(vx * vy, dim=(1,2,3))
    den = torch.sqrt(torch.sum(vx ** 2, dim=(1,2,3)) * torch.sum(vy ** 2, dim=(1,2,3))) + 1e-8
    return (cost / den).cpu().numpy()

# --- 3. Baseline Models (Fixed Versions) ---

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

class MambaBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, d_model=64, n_layers=1):
        super(MambaBaseline, self).__init__()
        from mamba_ssm import Mamba
        self.dynamic_len = dynamic_len; self.static_len = static_len
        self.embedding = nn.Linear(1 + static_len, d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        x = self.embedding(seq_input).float(); x = torch.clamp(x, min=-5.0, max=5.0)
        for layer, norm in zip(self.layers, self.norms):
            residual = x; x = norm(x); out = layer(x)
            x = residual + out * 0.1
        x = self.final_norm(x).mean(dim=1)
        return self.decoder(torch.nan_to_num(x))

# --- 4. Main Evaluation Logic ---

def evaluate(args):
    print("================================================================")
    print("  Physics-Consistent Baseline Evaluation (STRESS) - Filtered")
    print(f"  Condition: GT-PCR >= {args.min_gt_pcr}")
    print("================================================================")
    
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    all_files.sort()
    
    # Auto-detect Dimensions
    with np.load(all_files[0]) as f: 
        total_dim = f['x'].shape[0]
        dynamic_feats = total_dim - STATIC_FEATURES
    
    # [CRITICAL UPDATE] Load Stats from Training
    stats_path = os.path.join(MODEL_DIR, "baseline_stress_stats.pt")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        stats = torch.load(stats_path)
        mean, std = stats['mean'], stats['std']
    else:
        print("Stats file not found! Calculating on FULL training set (Fallback)...")
        # Fallback to full calc
        temp_loader = DataLoader(SequentialStressDataset(all_files, PARAMS_JSON_PATH), batch_size=128)
        all_x = []
        for x, _, _, _, _ in tqdm(temp_loader, desc="Calculating Stats"):
            all_x.append(x)
        x_tensor = torch.cat(all_x, dim=0)
        mean, std = x_tensor.mean(dim=0), x_tensor.std(dim=0)
        std[std < 1e-6] = 1.0

    transform = NormalizeTransform(mean, std)
    
    # [FIX] Build Global File Index
    print("Building Global File Index...")
    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample"); st_pos = fn.rfind("step")
            s_id = int(fn[s_pos+7 : s_pos+11])
            st_id = int(fn[st_pos+5 : st_pos+8])
            global_index_map[(s_id, st_id)] = fp
        except: continue
    
    # Validation Split
    np.random.seed(42); np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    val_dataset = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(DEVICE)
    
    model_paths = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    results_table = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n> Evaluating: {model_name}")
        
        if "CNN" in model_name: model = DeepCNNBaseline(input_size=total_dim).to(DEVICE)
        elif "LSTM" in model_name: model = BiLSTMBaseline(dynamic_len=dynamic_feats, static_len=STATIC_FEATURES).to(DEVICE)
        elif "MAMBA" in model_name: model = MambaBaseline(dynamic_len=dynamic_feats, static_len=STATIC_FEATURES, d_model=64, n_layers=1).to(DEVICE)
        else: continue
            
        try: model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e: print(f"Load failed: {e}"); continue
        model.eval()
        
        metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
        total_samples = 0
        kept_samples = 0
        
        # Global Accums
        total_evo_error_sq = 0.0
        total_evo_energy_sq = 0.0
        
        with torch.no_grad():
            for inputs, targets, targets_prev, dists, phys_params in tqdm(val_loader, leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                targets_prev, dists, phys_params = targets_prev.to(DEVICE), dists.to(DEVICE), phys_params.to(DEVICE)
                
                # --- Step 1: Pre-calculate Mask and GT-PCR ---
                arch_masks = mask_generator(dists, phys_params) # [B, 1, 64, 64]
                tgt_imgs = targets.view(-1, 1, 64, 64)
                
                # Binarize GT for stress (Threshold 0.1 matched to main script)
                gt_bin = (tgt_imgs > 0.1).float()
                
                gt_intersection = (gt_bin * arch_masks).sum(dim=(1,2,3))
                gt_total = gt_bin.sum(dim=(1,2,3)) + 1e-6
                gt_pcr = gt_intersection / gt_total
                
                # --- Step 2: Filtering ---
                keep_indices = gt_pcr >= args.min_gt_pcr
                
                total_samples += inputs.size(0)
                kept_samples += keep_indices.sum().item()
                
                if not keep_indices.any(): continue
                
                # Apply filter
                inputs = inputs[keep_indices]
                targets = targets[keep_indices]
                targets_prev = targets_prev[keep_indices]
                arch_masks = arch_masks[keep_indices]
                tgt_imgs = tgt_imgs[keep_indices]
                
                # --- Step 3: Inference ---
                outputs = model(inputs) # [Batch, 4096]
                
                # --- Step 4: Metrics ---
                
                # Flatten Targets for Value Metrics
                targets_flat = targets.view(targets.size(0), -1)
                targets_prev_flat = targets_prev.view(targets_prev.size(0), -1)
                
                metrics['MSE'].extend(nn.MSELoss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
                metrics['MAE'].extend(nn.L1Loss(reduction='none')(outputs, targets_flat).mean(dim=1).cpu().numpy())
                
                # Evo: Global accumulation
                pred_delta = outputs - targets_prev_flat
                gt_delta = targets_flat - targets_prev_flat
                
                cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
                cur_norm_sq = torch.sum(gt_delta ** 2).item()
                
                total_evo_error_sq += cur_diff_sq
                total_evo_energy_sq += cur_norm_sq
                
                # Reshape for Image Metrics
                pred_imgs = outputs.view(-1, 1, 64, 64)
                
                # PCC
                metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
                
                # Model PCR (Threshold 0.15 matched to main script)
                pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
                pred_clean = pred_clamped.clone()
                pred_clean[pred_clean < 0.15] = 0.0 # Denoise Threshold
                
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

        # Result Aggregation
        if kept_samples > 0:
            res = {k: np.mean(v) for k, v in metrics.items()}
            # Global Evo Calc
            res["Evo"] = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
        else:
            res = {k: 0.0 for k in metrics.keys()}
            res["Evo"] = 0.0

        res["Model"] = model_name.replace("best_baseline_stress_", "").replace(".pth", "")
        results_table.append(res)
        
        print(f"  [Samples: {kept_samples}/{total_samples}] MSE: {res['MSE']:.6f} | PCR: {res['PCR']:.4f}")

    # --- Final Table ---
    print("\n" + "="*95)
    print(f"{'Model':<20} | {'MSE':<9} | {'MAE':<9} | {'SSIM':<7} | {'PCC':<7} | {'Evo':<9} | {'PCR':<7}")
    print("-" * 95)
    for res in results_table:
        print(f"{res['Model']:<20} | {res['MSE']:.5f}   | {res['MAE']:.5f}   | {res['SSIM']:.4f}  | {res['PCC']:.4f}  | {res['Evo']:.5f}   | {res['PCR']:.4f}")
    print("="*95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_gt_pcr", type=float, default=0.5, 
                        help="Minimum Ground Truth PCR to include sample in evaluation.")
    args = parser.parse_args()
    evaluate(args)