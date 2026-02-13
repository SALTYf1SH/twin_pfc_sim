# -*- coding: utf-8 -*-
"""
Full Metrics Baseline Evaluation for SUBSIDENCE (v2) - FIXED VERSION.
FIXES:
1. Shape Mismatch: Flattens targets for MSE/MAE/Evo calculations.
2. Dataset: Maintains y.T (Transpose) for physical alignment.
3. Mamba: Uses stable architecture (d_model=64, n_layers=1).
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

# --- Dependency Check ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM will be 0.")

# --- 0. Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models_baselines_subsidence")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")

STATIC_FEATURES = 11
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Physics Mask Generator (Unchanged) ---
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
            in_arch = (x_term.abs() <= 1.0).float()
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * in_arch
            
            spatial_mask = (self.y_grid <= y_boundary).float()
            masks.append(spatial_mask)
            
        return torch.stack(masks).unsqueeze(1)

# --- 2. Metrics Helpers ---
def calculate_pcc(pred, target):
    # Pred/Target shapes: [Batch, 1, 64, 64]
    vx = pred - torch.mean(pred, dim=(1,2,3), keepdim=True)
    vy = target - torch.mean(target, dim=(1,2,3), keepdim=True)
    cost = torch.sum(vx * vy, dim=(1,2,3))
    den = torch.sqrt(torch.sum(vx ** 2, dim=(1,2,3)) * torch.sum(vy ** 2, dim=(1,2,3))) + 1e-8
    return (cost / den).cpu().numpy()

# --- 3. Dataset (Fixed Transpose) ---
class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialDataset(Dataset):
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
            x_t = data['x'].astype(np.float32)
            y_raw = data['y'].astype(np.float32)
            
            # [FIX] Transpose Logic preserved
            if y_raw.ndim == 1: 
                y_raw = y_raw.reshape(64, 64)
            y_t = torch.from_numpy(y_raw.T) 
            
            x_t = torch.from_numpy(x_t)
            
        if st_id > 1:
            prev_path = None
            if self.global_index_map is not None:
                prev_path = self.global_index_map.get((s_id, st_id - 1))
            
            if prev_path is None:
                prev_idx = self.index_map.get((s_id, st_id - 1))
                if prev_idx is not None: prev_path = self.file_list[prev_idx]

            if prev_path is not None:
                with np.load(prev_path) as data:
                    y_prev_raw = data['y'].astype(np.float32)
                    if y_prev_raw.ndim == 1: y_prev_raw = y_prev_raw.reshape(64, 64)
                    y_prev = torch.from_numpy(y_prev_raw.T) 
            else: y_prev = y_t.clone()
        else: y_prev = torch.zeros_like(y_t)

        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 4. Models ---

class MambaBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096, d_model=64, n_layers=1):
        super(MambaBaseline, self).__init__()
        from mamba_ssm import Mamba
        self.dynamic_len, self.static_len = dynamic_len, static_len
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

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels, self.init_size = 128, 8
        self.fc = nn.Linear(input_size, 128 * 8 * 8)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        return self.conv_blocks(self.fc(x).view(-1, 128, 8, 8)).view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.dynamic_len, self.static_len = dynamic_len, static_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(), 
            # No Dropout for Subsidence LSTM (Matched to Training)
            nn.Linear(1024, output_size)
        )
    def forward(self, x):
        seq_input = torch.cat([x[:, self.static_len:].unsqueeze(-1), x[:, :self.static_len].unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

# --- 5. Main Evaluation ---

def evaluate():
    print(f"--- Evaluating Baselines (SUBSIDENCE) with Fixed Alignment ---")
    
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    all_files.sort()
    
    # Auto-detect Dimensions
    with np.load(all_files[0]) as f: 
        total_dim = f['x'].shape[0]; dynamic_feats = total_dim - STATIC_FEATURES
    
    # [CRITICAL UPDATE] Load Stats from Training
    stats_path = os.path.join(MODEL_DIR, "baseline_subsidence_stats.pt")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        stats = torch.load(stats_path)
        mean, std = stats['mean'], stats['std']
    else:
        print("Stats file not found! Calculating on FULL training set (Fallback)...")
        split_idx = int(0.9 * len(all_files))
        train_files = all_files[:split_idx]
        temp_loader = DataLoader(SequentialDataset(train_files, PARAMS_JSON_PATH), batch_size=128)
        all_x = []
        for x, _, _, _, _ in tqdm(temp_loader, desc="Calculating Stats"):
            all_x.append(x)
        x_tensor = torch.cat(all_x, dim=0)
        mean, std = x_tensor.mean(dim=0), x_tensor.std(dim=0)
        std[std < 1e-6] = 1.0

    transform = NormalizeTransform(mean, std)
    
    # [FIX] Global Index
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

    # Split
    np.random.seed(42); np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    val_dataset = SequentialDataset(val_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    mask_generator = ActivityArchMaskGenerator(output_size=64).to(DEVICE)
    
    model_paths = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    results_table = []
    gt_pcr_list = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n> Evaluating: {model_name}")
        
        if "CNN" in model_name: model = DeepCNNBaseline(input_size=total_dim).to(DEVICE)
        elif "LSTM" in model_name: model = BiLSTMBaseline(dynamic_len=dynamic_feats, static_len=STATIC_FEATURES).to(DEVICE)
        elif "MAMBA" in model_name: 
            model = MambaBaseline(dynamic_len=dynamic_feats, static_len=STATIC_FEATURES, d_model=64, n_layers=1).to(DEVICE)
        else: continue
            
        try: model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e: print(f"Load failed: {e}"); continue
        model.eval()
        
        all_metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
        # Global Accums
        total_evo_error_sq = 0.0
        total_evo_energy_sq = 0.0
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_mae = nn.L1Loss(reduction='none')
        
        with torch.no_grad():
            for inputs, targets, targets_prev, dists, phys_params in tqdm(val_loader, leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                targets_prev, dists, phys_params = targets_prev.to(DEVICE), dists.to(DEVICE), phys_params.to(DEVICE)
                
                outputs = model(inputs) # Shape: [Batch, 4096]
                
                # --- [CRITICAL FIX] Shape Handling ---
                
                # 1. Flatten Targets for Vector Metrics (MSE, MAE, Evo)
                # targets: [Batch, 64, 64] -> [Batch, 4096]
                targets_flat = targets.view(targets.size(0), -1)
                targets_prev_flat = targets_prev.view(targets_prev.size(0), -1)
                
                # 2. Reshape Outputs for Image Metrics (PCC, PCR, SSIM)
                # outputs: [Batch, 4096] -> [Batch, 1, 64, 64]
                pred_imgs = outputs.view(-1, 1, 64, 64)
                tgt_imgs = targets.view(-1, 1, 64, 64) # Add channel dim
                
                # --- Metric Calculation ---
                
                all_metrics['MSE'].extend(criterion_mse(outputs, targets_flat).mean(dim=1).cpu().numpy())
                all_metrics['MAE'].extend(criterion_mae(outputs, targets_flat).mean(dim=1).cpu().numpy())
                
                # Evo: Global accumulation
                pred_delta = outputs - targets_prev_flat
                gt_delta = targets_flat - targets_prev_flat
                
                cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
                cur_norm_sq = torch.sum(gt_delta ** 2).item()
                
                total_evo_error_sq += cur_diff_sq
                total_evo_energy_sq += cur_norm_sq
                
                all_metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
                
                if SKIMAGE_AVAILABLE:
                    np_pred = torch.clamp(pred_imgs, 0, 1).cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                    for p, t in zip(np_pred, np_tgt):
                        all_metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

                # 3. PCR with Thresholding (Uses Image Shapes)
                pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
                arch_masks = mask_generator(dists, phys_params)
                
                pred_clean = pred_clamped.clone()
                pred_clean[pred_clean < 0.05] = 0.0
                
                masked_energy = (pred_clean * arch_masks).sum(dim=(1,2,3))
                total_energy = pred_clean.sum(dim=(1,2,3)) + 1e-6
                
                valid_idx = total_energy > 1e-3
                if valid_idx.any():
                    all_metrics['PCR'].extend((masked_energy[valid_idx] / total_energy[valid_idx]).cpu().numpy())

                # GT PCR (Once)
                if len(results_table) == 0:
                    # Binarize GT (Subsidence Threshold > 0.05)
                    gt_bin = (tgt_imgs > 0.05).float()
                    gt_masked = (gt_bin * arch_masks).sum(dim=(1,2,3))
                    gt_total = gt_bin.sum(dim=(1,2,3)) + 1e-6
                    gt_pcr_list.extend((gt_masked / gt_total).cpu().numpy())

        res = {k: np.mean(v) for k, v in all_metrics.items()}
        # Global Evo
        res['Evo'] = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))
        res["Model"] = model_name.replace("best_baseline_subsidence_", "").replace(".pth", "")
        results_table.append(res)
        print(f"  MSE:{res['MSE']:.5f} | PCR:{res['PCR']:.4f}")

    if gt_pcr_list:
        gt_res = {"Model": "Ground Truth", "MSE": 0.0, "MAE": 0.0, "SSIM": 1.0, "Evo": 0.0, "PCC": 1.0, "PCR": np.mean(gt_pcr_list)}
        results_table.insert(0, gt_res)

    print("\n" + "="*95)
    print(f"{'Model':<20} | {'MSE':<9} | {'MAE':<9} | {'SSIM':<7} | {'PCC':<7} | {'Evo':<9} | {'PCR':<7}")
    print("-" * 95)
    for res in results_table:
        print(f"{res['Model']:<20} | {res['MSE']:.5f}   | {res['MAE']:.5f}   | {res['SSIM']:.4f}  | {res['PCC']:.4f}  | {res['Evo']:.5f}   | {res['PCR']:.4f}")
    print("="*95)

if __name__ == "__main__":
    evaluate()