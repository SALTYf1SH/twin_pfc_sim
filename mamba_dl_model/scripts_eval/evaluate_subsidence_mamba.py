# -*- coding: utf-8 -*-
"""
Physics-Informed SUBSIDENCE Model Evaluation Script (FINAL FIXED with Stats Loading)

Crucial Fixes:
1. [Normalization] LOADS 'subsidence_stats.pt' from training to ensure exact distribution match.
   (Falls back to on-the-fly calculation with Zero-Variance Protection if file missing).
2. [Dataset] Corrects shape handling (1D input, Transposed 2D target).
3. [Physics] Uses ODE-based Mask Generator for accurate PCR.
"""

import os
import glob
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import math
from torch.utils.data import Dataset, DataLoader
# [Improvement] Replacing Transformer with Mamba
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
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset") 
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

# Default paths
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../trained_models_subsidence_physics_mamba/best_subsidence_full_dual.pth")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "../evaluation_results_subsidence_mamba")

STATIC_FEATURES = 11      
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH 

STEP_DISTANCE_M = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Physics Mask Generator (ODE-based) ---

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
        h_max = phys_params[:, 0]; w_arch = phys_params[:, 1]; beta = phys_params[:, 2]
        lag = phys_params[:, 3]; k = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]; ks_b = phys_params[:, 7:9]
        
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

# --- 2. Metrics Helpers ---

def calculate_pcc(pred, target):
    vx = pred - torch.mean(pred, dim=(1,2,3), keepdim=True)
    vy = target - torch.mean(target, dim=(1,2,3), keepdim=True)
    cost = torch.sum(vx * vy, dim=(1,2,3))
    den = torch.sqrt(torch.sum(vx ** 2, dim=(1,2,3)) * torch.sum(vy ** 2, dim=(1,2,3))) + 1e-8
    return (cost / den).cpu().numpy()

# --- 3. Dataset & Model ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialSubsidenceDataset(Dataset):
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
            
            # [Fix] Keep x as 1D, Reshape y to 2D + Transpose
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

        default_params = {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]}
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, y_prev, np.float32(mining_dist), phys_vec, os.path.basename(curr_path)

# --- 3. Model Architecture (Dual-Branch Mamba) ---

class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, 
                 d_model=128, n_layers=2, dropout=0.1): # Tuned for stability
        super(DualBranchMambaModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
        # 1. Static Branch (MLP) - Unchanged
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(64, 32), 
            nn.ReLU()
        )
        
        # 2. Dynamic Branch (Mamba) - [NEW]
        # We process the sequence of mining steps using Mamba
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
            raise ImportError("mamba_ssm is required for this model")

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
        static_out = self.static_branch(x_static)
        
        # Branch 2: Dynamic (Mamba)
        # Reshape for Mamba: [Batch, Seq_Len, 1] -> Embedding -> [Batch, Seq_Len, d_model]
        x_dynamic_seq = x_dynamic.unsqueeze(-1)
        x_mamba = self.dynamic_embedder(x_dynamic_seq)
        
        # Mamba Forward
        for layer, norm in zip(self.mamba_layers, self.norms):
            residual = x_mamba
            x_mamba = norm(x_mamba)
            x_mamba = layer(x_mamba)
            x_mamba = residual + x_mamba # Residual connection
            
        # Pooling: Mamba usually uses the last state or mean. 
        # For physical evolution, Mean Pooling captures the "cumulative" effect well.
        dynamic_out = x_mamba.mean(dim=1) 
        
        # Fusion
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 4. Evaluation Main ---

def evaluate(args):
    print("========================================================")
    print("   Physics-Informed SUBSIDENCE Model Evaluation (Final) ")
    print("========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Dataset Prep
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    # 2. Stats Loading (Match Training Distribution)
    # [CRITICAL] Try to load stats saved during training
    train_dir = os.path.dirname(args.model_path)
    stats_path = os.path.join(train_dir, "stress_stats.pt")
    
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        # [Fix] 强制加载到 CPU，因为 Dataset.__getitem__ 在 CPU 上运行
        stats = torch.load(stats_path, map_location='cpu') 
        mean = stats['mean']
        std = stats['std']
    else:
        print("[WARNING] Stats file not found! Calculating on-the-fly (Fallback).")
        # Fallback: Calculate on random subset with protection
        np.random.seed(42)
        random_files = np.random.choice(all_files, min(2000, len(all_files)), replace=False)
        temp_dataset = SequentialSubsidenceDataset(random_files, "") 
        temp_loader = DataLoader(temp_dataset, batch_size=32)
        
        all_x = []
        for data in temp_loader:
            x = data[0]
            all_x.append(x)
        x_tensor = torch.cat(all_x, dim=0)
        mean = x_tensor.mean(dim=0)
        std = x_tensor.std(dim=0)
        
        # [Protection] Zero Variance Fix
        std[std < 1e-6] = 1.0 
        print(f"Fallback Stats - Max Std: {std.max():.2f}")

    transform = NormalizeTransform(mean, std)
    
    # [FIX] Build Global File Index
    print("Building Global File Index for Sequential Lookup...")
    global_index_map = {}
    for fp in all_files:
        try:
            fn = os.path.basename(fp)
            s_pos = fn.rfind("sample"); s_start = s_pos + 7; s_id = int(fn[s_start : s_start + 4])
            st_pos = fn.rfind("step"); st_start = st_pos + 5; st_id = int(fn[st_start : st_start + 3])
            global_index_map[(s_id, st_id)] = fp
        except: continue
    
    # Split
    np.random.seed(42); np.random.shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    val_files = all_files[split_idx:]
    
    # 3. Model Load
    with np.load(all_files[0]) as f: total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES
    
    model = DualBranchMambaModel(STATIC_FEATURES, dynamic_feats, OUTPUT_FEATURES).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model: {args.model_path}")
    except Exception as e:
        print(f"Failed to load model weights: {e}"); return
    model.eval()
    
    # 4. Evaluation Loop
    # Pass global_index_map
    val_dataset = SequentialSubsidenceDataset(val_files, PARAMS_JSON_PATH, transform=transform, global_index_map=global_index_map)
    
    if args.all:
        print(f"\n[MODE] Statistical Evaluation on FULL Validation Set ({len(val_files)} samples)")
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(device)
        
        all_metrics = {'MSE': [], 'MAE': [], 'SSIM': [], 'PCC': [], 'PCR': []}
        gt_pcr_list = []
        
        # New: Global accumulations
        total_evo_error_sq = 0.0
        total_evo_energy_sq = 0.0
        
        with torch.no_grad():
            for inputs, targets, targets_prev, dists, phys_params, _ in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets_prev, dists, phys_params = targets_prev.to(device), dists.to(device), phys_params.to(device)
                
                outputs = model(inputs)
                pred_imgs = outputs.view(-1, 1, 64, 64)
                tgt_imgs = targets.view(-1, 1, 64, 64)
                
                # 1. MSE/MAE
                all_metrics['MSE'].extend(nn.MSELoss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy())
                all_metrics['MAE'].extend(nn.L1Loss(reduction='none')(outputs, targets.view(targets.size(0), -1)).mean(dim=1).cpu().numpy())
                
                # 2. Evo: Global accumulation
                pred_delta = outputs - targets_prev.view(outputs.shape)
                gt_delta = targets.view(outputs.shape) - targets_prev.view(outputs.shape)
                
                cur_diff_sq = torch.sum((pred_delta - gt_delta) ** 2).item()
                cur_norm_sq = torch.sum(gt_delta ** 2).item()
                
                total_evo_error_sq += cur_diff_sq
                total_evo_energy_sq += cur_norm_sq
                
                # 3. PCC
                all_metrics['PCC'].extend(calculate_pcc(pred_imgs, tgt_imgs))
                
                # 4. PCR (with Thresholding)
                arch_masks = mask_generator(dists, phys_params)
                pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
                pred_clean = pred_clamped.clone()
                pred_clean[pred_clean < 0.05] = 0.0
                
                masked = (pred_clean * arch_masks).sum(dim=(1,2,3))
                total = pred_clean.sum(dim=(1,2,3)) + 1e-6
                valid_idx = total > 1e-3
                if valid_idx.any():
                    all_metrics['PCR'].extend((masked[valid_idx] / total[valid_idx]).cpu().numpy())
                    
                # 5. GT PCR (Benchmark)
                gt_masked = (tgt_imgs * arch_masks).sum(dim=(1,2,3))
                gt_total = tgt_imgs.sum(dim=(1,2,3)) + 1e-6
                gt_pcr_list.extend((gt_masked / gt_total).cpu().numpy())
                
                # 6. SSIM
                if SKIMAGE_AVAILABLE:
                    np_pred = pred_clamped.cpu().numpy(); np_tgt = tgt_imgs.cpu().numpy()
                    for p, t in zip(np_pred, np_tgt):
                        all_metrics['SSIM'].append(ssim(t.squeeze(), p.squeeze(), data_range=max(t.max(), 1e-6)))

        # Calculate Global Evo
        global_evo = math.sqrt(total_evo_error_sq / (total_evo_energy_sq + 1e-8))

        print("\n=== Subsidence Model Performance ===")
        print(f"MSE : {np.mean(all_metrics['MSE']):.6f}")
        print(f"MAE : {np.mean(all_metrics['MAE']):.6f}")
        print(f"SSIM: {np.mean(all_metrics['SSIM']):.4f}")
        print(f"PCC : {np.mean(all_metrics['PCC']):.4f}")
        print(f"Evo : {global_evo:.6f} (Global Rel_RMSE)")
        print(f"PCR : {np.mean(all_metrics['PCR']):.4f} (Ours)")
        print(f"GT-PCR: {np.mean(gt_pcr_list):.4f} (Benchmark)")
        
    else:
        print(f"\n[MODE] Visualization Preview (Max {args.num_samples} samples)")
        print("To run full statistics, use flag: --all")
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        count = 0
        
        for inputs, targets, _, _, _, fname in val_loader:
            if count >= args.num_samples: break
            
            outputs = model(inputs.to(device))
            pred_img = outputs.view(64, 64).detach().cpu().numpy()
            tgt_img = targets.view(64, 64).numpy()
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            im1 = ax[0].imshow(tgt_img, cmap='jet', vmin=0, vmax=1); ax[0].set_title("GT")
            im2 = ax[1].imshow(pred_img, cmap='jet', vmin=0, vmax=1); ax[1].set_title("Pred")
            plt.colorbar(im1, ax=ax[0]); plt.colorbar(im2, ax=ax[1])
            plt.suptitle(f"Sample: {fname[0]}")
            plt.savefig(os.path.join(args.output_dir, f"sub_vis_sample_{count}.png"))
            plt.close()
            print(f"Saved visualization: sub_vis_sample_{count}.png")
            count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Run full stats on validation set")
    args = parser.parse_args()
    evaluate(args)