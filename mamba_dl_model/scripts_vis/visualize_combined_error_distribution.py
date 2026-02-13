
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed.")
    Mamba = None

# --- Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['svg.fonttype'] = 'none'

FIG_WIDTH_CM = 17 
FIG_WIDTH_INCH = FIG_WIDTH_CM / 2.54

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Shared ---
class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, n_layers=2, dropout=0.1): 
        super(DualBranchMambaModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static) if self.branch_mode in ['dual', 'static_only'] else torch.zeros(x.size(0), 32, device=x.device)
        
        if self.branch_mode in ['dual', 'dynamic_only']:
            x_mamba = self.dynamic_embedder(x_dynamic.unsqueeze(-1))
            if hasattr(self, 'mamba_layers'):
                for layer, norm in zip(self.mamba_layers, self.norms):
                    x_mamba = layer(norm(x_mamba)) + x_mamba
                dynamic_out = x_mamba.mean(dim=1)
            else: dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        else: dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        return self.fusion_head(torch.cat((static_out, dynamic_out), dim=1))

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

# --- Task 1: Stress ---
class SequentialStressDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list; self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}

    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        filename = os.path.basename(curr_path)
        s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
        s_id = int(filename[s_pos+7 : s_pos+11]); st_id = int(filename[st_pos+5 : st_pos+8])
        mining_dist = st_id * 10.0
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(64, 64)
            y_t = y_t.T # Corrected
        
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]})
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

class TheoryConsistentMaskGenerator(nn.Module):
    def __init__(self):
        super(TheoryConsistentMaskGenerator, self).__init__()
        y_vals = torch.linspace(0, 150.0, 64); x_vals = torch.linspace(0, 500.0, 64)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid); self.register_buffer('xx', self.x_grid)
        self.ks_sigma = 5.0

    def forward(self, mining_distances, phys_params):
        h_max = phys_params[:, 0]; w_arch = phys_params[:, 1]; beta = phys_params[:, 2]
        lag = phys_params[:, 3]; k = phys_params[:, 4]
        ks_h = phys_params[:, 5:7]; ks_b = phys_params[:, 7:9]
        
        batch_size = mining_distances.size(0); max_d = max(1.0, mining_distances.max().item())
        steps = int(max_d) + 2
        h_curr = torch.zeros(batch_size, device=mining_distances.device); h_trace = [h_curr.clone()]
        for _ in range(steps):
            diff = h_curr.unsqueeze(1) - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum(dim=1)
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh; h_curr = torch.min(h_curr, h_max); h_trace.append(h_curr.clone())
        h_final = torch.stack(h_trace).gather(0, mining_distances.long().clamp(max=steps).unsqueeze(0)).squeeze(0)
        
        xc = (mining_distances - lag).view(-1, 1, 1); curr_H = h_final.view(-1, 1, 1)
        x_term = (self.xx.unsqueeze(0) - xc) / (w_arch.view(-1, 1, 1) + 1e-6)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta.view(-1, 1, 1))
        
        height_limit = torch.where((self.xx.unsqueeze(0) > xc), arch_curve, curr_H)
        height_limit = torch.where((self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0), curr_H, height_limit)
        return (self.yy.unsqueeze(0) <= height_limit).float().unsqueeze(1)

# --- Task 2: Subsidence ---
class SequentialFractureDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list; self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}

    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        filename = os.path.basename(curr_path)
        s_pos = filename.rfind("sample"); s_id = int(filename[s_pos+7 : s_pos+11])
        st_pos = filename.rfind("step"); st_id = int(filename[st_pos+5 : st_pos+8])
        mining_dist = st_id * 10.0
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(64, 64)
            y_t = y_t.T # Corrected
        
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

class ActivityArchMaskGenerator(nn.Module):
    def __init__(self):
        super(ActivityArchMaskGenerator, self).__init__()
        y_grid, x_grid = torch.meshgrid(torch.linspace(0, 150.0, 64), torch.linspace(0, 500.0, 64), indexing='ij')
        self.register_buffer('y_grid', y_grid); self.register_buffer('x_grid', x_grid)

    def forward(self, mining_distances, phys_params):
        masks = []
        for i in range(mining_distances.size(0)):
            d = mining_distances[i]
            h_max = phys_params[i, 0]; w_arch = phys_params[i, 1]
            beta = phys_params[i, 2]; lag = phys_params[i, 3]
            
            xc = d - lag; curr_H = h_max * torch.tanh(d / 100.0)
            x_term = (self.x_grid - xc) / (w_arch + 1e-6)
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * (x_term.abs() <= 1.0).float()
            masks.append((self.y_grid <= y_boundary).float())
        return torch.stack(masks).unsqueeze(1)

# --- Helper: Get Top Samples ---
def get_top_samples(task_name, dataset_dir, model_dir, params_json, StaticFeats, MaskGenClass, DatasetClass, args):
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files: return None, None
    np.random.seed(53); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    stats_path = os.path.join(model_dir, f"{task_name}_stats_seed53.pt")
    if not os.path.exists(stats_path): stats_path = os.path.join(model_dir, f"{task_name}_stats.pt")
    if not os.path.exists(stats_path): print(f"Stats missing for {task_name}"); return None, None
    stats = torch.load(stats_path)
    transform = NormalizeTransform(stats['mean'], stats['std'])
    
    dataset = DatasetClass(val_files, params_json, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    st_dim = StaticFeats; dyn_dim = total_feats - StaticFeats
    
    model = DualBranchMambaModel(st_dim, dyn_dim, 64*64, 'dual').to(DEVICE)
    model_path = os.path.join(model_dir, f"best_{task_name}_full_dual_seed53.pth")
    if not os.path.exists(model_path): model_path = os.path.join(model_dir, f"best_{task_name}_full_dual.pth") # Fallback
    
    if not os.path.exists(model_path): print(f"Model missing: {model_path}"); return None, None
    try: model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except: model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()

    max_step = 0
    all_steps = []
    for _, _, _, st_id, _, _ in loader: 
        s = st_id.item()
        if s > max_step: max_step = s
        all_steps.append(s)
    
    # 1. Try Grouping
    step_per_group = math.ceil(max_step / args.samples)
    groups = [{'range': (i*step_per_group + 1, min((i + 1)*step_per_group, max_step)), 
               'best_sample': None, 'best_mse': float('inf'),
               'fallback_sample': None, 'fallback_mse': float('inf')} for i in range(args.samples)]
    
    mask_generator = MaskGenClass().to(DEVICE)
    criterion = nn.MSELoss()
    
    # Also collect ALL filtered samples for emergency fallback
    all_valid_samples_fallback = [] 
    
    with torch.no_grad():
        for x, y, s_id, st_id, dist, phys_params in loader:
            current_step = st_id.item()
            
            x = x.to(DEVICE); y = y.to(DEVICE); dist = dist.to(DEVICE); phys_params = phys_params.to(DEVICE)
            pred = model(x); mse = criterion(pred, y.view(1, -1)).item()
            if math.isnan(mse) or math.isinf(mse): continue
            
            sample_data = {'x': x.cpu(), 'y': y.cpu(), 'mse': mse, 'info': (s_id.item(), st_id.item())}
            
            # Physics Filter
            arch_masks = mask_generator(dist, phys_params)
            gt_bin = (y.view(1, 1, 64, 64) > (0.1 if task_name == "stress" else 0.05)).float()
            gt_pcr = (gt_bin * arch_masks).sum() / (gt_bin.sum() + 1e-6)
            
            target_group = next((g for g in groups if g['range'][0] <= current_step <= g['range'][1]), None)

            # Always update fallback for group
            if target_group and mse < target_group['fallback_mse']:
                target_group['fallback_mse'] = mse
                target_group['fallback_sample'] = sample_data
            
            # --- Check Filters ---
            passed_filter = True
            
            if target_group:
                group_idx = groups.index(target_group)
                threshold = -1.0 if group_idx == 0 else (0.4 if task_name == "subsidence" and group_idx==1 else 0.5)
                if task_name == "subsidence" and group_idx > 1: threshold = 0.6
                if gt_pcr <= threshold: passed_filter = False
                if group_idx > 0 and y.sum() <= 1.0: passed_filter = False
            else:
                # Default filter if out of range? likely strict
                if gt_pcr <= 0.5: passed_filter = False
            
            if passed_filter:
                all_valid_samples_fallback.append(sample_data)
                if target_group and mse < target_group['best_mse']:
                    target_group['best_mse'] = mse
                    target_group['best_sample'] = sample_data

    # Gather Final Samples
    final_samples = []
    
    # 1. From Groups
    for g in groups:
        if g['best_sample']: final_samples.append(g['best_sample'])
        elif g['fallback_sample']: final_samples.append(g['fallback_sample'])
    
    # 2. Fill if missing
    if len(final_samples) < args.samples:
        print(f"Warning: Only found {len(final_samples)} samples via grouping for {task_name}. Filling from general pool.")
        # Sort general pool by MSE
        all_valid_samples_fallback.sort(key=lambda d: d['mse'])
        
        # Add samples not already in final_samples
        existing_infos = set([s['info'] for s in final_samples])
        for s in all_valid_samples_fallback:
            if len(final_samples) >= args.samples: break
            if s['info'] not in existing_infos:
                final_samples.append(s)
                existing_infos.add(s['info'])
    
    # 3. If STILL missing (e.g. strict filters killed too many), just duplicate last or take any
    # This shouldn't happen with fallback, but just in case
    while len(final_samples) < args.samples and len(final_samples) > 0:
        final_samples.append(final_samples[-1])
        
    return model, final_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    
    # Stress Data
    stress_model, stress_samples = get_top_samples(
        "stress", 
        os.path.join(BASE_DIR, "../../final_dataset_stress"),
        os.path.join(BASE_DIR, "../robustness_results_stress"),
        os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json"),
        17, TheoryConsistentMaskGenerator, SequentialStressDataset, args
    )
    
    # Subsidence Data
    sub_model, sub_samples = get_top_samples(
        "subsidence",
        os.path.join(BASE_DIR, "../../final_dataset"),
        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
        os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json"),
        11, ActivityArchMaskGenerator, SequentialFractureDataset, args
    )
    
    if not stress_samples or not sub_samples: print("Missing samples."); return

    # Plot Layout: Side by Side
    # Columns: 0-2 (Stress: GT, Pred, Err) | 3-5 (Sub: GT, Pred, Err)
    # Rows: args.samples
    
    n_rows = args.samples
    n_cols = 6
    fig_width = FIG_WIDTH_INCH * 1.5 # Wider figure for 6 cols
    fig_height = (fig_width / n_cols) * n_rows 
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    vmin, vmax = 0.0, 1.0
    
    for r in range(n_rows):
        # Stress (Left Block)
        # Handle if we somehow have fewer samples (though we forced fill)
        if r < len(stress_samples):
            sample = stress_samples[r]
            x = sample['x'].to(DEVICE); y_gt = sample['y'].squeeze().numpy()
            with torch.no_grad(): pred = stress_model(x).view(64, 64).cpu().numpy()
            err = np.abs(pred - y_gt)
            
            # GT
            ax = axes[r, 0]; ax.imshow(y_gt, cmap='jet', vmin=vmin, vmax=vmax, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(f"({r+1})", rotation=0, fontweight='bold', labelpad=20, va='center', fontsize=9)
            # if r == 0: ax.set_title("Stress GT", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(a) Ground Truth", fontweight='bold')

            # Pred
            ax = axes[r, 1]; ax.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title("Stress Pred", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(b) Prediction", fontweight='bold')
            
            # Err
            ax = axes[r, 2]; ax.imshow(err, cmap='jet', vmin=0, vmax=0.3, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title("Stress Error", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(c) Error", fontweight='bold')

        # Subsidence (Right Block)
        if r < len(sub_samples):
            sample = sub_samples[r]
            x = sample['x'].to(DEVICE); y_gt = sample['y'].squeeze().numpy()
            with torch.no_grad(): pred = sub_model(x).view(64, 64).cpu().numpy()
            err = np.abs(pred - y_gt)
            
            # GT
            ax = axes[r, 3]; ax.imshow(y_gt, cmap='jet', vmin=vmin, vmax=vmax, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title("Sub GT", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(a) Ground Truth", fontweight='bold')
            
            # Pred
            ax = axes[r, 4]; ax.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title("Sub Pred", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(b) Prediction", fontweight='bold')
            
            # Err
            ax = axes[r, 5]; ax.imshow(err, cmap='jet', vmin=0, vmax=0.3, origin='lower'); ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title("Sub Error", fontweight='bold', fontsize=9)
            if r == n_rows - 1: ax.set_xlabel("(c) Error", fontweight='bold')

    # Adjust layout to make room for colorbars on the right
    plt.subplots_adjust(right=0.88)

    # Colorbar 1: Normalized Value (Stress/Subsidence)
    # Position: Right side, upper half (roughly) or full length if scales are comparable.
    # Since specific ranges might differ (0-1.2 for both roughly), we can use one bar or two if needed.
    # Here we use one bar for the main values (GT, Pred)
    cbar_ax_val = fig.add_axes([0.90, 0.53, 0.015, 0.35])
    norm_val = plt.Normalize(vmin=vmin, vmax=vmax)
    sm_val = plt.cm.ScalarMappable(cmap='jet', norm=norm_val)
    sm_val.set_array([])
    cb_val = fig.colorbar(sm_val, cax=cbar_ax_val)
    cb_val.set_label("Normalized Fracture Density", fontsize=10, labelpad=10)
    cb_val.ax.tick_params(labelsize=8)

    # Colorbar 2: Absolute Error
    # Position: Right side, lower half
    cbar_ax_err = fig.add_axes([0.90, 0.12, 0.015, 0.35])
    norm_err = plt.Normalize(vmin=0, vmax=0.3)
    sm_err = plt.cm.ScalarMappable(cmap='jet', norm=norm_err)
    sm_err.set_array([])
    cb_err = fig.colorbar(sm_err, cax=cbar_ax_err)
    cb_err.set_label("Absolute Error", fontsize=10, labelpad=10)
    cb_err.ax.tick_params(labelsize=8)

    save_path = os.path.join(os.path.join(BASE_DIR, "../visualization_results_stress"), "combined_error_distribution.svg")
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
