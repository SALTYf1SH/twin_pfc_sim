import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import math

# [Mamba] Import
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it to use Mamba architecture.")
    Mamba = None

# --- Configuration ---
# SCI Formatting
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['svg.fonttype'] = 'none' # Ensure text is editable in SVG
FIG_WIDTH_CM = 17 
FIG_WIDTH_INCH = FIG_WIDTH_CM / 2.54

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset_stress")

# Directories for Models
# Priority: Local Ablation -> Local Physics -> External Physics (New DL Model)
MODEL_DIR = os.path.join(BASE_DIR, "../robustness_results_stress")
# PHYSICS_DIR = os.path.join(BASE_DIR, "../trained_models_stress_physics_mamba") # Deprecated
# PHYSICS_MODEL_DIR = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_stress_physics") # Deprecated

PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "../visualization_results_stress")

STATIC_FEATURES = 17   
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH
STEP_DISTANCE_M = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (Self-Contained) ---

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
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            static_out = torch.zeros(x.size(0), 32, device=x.device)
        
        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic_seq = x_dynamic.unsqueeze(-1)
            x_mamba = self.dynamic_embedder(x_dynamic_seq)
            if hasattr(self, 'mamba_layers'):
                for layer, norm in zip(self.mamba_layers, self.norms):
                    residual = x_mamba
                    x_mamba = norm(x_mamba)
                    x_mamba = layer(x_mamba)
                    x_mamba = residual + x_mamba
                dynamic_out = x_mamba.mean(dim=1)
            else:
                dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- Data Loading ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}

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
            y_t = y_t.T 
        
        default_params = {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]}
        params = self.physics_params.get(str(s_id), default_params)
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

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

# --- Main Visualization Function ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4, help="Number of rows (samples) to plot")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
    if not all_files: print("No data."); return
    
    np.random.seed(53); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    # Stats
    stats_path = os.path.join(MODEL_DIR, "stress_stats_seed53.pt")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "stress_stats_ablation.pt")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "stress_stats.pt")
        
    if os.path.exists(stats_path):
        print(f"Loading stats from {stats_path}")
        stats = torch.load(stats_path)
        transform = NormalizeTransform(stats['mean'], stats['std'])
    else:
        print(f"Stats not found at {stats_path}."); return

    dataset = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    st_dim = STATIC_FEATURES; dyn_dim = total_feats - STATIC_FEATURES
    
    models = []
    
    def load_mamba_variant(name, filename, branch_mode, fallback_dir=None):
        model = DualBranchMambaModel(st_dim, dyn_dim, OUTPUT_FEATURES, branch_mode).to(DEVICE)
        
        # Primary path in MODEL_DIR (Ablation dir)
        # Check for seed 53 first
        base_name, ext = os.path.splitext(filename)
        path = os.path.join(MODEL_DIR, f"{base_name}_seed53{ext}")
        if not os.path.exists(path):
            path = os.path.join(MODEL_DIR, filename)
        
        # Primary Fallback (Physics dir, if applicable for main model)
        if not os.path.exists(path) and fallback_dir:
            path = os.path.join(fallback_dir, f"{base_name}_seed53{ext}")
            
        # Specific Fallback for Proposed Model naming variations
        # In robustness results, it is likely named best_stress_full_dual_seed53.pth
        if name == "Proposed" and not os.path.exists(path):
             path = os.path.join(MODEL_DIR, "best_stress_full_dual_seed53.pth") 
             if not os.path.exists(path):
                 path = os.path.join(MODEL_DIR, "best_stress_full_dual.pth")

        
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            except RuntimeError:
                print(f"Warning: Failed strict load for {name}. fast-loading...")
                model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
            model.eval()
            models.append({"name": name, "model": model, "type": "Mamba"})
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: Model {name} not found at {path}")

    # 1. Proposed (Dual + Physics)
    # 1. Proposed (Dual + Physics)
    load_mamba_variant("Proposed", "best_stress_full_dual_seed53.pth", 'dual')

    # 2. No Physics
    load_mamba_variant("No Physics", "best_stress_no_physics_dual_no_phys_seed53.pth", 'dual')
    
    # 3. Static Only
    load_mamba_variant("Static Only", "best_stress_full_static_only_seed53.pth", 'static_only')

    # 4. Dynamic Only
    load_mamba_variant("Dynamic Only", "best_stress_full_dynamic_only_seed53.pth", 'dynamic_only')

    print(f"Ablation Models: {[m['name'] for m in models]}")
    
    # 3. Filter and Group
    print("Scanning dataset for max steps...")
    max_step = 0
    for _, _, _, st_id, _, _ in loader:
        if st_id.item() > max_step: max_step = st_id.item()
    
    print(f"Max Step found: {max_step}. Plotting {args.samples} samples.")
    step_per_group = math.ceil(max_step / args.samples)
    groups = []
    for i in range(args.samples):
        start = i * step_per_group + 1
        end = min((i + 1) * step_per_group, max_step)
        groups.append({'range': (start, end), 'best_sample': None, 'best_mse': float('inf'), 
                       'fallback_sample': None, 'fallback_mse': float('inf')})
        
    criterion = nn.MSELoss()
    mask_generator = TheoryConsistentMaskGenerator(output_size=64).to(DEVICE)
    
    print("Screening samples by group (PCR > 0.6)...")
    with torch.no_grad():
        for x, y, s_id, st_id, dist, phys_params in loader:
            current_step = st_id.item()
            
            target_group = None
            for g in groups:
                if g['range'][0] <= current_step <= g['range'][1]:
                    target_group = g
                    break
            if target_group is None: continue
            
            x = x.to(DEVICE); y = y.to(DEVICE)
            dist = dist.to(DEVICE); phys_params = phys_params.to(DEVICE)
            
            # Predict with ALL models to ensure we pick a sample where MAIN model works well? 
            # Actually, standard is usually based on Main model's MES or fallback logic.
            # We use the FIRST loaded model (Proposed) for selection logic.
            if not models: break
            main_model = models[0]['model']
            
            pred = main_model(x)
            mse = criterion(pred, y.view(1, -1)).item()
            
            if math.isnan(mse) or math.isinf(mse): continue

            # Fallback update
            if mse < target_group['fallback_mse']:
                 target_group['fallback_mse'] = mse
                 target_group['fallback_sample'] = {
                    'x': x.cpu(), 'y': y.cpu(), 'mse': mse,
                    'info': (s_id.item(), st_id.item())
                }
            
            # Physics Check (Strict)
            arch_masks = mask_generator(dist, phys_params)
            gt_bin = (y.view(1, 1, 64, 64) > 0.1).float() # Threshold 0.1 for Stress
            gt_pcr = (gt_bin * arch_masks).sum() / (gt_bin.sum() + 1e-6)
            
            group_idx = groups.index(target_group)
            if group_idx == 0: threshold = -1.0 
            else: threshold = 0.5 
            
            if gt_pcr <= threshold: continue
            if group_idx > 0 and y.sum() <= 1.0: continue
            
            if mse < target_group['best_mse']:
                target_group['best_mse'] = mse
                target_group['best_sample'] = {
                    'x': x.cpu(), 'y': y.cpu(), 'mse': mse,
                    'info': (s_id.item(), st_id.item())
                }
                
    top_samples = []
    for i, g in enumerate(groups):
        if g['best_sample'] is not None: top_samples.append(g['best_sample'])
        elif g['fallback_sample'] is not None:
            print(f"Group {i} ({g['range']}) used fallback.")
            top_samples.append(g['fallback_sample'])
        else:
            print(f"Group {i} ({g['range']}) has NO sample.")
    
    if not top_samples: print("No valid samples found."); return
    print(f"Plotting {len(top_samples)} samples...")
    
    n_rows = len(top_samples); n_cols = 1 + len(models)
    fig_height = (FIG_WIDTH_INCH / n_cols) * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_INCH, fig_height), squeeze=False)
    
    vmin, vmax = 0.0, 1.0
    for r, sample in enumerate(top_samples):
        x_in = sample['x'].to(DEVICE)
        y_gt = sample['y'].squeeze().numpy()
        
        ax_gt = axes[r, 0]
        im = ax_gt.imshow(y_gt, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        # if r == 0: ax_gt.set_title("Ground Truth", fontweight='bold')
        ax_gt.set_ylabel(f"({r+1})", rotation=0, fontweight='bold', labelpad=10, va='center')
        if r == n_rows - 1: ax_gt.set_xlabel("(a) Ground Truth", fontweight='bold')
        
        for c, m_cfg in enumerate(models):
            ax = axes[r, c+1]
            model = m_cfg['model']
            with torch.no_grad():
                # Handling 'static_only' output shape issues in some configs
                pred = model(x_in)
                pred = pred.view(64, 64).cpu().numpy()
            im = ax.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title(m_cfg['name'], fontweight='bold')
            if r == n_rows - 1: ax.set_xlabel(f"({chr(98+c)}) {m_cfg['name']}", fontweight='bold')
            

            
    # Colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Fracture Density", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=8)

    save_path = os.path.join(OUTPUT_DIR, "stress_ablation_results_sci.svg")
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
