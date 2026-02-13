
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
            y_t = y_t.T 
        
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 3.0, "lag": 20.0, "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]})
        p_list = [params['h_max'], params['width'], params['beta'], params['lag'], params['k_growth']]
        p_list.extend(params['ks_heights']); p_list.extend(params['ks_betas'])
        phys_vec = torch.tensor(p_list, dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

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
            y_t = y_t.T 
        
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

# --- Saliency Computation ---
def compute_saliency(model, x):
    """
    Computes Saliency Map: Gradient of Output Sum w.r.t Input.
    Highlights which input features maximize the output activation.
    For regression, this roughly corresponds to "sensitivity".
    """
    model.eval()
    x.requires_grad_()
    
    # Forward
    output = model(x)
    
    # Backward: Gradient of sum of outputs (sensitivity to input)
    # Alternatively, could be gradient of deviation from mean, but sum is standard for activation mapping
    model.zero_grad()
    output.sum().backward()
    
    # Get Gradient: (B, Features)
    saliency = x.grad.abs()
    
    # Reshape if possible? 
    # Mamba inputs are flat or sequence. Here inputs are (B, Features).
    # We visualizing the DYNAMIC part which corresponds to the sequence/spatial domain?
    # Our input x has static + dynamic features.
    
    return saliency, output

# --- Helper: Get Top Samples ---
def get_top_samples(task_name, dataset_dir, model_dir, params_json, StaticFeats, DatasetClass, args):
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
    if not os.path.exists(model_path): model_path = os.path.join(model_dir, f"best_{task_name}_full_dual.pth")
    
    if not os.path.exists(model_path): print(f"Model missing: {model_path}"); return None, None
    try: model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except: model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()

    # Just get 'samples' random samples that are somewhat deep in the sequence (valid steps)
    final_samples = []
    
    # Use a simple stride to get diverse samples
    target_count = args.samples
    count = 0
    
    # We want samples that have some "structure" (not step 0), so skip early ones
    skip_steps = 10
    
    for x, y, s_id, st_id, dist, phys_vec in loader:
        if st_id.item() < skip_steps: continue
        
        # Check if we should keep this sample (simple deterministic selection)
        if count < target_count:
            # Run Saliency
            x = x.to(DEVICE); y = y.to(DEVICE)
            saliency, pred = compute_saliency(model, x)
            
            # Post-process Saliency
            # x is (1, Static+Dynamic). Dynamic part is what maps to spatial?
            # Actually, `x` here IS the raw input features.
            # In 'twin_pfc_sim', the input x is likely flattened elements or similar.
            # However, we don't have a 2048 -> 64x64 mapping for INPUT easily unless input is also an image.
            # Let's check: x shape.
            # If x is 1D feature vector, we can just plot it as a bar or line?
            # Or if it comes from a grid, reshape it.
            # Based on DualBranchMambaModel: x_dynamic = x[:, static_size:]
            # The dynamic part is fed to Mamba.
            
            # Let's just visualize the reshape of input if possible, or just the output saliency?
            # Wait, Saliency = gradient w.r.t INPUT. 
            # If input is not spatial (it's a list of features), visualizing it as an image might be wrong unless reshaping makes sense.
            # But the OUTPUT is an image (64x64). 
            # We want to see which PARTS of the input affect the output.
            
            # Let's assume input has some spatial structure or we just visualize the raw gradient vector as a "signal".
            # Better yet: Color code the input vector?
            
            # FOR VISUAL APPEAL: Let's reshape the Dynamic part of input Saliency to square if it's square number?
            # dyn_dim = total - static.
            
            dyn_grad = saliency[0, st_dim:].cpu().numpy()
            
            # Try to see if it makes a square
            side = int(np.sqrt(dyn_grad.shape[0]))
            if side * side == dyn_grad.shape[0]:
                dyn_grad_img = dyn_grad.reshape(side, side)
            else:
                # If not square, just pad or line plot? 
                # Let's just create a 1D strip image
                dyn_grad_img = np.expand_dims(dyn_grad, 0)
            
            final_samples.append({
                'x': x.cpu().detach(), 'y': y.cpu().detach(), 'pred': pred.cpu().detach(),
                'saliency': dyn_grad_img, 'info': (s_id.item(), st_id.item())
            })
            count += 1
        else:
            break

    return model, final_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    # Stress Data
    stress_model, stress_samples = get_top_samples(
        "stress", 
        os.path.join(BASE_DIR, "../../final_dataset_stress"),
        os.path.join(BASE_DIR, "../robustness_results_stress"),
        os.path.join(BASE_DIR, "../stress_para/stress_physics_params.json"),
        17, SequentialStressDataset, args
    )
    
    # Subsidence Data
    sub_model, sub_samples = get_top_samples(
        "subsidence",
        os.path.join(BASE_DIR, "../../final_dataset"),
        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
        os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json"),
        11, SequentialFractureDataset, args
    )
    
    if not stress_samples or not sub_samples: print("Missing samples."); return

    # Plot Layout: 
    # Rows: args.samples
    # Cols: 4 columns per task? Or just combine tasks vertically?
    # User said "add script", didn't specify strict Combined layout for this one.
    # But usually separate or combined. Let's do Combined for consistency/comparison.
    # Col 1: Stress Pred
    # Col 2: Stress Saliency (Input Gradient)
    # Col 3: Sub Pred
    # Col 4: Sub Saliency (Input Gradient)
    
    # Output Directory for Individual Plots
    indiv_dir = os.path.join(BASE_DIR, "../visualization_results_stress/saliency_individual")
    os.makedirs(indiv_dir, exist_ok=True)
    
    # --- Plotting Individual Samples ---
    
    # 1. Stress
    for i, s in enumerate(stress_samples):
        fig, ax = plt.subplots(figsize=(4, 4))
        
        pred = s['pred'].view(64, 64).numpy()
        sal_grad = s['saliency'].flatten()
        sal_norm = (sal_grad - sal_grad.min()) / (sal_grad.max() - sal_grad.min() + 1e-8)
        
        # Image
        ax.imshow(pred, cmap='jet', origin='lower', extent=[0, 64, 0, 64], alpha=0.9)
        
        # Curve
        x_vals = np.linspace(0, 64, len(sal_norm))
        y_vals = sal_norm * 30.0 + 10.0
        
        ax.plot(x_vals, y_vals, color='white', linewidth=2.0, alpha=0.9, label='Saliency')
        ax.fill_between(x_vals, y_vals, 0, color='white', alpha=0.1)
        
        ax.set_xticks([]); 
        ax.tick_params(axis='y', colors='black', labelsize=12)
        
        s_id, st_id = s['info']
        ax.set_title(f"Stress Sample {s_id} Step {st_id}", fontweight='bold', fontsize=10)
        
        save_name = f"stress_s{s_id}_st{st_id}.svg"
        plt.tight_layout()
        plt.savefig(os.path.join(indiv_dir, save_name), format='svg', dpi=300)
        plt.close(fig)
        
    print(f"Saved {len(stress_samples)} individual Stress plots to {indiv_dir}")

    # 2. Subsidence
    for i, s in enumerate(sub_samples):
        fig, ax = plt.subplots(figsize=(4, 4))
        
        pred = s['pred'].view(64, 64).numpy()
        sal_grad = s['saliency'].flatten()
        sal_norm = (sal_grad - sal_grad.min()) / (sal_grad.max() - sal_grad.min() + 1e-8)
        
        ax.imshow(pred, cmap='jet', origin='lower', extent=[0, 64, 0, 64], alpha=0.9)
        
        x_vals = np.linspace(0, 64, len(sal_norm))
        y_vals = sal_norm * 30.0 + 10.0
        
        ax.plot(x_vals, y_vals, color='white', linewidth=2.0, alpha=0.9)
        ax.fill_between(x_vals, y_vals, 0, color='white', alpha=0.1)
        
        ax.set_xticks([]); 
        ax.tick_params(axis='y', colors='black', labelsize=12)
        
        s_id, st_id = s['info']
        ax.set_title(f"Subsidence Sample {s_id} Step {st_id}", fontweight='bold', fontsize=10)
        
        save_name = f"subsidence_s{s_id}_st{st_id}.svg"
        plt.tight_layout()
        plt.savefig(os.path.join(indiv_dir, save_name), format='svg', dpi=300)
        plt.close(fig)

    print(f"Saved {len(sub_samples)} individual Subsidence plots to {indiv_dir}")

if __name__ == "__main__":
    main()
