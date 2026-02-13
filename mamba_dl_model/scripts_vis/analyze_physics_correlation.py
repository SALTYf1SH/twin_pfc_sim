
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.stats

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & Dataset ---
class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual', d_model=128, n_layers=2, dropout=0.1): 
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
            self.mamba_layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        self.fusion_head = nn.Sequential(
            nn.Linear(32+d_model, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )
    
    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        x_mamba = self.dynamic_embedder(x_dynamic.unsqueeze(-1))
        for layer, norm in zip(self.mamba_layers, self.norms): x_mamba = layer(norm(x_mamba)) + x_mamba
        dynamic_out = x_mamba.mean(dim=1)
        return self.fusion_head(torch.cat((static_out, dynamic_out), dim=1))

class NormalizeTransform:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SimpleDataset(Dataset):
    def __init__(self, file_list, static_dim, transform=None):
        self.file_list = file_list; self.static_dim = static_dim; self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        with np.load(self.file_list[idx]) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(64, 64)
            y_t = y_t.T
            if self.transform: x_t = self.transform(x_t)
            return x_t, y_t

def compute_saliency(model, x):
    model.eval(); x.requires_grad_()
    output = model(x)
    model.zero_grad()
    output.sum().backward()
    return x.grad.abs().cpu().numpy().flatten()

def resample_1d(arr, target_len):
    # Linear interpolation to resize array
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)

def analyze_correlation(task_name, dataset_dir, model_dir, static_dim, output_plot_dir):
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files: return
    # Use same seed for consistency usually, but here we scan many
    samples_to_check = 50
    np.random.seed(53); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    stats = torch.load(os.path.join(model_dir, f"{task_name}_stats_seed53.pt"))
    dataset = SimpleDataset(val_files, static_dim, transform=NormalizeTransform(stats['mean'], stats['std']))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with np.load(val_files[0]) as f: dyn_dim = f['x'].shape[0] - static_dim
    model = DualBranchMambaModel(static_dim, dyn_dim, 64*64, 'dual').to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"best_{task_name}_full_dual_seed53.pth"), map_location=DEVICE))
    
    correlations = []
    
    os.makedirs(output_plot_dir, exist_ok=True)
    
    count = 0
    fig, axes = plt.subplots(4, 5, figsize=(15, 10)) # Plot first 20 comparisons
    axes = axes.flatten()
    
    for x, y in loader:
        if count >= samples_to_check: break
        x = x.to(DEVICE)
        
        # 1. Attention (Saliency)
        saliency = compute_saliency(model, x)
        attn_curve_raw = saliency[static_dim:]
        # Resample to 64 to match spatial width
        attn_curve_64 = resample_1d(attn_curve_raw, 64)
        attn_norm = (attn_curve_64 - attn_curve_64.min()) / (attn_curve_64.max() - attn_curve_64.min() + 1e-8)
        
        # 2. Physics (Fracture Density / Stress accumulation)
        # Sum Ground Truth column-wise to get horizontal distribution
        gt_img = y.squeeze().cpu().numpy() # 64x64
        # Vertical sum -> Horizontal Profile
        phys_profile = np.sum(gt_img, axis=0) # Shape (64,)
        phys_norm = (phys_profile - phys_profile.min()) / (phys_profile.max() - phys_profile.min() + 1e-8)
        
        # Check ranges
        if (attn_norm.max() - attn_norm.min()) < 1e-6:
            # print("Constant Attention")
            continue
        if (phys_norm.max() - phys_norm.min()) < 1e-6:
            # print("Constant Physics")
            continue
            
        # 3. Correlation
        corr, _ = scipy.stats.pearsonr(attn_norm, phys_norm)
        if not np.isnan(corr):
            correlations.append(corr)
        
        # Plot first 20
        if count < 20:
            ax = axes[count]
            ax.plot(attn_norm, 'r-', label='Attention')
            ax.plot(phys_norm, 'b--', label='Fracture Dist')
            ax.set_title(f"R={corr:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if count == 0: ax.legend(fontsize=6)
        
        count += 1
    
    avg_corr = np.mean(correlations)
    print(f"Task: {task_name} | Avg Spatial Correlations: {avg_corr:.4f} (over {len(correlations)} samples)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{task_name}_attention_vs_physics.png"))
    plt.close()
    
    return avg_corr

def main():
    vis_dir = os.path.join(BASE_DIR, "../visualization_results_stress")
    
    s_corr = analyze_correlation("stress", os.path.join(BASE_DIR, "../../final_dataset_stress"), os.path.join(BASE_DIR, "../robustness_results_stress"), 17, vis_dir)
    sub_corr = analyze_correlation("subsidence", os.path.join(BASE_DIR, "../../final_dataset"), os.path.join(BASE_DIR, "../robustness_results_subsidence"), 11, vis_dir)
    
    print("Interpretation:")
    print("High correlation (>0.5) implies Mamba attention spatially aligns with fracture/subsidence concentration.")

if __name__ == "__main__":
    main()
