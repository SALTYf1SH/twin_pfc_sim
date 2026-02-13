
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10 

# --- Model Shared ---
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
    def inverse(self, x): return x * (self.std + 1e-8) + self.mean

def analyze_sensitivity(task_name, dataset_dir, model_dir, static_dim, param_index, param_name, vis_dir):
    # 1. Load Model & Data
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files: return
    # Pick a specific sample deep in sequence
    np.random.seed(53) 
    # Pick 5 random files to test robustness average
    test_files = np.random.choice(all_files, 5)
    
    stats = torch.load(os.path.join(model_dir, f"{task_name}_stats_seed53.pt"))
    transformer = NormalizeTransform(stats['mean'], stats['std'])
    
    with np.load(test_files[0]) as f: 
        raw_dim = f['x'].shape[0]
        dyn_dim = raw_dim - static_dim
    
    model = DualBranchMambaModel(static_dim, dyn_dim, 64*64, 'dual').to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"best_{task_name}_full_dual_seed53.pth"), map_location=DEVICE))
    model.eval()
    
    # 2. Perturbation Loop
    multipliers = np.linspace(0.5, 1.5, 21) # 0.5x to 1.5x
    
    results = {m: [] for m in multipliers}
    
    print(f"Analyzing {task_name}, Param: {param_name} (Index {param_index})...")
    
    for filename in test_files:
        with np.load(filename) as f:
            x_raw = torch.from_numpy(f['x'].astype(np.float32))
            
        # We need to perturb raw X, then normalize
        base_val = x_raw[param_index].item()
        
        for m in multipliers:
            x_mod = x_raw.clone()
            x_mod[param_index] = base_val * m
            
            # Normalize
            x_norm = transformer(x_mod).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = model(x_norm).view(64, 64)
                # Metric: Max Magnitude (Absolute Value)
                # This handles both Stress (Concentration > 0) and Subsidence (Depth vs Elevation) consistently
                val = pred.abs().max().item()
                results[m].append(val)

    # 3. Aggregate
    avg_vals = [np.mean(results[m]) for m in multipliers]
    std_vals = [np.std(results[m]) for m in multipliers]
    
    # Normalize result to start at 1.0 for comparison
    ref_idx = 10 # 1.0x
    ref_val = avg_vals[ref_idx]
    norm_vals = [v / ref_val for v in avg_vals]
    
    # Calculate Linear Regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(multipliers, avg_vals)
    r_squared = r_value**2
    
    # Check Monotonicity
    is_monotonic = np.all(np.diff(avg_vals) >= -1e-4) or np.all(np.diff(avg_vals) <= 1e-4)
    print(f"Monotonic: {is_monotonic}, Slope: {slope:.4f}, R2: {r_squared:.4f}")
    
    # Return results for plotting
    return {
        "multipliers": multipliers,
        "norm_vals": norm_vals,
        "std_vals": [s/ref_val for s in std_vals], # Normalize std dev
        "slope": slope,
        "r2": r_squared,
        "param_name": param_name,
        "intercept": intercept,
        "ref_val": ref_val
    }

# --- SCI Plotting Style ---
def set_sci_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10 
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['svg.fonttype'] = 'none' # Keep text editable
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Color Palette (Scientific)
    # Deep Blue for data, Red for fit
    return '#1f77b4', '#d62728' 

def plot_2x2_grid(task_name, results_list, vis_dir):
    """
    Plots a 2x2 grid of sensitivity analysis.
    results_list: list of 4 result dicts [Sand Mod, Sand Coh, Mud Mod, Mud Coh]
    """
    from scipy.stats import linregress
    
    data_color, fit_color = set_sci_style()
    
    # Width 17cm = 6.7 inches. Height can be ~5.5 inches for 2x2.
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.5))
    # fig.suptitle(f"{task_name.capitalize()} Sensitivity Analysis", fontsize=10, y=0.98)
    
    axes = axes.flatten()
    
    for i, res in enumerate(results_list):
        ax = axes[i]
        multipliers = res['multipliers']
        norm_vals = res['norm_vals']
        std_vals = res['std_vals']
        slope = res['slope']
        r2 = res['r2']
        name = res['param_name']
        
        # Plot Data
        ax.plot(multipliers, norm_vals, 'o-', linewidth=1.5, markersize=4, 
                color=data_color, label=f'Simulated')
        
        # Plot Regression
        slope_n, intercept_n, r_val_n, _, _ = linregress(multipliers, norm_vals)
        r2_n = r_val_n**2
        reg_line = slope_n * multipliers + intercept_n
        
        ax.plot(multipliers, reg_line, '--', linewidth=1.0, 
                color=fit_color, label=f'Fit ($R^2$={r2_n:.2f})')
        
        # Uncertainty
        ax.fill_between(multipliers, 
                        [n - s for n,s in zip(norm_vals, std_vals)], 
                        [n + s for n,s in zip(norm_vals, std_vals)], 
                        color=data_color, alpha=0.15, edgecolor='none')
        
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_title(name, fontsize=9, pad=4)
        
        # Only bottom row gets X labels
        if i >= 2:
            ax.set_xlabel("Multiplier", fontsize=9)
        
        # Only left col gets Y labels
        if i % 2 == 0:
            ax.set_ylabel("Norm. Response", fontsize=9)
            
        ax.legend(fontsize=8, frameon=False, loc='best')
        ax.tick_params(direction='in')

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0) # Minimize whitespace
    
    # Save PNG
    save_path_png = os.path.join(vis_dir, f"sensitivity_combined_{task_name}.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    
    # Save SVG
    save_path_svg = os.path.join(vis_dir, f"sensitivity_combined_{task_name}.svg")
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    
    plt.close()
    print(f"Saved Combined Plots: {save_path_png} & .svg")

def main():
    vis_dir = os.path.join(BASE_DIR, "../visualization_results_stress")
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    
    # --- Stress Analysis ---
    print("\n--- STRESS SENSITIVITY ---")
    stress_results = []
    
    # 1. Sandstone Modulus
    stress_results.append(analyze_sensitivity("stress", 
                        os.path.join(BASE_DIR, "../../final_dataset_stress"),
                        os.path.join(BASE_DIR, "../robustness_results_stress"),
                        17, 0, "Sandstone Modulus", vis_dir))
    # 2. Sandstone Cohesion
    stress_results.append(analyze_sensitivity("stress", 
                        os.path.join(BASE_DIR, "../../final_dataset_stress"),
                        os.path.join(BASE_DIR, "../robustness_results_stress"),
                        17, 2, "Sandstone Cohesion", vis_dir))
    # 3. Mudstone Modulus
    stress_results.append(analyze_sensitivity("stress", 
                        os.path.join(BASE_DIR, "../../final_dataset_stress"),
                        os.path.join(BASE_DIR, "../robustness_results_stress"),
                        17, 4, "Mudstone Modulus", vis_dir))
    # 4. Mudstone Cohesion
    stress_results.append(analyze_sensitivity("stress", 
                        os.path.join(BASE_DIR, "../../final_dataset_stress"),
                        os.path.join(BASE_DIR, "../robustness_results_stress"),
                        17, 6, "Mudstone Cohesion", vis_dir))
                        
    plot_2x2_grid("stress", stress_results, vis_dir)

    # --- Subsidence Analysis ---
    print("\n--- SUBSIDENCE SENSITIVITY ---")
    sub_results = []
    
    # 1. Sandstone Modulus
    sub_results.append(analyze_sensitivity("subsidence",
                        os.path.join(BASE_DIR, "../../final_dataset"),
                        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
                        11, 0, "Sandstone Modulus", vis_dir))
    # 2. Sandstone Cohesion
    sub_results.append(analyze_sensitivity("subsidence",
                        os.path.join(BASE_DIR, "../../final_dataset"),
                        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
                        11, 2, "Sandstone Cohesion", vis_dir))
    # 3. Mudstone Modulus
    sub_results.append(analyze_sensitivity("subsidence",
                        os.path.join(BASE_DIR, "../../final_dataset"),
                        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
                        11, 4, "Mudstone Modulus", vis_dir))
    # 4. Mudstone Cohesion
    sub_results.append(analyze_sensitivity("subsidence",
                        os.path.join(BASE_DIR, "../../final_dataset"),
                        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
                        11, 6, "Mudstone Cohesion", vis_dir))
                        
    plot_2x2_grid("subsidence", sub_results, vis_dir)

if __name__ == "__main__":
    main()
