import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, skew, kurtosis
import math
from tqdm import tqdm

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed.")
    Mamba = None

# --- Configuration ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['svg.fonttype'] = 'none'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (Copied from visualize scripts) ---
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

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.static_len = static_len; self.dynamic_len = dynamic_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class BiLSTMBaselineSubsidence(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaselineSubsidence, self).__init__()
        self.static_len = static_len; self.dynamic_len = dynamic_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size)) # No dropout
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SimpleDataset(Dataset):
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list; self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        try:
            with np.load(self.file_list[idx]) as data:
                x_t = torch.from_numpy(data['x'].astype(np.float32))
                y_t = torch.from_numpy(data['y'].astype(np.float32))
                if y_t.ndim == 1: y_t = y_t.reshape(64, 64)
                y_t = y_t.T # Physical alignment
            if self.transform: x_t = self.transform(x_t)
            return x_t, y_t
        except:
            return torch.zeros(1), torch.zeros(1) # Fail safe

def analyze_task(task_name, dataset_dir, model_dir, baseline_dir, static_feats, output_dir):
    print(f"\nAnalyzing Error Distribution for Task: {task_name.upper()}")
    
    # 1. Data Prep
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files: print("No data."); return

    np.random.seed(53)
    np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    # Load Stats
    stats_path = os.path.join(model_dir, f"{task_name}_stats_seed53.pt")
    if not os.path.exists(stats_path):
        # Fallback
        if task_name == 'stress': stats_path = os.path.join(model_dir, "stress_stats_seed53.pt")
        else: stats_path = os.path.join(model_dir, "subsidence_stats_seed53.pt")

    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        transform = NormalizeTransform(stats['mean'], stats['std'])
    else:
        print("Stats not found, skipping."); return

    dataset = SimpleDataset(val_files, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False) # Batch for speed
    
    # 2. Models
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    dyn_feats = total_feats - static_feats
    output_feats = 64*64
    
    # Mamba
    mamba_model = DualBranchMambaModel(static_feats, dyn_feats, output_feats).to(DEVICE)
    mamba_path = os.path.join(model_dir, f"best_{task_name}_full_dual_seed53.pth")
    mamba_model.load_state_dict(torch.load(mamba_path, map_location=DEVICE), strict=False)
    mamba_model.eval()
    
    # 3. Compute Errors
    mamba_errors = []
    
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Computing Errors"):
            if x.dim() < 2: continue # skip fail safe
            x = x.to(DEVICE); y = y.to(DEVICE)
            y_flat = y.reshape(x.size(0), -1)
            
            # Mamba
            pred_m = mamba_model(x)
            mse_m = criterion(pred_m, y_flat).mean(dim=1).cpu().numpy()
            mamba_errors.extend(mse_m)
            
    mamba_errors = np.array(mamba_errors)
    
    # 4. Log Transform
    # Add epsilon to avoid log(0) if perfectly 0 (unlikely for MSE)
    log_mamba = np.log10(mamba_errors + 1e-12)
    
    # 5. Plotting
    # Width 7cm = 2.76 inches. 
    fig, ax = plt.subplots(figsize=(2.76, 2.4))
    
    # Params for Fit
    mu_m, std_m = norm.fit(log_mamba)
    
    # Normality Test
    _, p_m = shapiro(log_mamba[:5000]) if len(log_mamba) > 3 else (0, 0)
    
    print(f"Mamba: Mean={mu_m:.4f}, Std={std_m:.4f}, Skew={skew(log_mamba):.4f}, SW-p={p_m:.4e}")
    
    # Histogram range
    xmin = log_mamba.min()
    xmax = log_mamba.max()
    x = np.linspace(xmin, xmax, 100)
    
    # Plot Mamba
    # Use density=True for comparison with PDF
    ax.hist(log_mamba, bins=40, density=True, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5, label='Distribution')
    p_m_curve = norm.pdf(x, mu_m, std_m)
    ax.plot(x, p_m_curve, color='#d62728', linestyle='-', linewidth=1.5, label='Normal Fit')
    
    # Styling for 7cm width
    ax.set_xlabel('Log$_{10}$(MSE)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add simplified stats to legend or title? Legend is better for space.
    # ax.legend(fontsize=8, frameon=False, loc='upper right')
    # Or just no legend if obvious? 
    # Let's keep legend but minimal
    ax.legend(fontsize=8, frameon=False, loc='best', handlelength=1.5)
    
    ax.grid(linestyle=':', alpha=0.5)
    # Title might take too much space, use compact title or skip if caption handles it
    ax.set_title(f"{task_name.capitalize()} Error", fontsize=9, pad=3)
    
    plt.tight_layout(pad=0.2)
    save_path = os.path.join(output_dir, f"error_dist_{task_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace(".png", ".svg"), format='svg')
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    vis_dir = os.path.join(BASE_DIR, "../visualization_results_stress")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Stress
    analyze_task(
        "stress",
        os.path.join(BASE_DIR, "../../final_dataset_stress"),
        os.path.join(BASE_DIR, "../robustness_results_stress"),
        os.path.join(BASE_DIR, "../robustness_results_stress_baselines"),
        17,
        vis_dir
    )
    
    # Subsidence
    analyze_task(
        "subsidence",
        os.path.join(BASE_DIR, "../../final_dataset"),
        os.path.join(BASE_DIR, "../robustness_results_subsidence"),
        os.path.join(BASE_DIR, "../robustness_results_subsidence_baselines"),
        11,
        vis_dir
    )

if __name__ == "__main__":
    main()
