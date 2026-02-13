
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reusing Model & Dataset Definitions (Simplified) ---
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
        
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
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
        self.file_list = file_list; self.transform = transform; self.static_dim = static_dim
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        with np.load(self.file_list[idx]) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            if self.transform: x_t = self.transform(x_t)
            return x_t

def compute_saliency(model, x):
    model.eval(); x.requires_grad_()
    output = model(x)
    model.zero_grad()
    output.sum().backward()
    return x.grad.abs().cpu().numpy().flatten()

def analyze_task(task_name, dataset_dir, model_dir, static_dim, count=50):
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not all_files: return 0, 0
    np.random.seed(53); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    stats = torch.load(os.path.join(model_dir, f"{task_name}_stats_seed53.pt"))
    dataset = SimpleDataset(val_files, static_dim, transform=NormalizeTransform(stats['mean'], stats['std']))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with np.load(val_files[0]) as f: 
        raw_shape = f['x'].shape
        dyn_dim = raw_shape[0] - static_dim
        print(f"Task: {task_name}, Raw Info: shape={raw_shape}, dyn_dim={dyn_dim}")
    model = DualBranchMambaModel(static_dim, dyn_dim, 64*64, 'dual').to(DEVICE)
    model_path = os.path.join(model_dir, f"best_{task_name}_full_dual_seed53.pth")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    active_ratios = []; total_variations = []
    
    processed = 0
    for x in loader:
        if processed >= count: break
        x = x.to(DEVICE)
        saliency = compute_saliency(model, x)
        
        # Focus on Dynamic Features (the curve)
        curve = saliency[static_dim:]
        # Normalize
        curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)
        
        # Metric 1: Active Ratio (Sparsity Proxy) - Fraction of points > 0.2
        active_ratios.append(np.mean(curve > 0.2))
        
        # Metric 2: Smoothness (Total Variation) - lower means smoother
        tv = np.sum(np.abs(np.diff(curve)))
        total_variations.append(tv)
        
        processed += 1
        
    return np.mean(active_ratios), np.mean(total_variations)

def main():
    s_act, s_tv = analyze_task("stress", os.path.join(BASE_DIR, "../../final_dataset_stress"), os.path.join(BASE_DIR, "../robustness_results_stress"), 17)
    sub_act, sub_tv = analyze_task("subsidence", os.path.join(BASE_DIR, "../../final_dataset"), os.path.join(BASE_DIR, "../robustness_results_subsidence"), 11)
    
    print("-" * 30)
    print(f"Task        | Active Ratio (Density) | Total Variation (Roughness)")
    print("-" * 30)
    print(f"Stress      | {s_act:.4f}                 | {s_tv:.4f}")
    print(f"Subsidence  | {sub_act:.4f}                 | {sub_tv:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
