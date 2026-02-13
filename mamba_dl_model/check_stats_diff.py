
import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

# Define paths
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "..", "final_dataset_stress") # Correct path for robustness script
BASELINE_STATS_PATH = os.path.join(BASE_DIR, "trained_models_baselines_stress", "baseline_stress_stats.pt")

print(f"Checking dataset at: {DATASET_DIR}")
print(f"Checking baseline stats at: {BASELINE_STATS_PATH}")

def load_baseline_stats():
    if os.path.exists(BASELINE_STATS_PATH):
        stats = torch.load(BASELINE_STATS_PATH)
        return stats['mean'], stats['std']
    return None, None

def calculate_fresh_stats():
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print("No files found!")
        return None, None
        
    print(f"Found {len(all_files)} files. Calculating stats...")
    all_x = []
    # Just read first 200 for speed approximation, or all
    for fp in all_files:
        with np.load(fp) as data:
            all_x.append(data['x'].astype(np.float32))
            
    x_tensor = torch.tensor(np.array(all_x))
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    return mean, std

base_mean, base_std = load_baseline_stats()
fresh_mean, fresh_std = calculate_fresh_stats()

if base_mean is not None and fresh_mean is not None:
    print("\n--- Comparison ---")
    print(f"Baseline Mean Norm: {torch.norm(base_mean):.4f}")
    print(f"Fresh Mean Norm:    {torch.norm(fresh_mean):.4f}")
    
    diff_mean = torch.norm(base_mean - fresh_mean)
    diff_std = torch.norm(base_std - fresh_std)
    
    print(f"Difference in Mean: {diff_mean:.6f}")
    print(f"Difference in Std:  {diff_std:.6f}")
    
    if diff_mean > 1e-4 or diff_std > 1e-4:
        print(">>> DISCREPANCY DETECTED! Stats are different. <<<")
    else:
        print("Stats are identical.")
else:
    print("Could not load both sets of stats.")
