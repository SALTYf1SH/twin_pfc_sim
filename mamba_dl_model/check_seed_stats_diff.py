import torch
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUSTNESS_DIR = os.path.join(BASE_DIR, "robustness_results_stress")

seeds = [42, 43, 44]
stats_files = {}

print("Checking stats files in:", ROBUSTNESS_DIR)

for seed in seeds:
    fname = f"stress_stats_seed{seed}.pt"
    fpath = os.path.join(ROBUSTNESS_DIR, fname)
    if os.path.exists(fpath):
        stats_files[seed] = torch.load(fpath)
        print(f"Loaded {fname}")
    else:
        print(f"MISSING: {fname}")

if len(stats_files) < 2:
    print("Not enough files to compare.")
    sys.exit(0)

# Compare 42 vs others
base_mean = stats_files[42]['mean']
base_std = stats_files[42]['std']

for seed in [43, 44]:
    if seed not in stats_files: continue
    
    curr_mean = stats_files[seed]['mean']
    curr_std = stats_files[seed]['std']
    
    diff_mean = torch.norm(base_mean - curr_mean).item()
    diff_std = torch.norm(base_std - curr_std).item()
    
    print(f"\nComparing Seed 42 vs {seed}:")
    print(f"  Mean Diff Norm: {diff_mean}")
    print(f"  Std Diff Norm:  {diff_std}")
    
    if diff_mean > 1e-6 or diff_std > 1e-6:
        print(f"  >>> WARNING: Stats differ for seed {seed}!")
    else:
        print(f"  Stats match.")
