import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import argparse
import pandas as pd
from math import pi

# SCI Formatting
# SCI Formatting (Width <= 8cm ~ 3.15 inch)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9 
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['svg.fonttype'] = 'none' # Ensure text is editable in SVG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data Path for Stress Ablation
CSV_PATH_MAMBA = os.path.join(BASE_DIR, "../evaluation_results_stress_mamba", "ablation_summary_mamba.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "../visualization_results_stress_radar")

def normalize_data(df, metrics, directions):
    """
    Min-Max Normalization to [0.1, 1.0].
    Directions: -1 for Lower is Better (MSE), 1 for Higher is Better (PCR)
    """
    df_norm = df.copy()
    for m, d in zip(metrics, directions):
        if m not in df.columns: continue
        vals = df[m].values
        v_min, v_max = np.min(vals), np.max(vals)
        
        if v_max - v_min < 1e-8:
            df_norm[m] = 1.0 
        else:
            if d == 1: # Higher is Better (e.g. SSIM, PCR)
                # Ratio: Val / Max (Best is Max => 1.0)
                df_norm[m] = vals / (v_max + 1e-9)
            else: # Lower is Better (e.g. MSE, Evo)
                # Ratio: Min / Val (Best is Min => 1.0)
                df_norm[m] = (v_min + 1e-9) / (vals + 1e-9)
    return df_norm

def plot_radar(df, metrics, output_path):
    categories = metrics
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 8cm = 3.15 inches
    fig, ax = plt.subplots(figsize=(3.15, 3.15), subplot_kw=dict(polar=True))
    
    colors = ['#FF4500', '#4169E1', '#32CD32', '#FFD700', '#8A2BE2']
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.xticks(angles[:-1], categories, color='black', size=12)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)
    
    for idx, row in df.iterrows():
        label = row['Model']
        # [FILTER] Strictly exclude Old Baseline / High Physics as requested
        if "High Physics" in label or "Old Baseline" in label:
            continue
            
        values = row[metrics].tolist()
        values += values[:1]
        
        label = row['Model']
        if "Proposed" in label: label = "Proposed (Mamba)"
        if "No Physics" in label: label = "No Physics"
        if "Static Only" in label: label = "Static Only"
        if "Dynamic Only" in label: label = "Dynamic Only"
        
        ax.plot(angles, values, linewidth=1.0, linestyle='solid', label=label, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], markersize=3)
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.1)
        
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved Radar Chart to: {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    data_path = None
    if os.path.exists(CSV_PATH_MAMBA): 
        data_path = CSV_PATH_MAMBA
    
    if data_path is None:
        print("Warning: No Stress Ablation CSV found. Generating Placeholder Data.")
        data = {
            "Model": ["Proposed (Mamba)", "No Physics", "Static Only", "Dynamic Only"],
            "MSE": [0.00037, 0.00030, 0.00468, 0.00087], 
            "Evo": [0.39, 0.35, 1.39, 0.59],
            "PCR": [0.63, 0.59, 0.39, 0.60],
            # Hypothetical
            "SSIM": [0.95, 0.94, 0.80, 0.92],
            "PCC": [0.98, 0.98, 0.85, 0.96]
        }
        df = pd.DataFrame(data)
    else:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    
    # 2. Config Metrics (Available in Eval)
    metric_config = {
        "MSE": -1,
        "MAE": -1,
        "SSIM": 1,
        "PCC": 1,
        "Evo": -1,
        "PCR": 1
    }
    
    # Filter columns
    available = df.columns
    metrics = [m for m in metric_config.keys() if m in available]
    directions = [metric_config[m] for m in metrics]
    
    print(f"Metrics: {metrics}")
    
    # 3. Normalize
    df_norm = normalize_data(df, metrics, directions)
    
    # 4. Plot
    output_path = os.path.join(OUTPUT_DIR, "stress_ablation_radar_comparison.svg")
    plot_radar(df_norm, metrics, output_path)

if __name__ == "__main__":
    main()
