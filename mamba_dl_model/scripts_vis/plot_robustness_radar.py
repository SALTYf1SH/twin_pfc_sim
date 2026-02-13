import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import pi

# Config
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['svg.fonttype'] = 'none'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Input CSVs
CSV_Unified = os.path.join(BASE_DIR, "../robustness_mamba_ablation_results.csv")

# Output Dir
OUTPUT_DIR = os.path.join(BASE_DIR, "../robustness_radar_plots")

METRIC_CONFIG = {
    "MSE": -1,  # Lower is better
    "MAE": -1,
    "SSIM": 1,  # Higher is better
    "PCC": 1,
    "PCR": 1,
    "Evo": -1
}

MODEL_STYLE = {
    "full_dual": {"color": '#FF4500', "marker": 'o', "label": "Proposed (Mamba)"},
    "full_dynamic_only": {"color": '#4169E1', "marker": 's', "label": "Dynamic Only"},
    "full_static_only": {"color": '#32CD32', "marker": '^', "label": "Static Only"},
    "no_physics_dual_no_phys": {"color": '#FFD700', "marker": 'D', "label": "No Physics"},
    "vanilla_mamba": {"color": '#8A2BE2', "marker": 'v', "label": "Vanilla Mamba"},
    "LSTM": {"color": '#DC143C', "marker": 'p', "label": "LSTM"},
    "CNN": {"color": '#00CED1', "marker": '*', "label": "CNN"},
    "TRANSFORMER": {"color": '#8B4513', "marker": 'X', "label": "Transformer"},
}

# Define Model Groups
ABLATION_MODELS = [
    "full_dual", 
    "full_dynamic_only", 
    "full_static_only", 
    "no_physics_dual_no_phys", 
    "vanilla_mamba"
]

BASELINE_MODELS = [
    "full_dual", 
    "LSTM", 
    "CNN", 
    "TRANSFORMER",
    "vanilla_mamba"
]

def normalize_value(val, v_min, v_max, direction):
    if v_max - v_min < 1e-8:
        return 1.0
    
    if direction == 1: # Higher is better
        return 0.1 + 0.9 * (val - v_min) / (v_max - v_min)
    else: # Lower is better
        return 0.1 + 0.9 * (v_max - val) / (v_max - v_min)

def normalize_row(row, metrics, min_max_dict):
    """
    Normalize Mean, CI_Lower, CI_Upper.
    Note: For 'Lower is better', CI_Lower becomes the upper visual bound and vice versa.
    """
    norm_data = {}
    for m in metrics:
        direction = METRIC_CONFIG[m]
        v_min = min_max_dict[m]['min']
        v_max = min_max_dict[m]['max']
        
        mean_norm = normalize_value(row['Mean'], v_min, v_max, direction)
        
        # Normalize bounds
        # If Higher is better: Lower -> Lower Visual, Upper -> Upper Visual
        # If Lower is better: Lower (better value) -> Upper Visual, Upper (worse value) -> Lower Visual
        
        # We process raw values first
        ci_lower = row.get('CI_Lower', row['Mean'])
        ci_upper = row.get('CI_Upper', row['Mean'])
        
        # Invert logic for normalization
        n_ci_lower = normalize_value(ci_lower, v_min, v_max, direction)
        n_ci_upper = normalize_value(ci_upper, v_min, v_max, direction)
        
        # Ensure lower is numerically smaller for plotting filling
        norm_data[m] = {
            'mean': mean_norm,
            'lower': min(n_ci_lower, n_ci_upper),
            'upper': max(n_ci_lower, n_ci_upper)
        }
    return norm_data

def plot_radar(data_groups, metrics, title, output_path):
    categories = metrics
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 8 cm = 3.1496 inches
    FIG_W = 8 / 2.54
    FIG_H = 8 / 2.54
    
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), subplot_kw=dict(polar=True))
    
    # Draw axes
    # Ensure min font size >= 8
    plt.xticks(angles[:-1], categories, color='black', size=8) 
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=8) # Removed inner labels to save space on small plot, or keep empty
    # Re-adding labels if essential, but 8cm is small. Let's keep ticks but maybe hide text or keep it minimal?
    # User said "min font size >= 8", not "remove labels". I will keep labels but make sure they are size 8.
    # Actually, for cleanliness on a small chart, let's keep the grid but assume values are 0.2-1.0
    
    plt.ylim(0, 1.1)
    
    for idx, (label, group_data) in enumerate(data_groups.items()):
        
        means = [group_data[m]['mean'] for m in metrics]
        lowers = [group_data[m]['lower'] for m in metrics]
        uppers = [group_data[m]['upper'] for m in metrics]
        
        # Close the loop
        means += means[:1]
        lowers += lowers[:1]
        uppers += uppers[:1]
        
        # Get Style
        style = group_data['_style']
        color = style['color']
        marker = style['marker']
        
        # Highlighting Proposed Model
        zorder = 2
        # linewidth = 1.5 (Default)
        if "Proposed" in label:
            zorder = 10
        
        # Plot Mean Line
        # Reduced linewidth to 1.0 as requested
        ax.plot(angles, means, linewidth=1.0, linestyle='solid', label=label, color=color, marker=marker, markersize=3, zorder=zorder)
        
        # Validation: Fill error area
        ax.fill_between(angles, lowers, uppers, color=color, alpha=0.15, zorder=zorder)
        
    plt.title(title, size=10, y=1.1)
    # Legend at bottom, spread out
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    png_path = output_path.replace('.svg', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {png_path}")

def process_and_plot(df, task, plot_name, model_list, group_col='Config'):
    # 1. Filter by Task
    df_task = df[df['Task'] == task].copy()
    if df_task.empty:
        print(f"No data for task: {task}")
        return
        
    # 2. Filter by Model List
    df_task = df_task[df_task[group_col].isin(model_list)]
    if df_task.empty:
        print(f"No matching models found for {plot_name} in task {task}")
        return

    # 2. Get available metrics
    available_metrics = [m for m in METRIC_CONFIG.keys() if m in df_task['Metric'].unique()]
    
    # 3. Pivot to get rows as Models, Columns as Metrics (for calculating Min/Max globally)
    # But current format is long format (one row per metric per model)
    # We need to find global min/max for each metric across ALL models in this task comparison
    
    min_max_dict = {}
    for m in available_metrics:
        # Get all Mean values for this metric in this sub-group
        vals_mean = df_task[df_task['Metric'] == m]['Mean'].values
        
        # Consider bounds too if we want absolute limits, but valid range is best determined by means generally
        # However, to be safe, let's include bounds in range finding
        vals_lower = df_task[df_task['Metric'] == m]['CI_Lower'].values
        vals_upper = df_task[df_task['Metric'] == m]['CI_Upper'].values
        
        all_vals = np.concatenate([vals_mean, vals_lower, vals_upper])
        
        min_max_dict[m] = {
            'min': np.min(all_vals),
            'max': np.max(all_vals)
        }
        
    # 4. Prepare data for plotting
    # Group by Model
    # Sort models by order in model_list (for consistent legend order)
    df_task[group_col] = pd.Categorical(df_task[group_col], categories=model_list, ordered=True)
    df_task.sort_values(group_col, inplace=True)
    models = df_task[group_col].unique()
    
    data_groups = {}
    
    for model in models:
        model_df = df_task[df_task[group_col] == model]
        
        # Organize normalized data
        model_metrics = {}
        for _, row in model_df.iterrows():
            m = row['Metric']
            if m in available_metrics:
                # Normalize this single row's metric
                norm_dict = normalize_row(row, [m], min_max_dict)
                model_metrics[m] = norm_dict[m]
        
        # Rename for label and get style
        if model in MODEL_STYLE:
            style = MODEL_STYLE[model]
            label = style['label']
        else:
            # Fallback
            style = {"color": 'black', "marker": 'o', "label": model}
            label = model
            
        model_metrics['_style'] = style
        data_groups[label] = model_metrics
        
    # 5. Plot
    output_path = os.path.join(OUTPUT_DIR, f"{task}_{plot_name}.svg")
    plot_radar(data_groups, available_metrics, f"{task.capitalize()} - {plot_name.replace('_', ' ').title()}", output_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Unified Data
    if os.path.exists(CSV_Unified):
        df_all = pd.read_csv(CSV_Unified)
        print(f"Loaded Unified Results: {CSV_Unified}")
        
        # 1. Ablation Comparison
        print("Processing Ablation Radar...")
        process_and_plot(df_all, "stress", "ablation_comparison", ABLATION_MODELS)
        process_and_plot(df_all, "subsidence", "ablation_comparison", ABLATION_MODELS)
        
        # 2. Baseline Comparison
        print("Processing Baseline Radar...")
        process_and_plot(df_all, "stress", "baseline_comparison", BASELINE_MODELS)
        process_and_plot(df_all, "subsidence", "baseline_comparison", BASELINE_MODELS)
        
    else:
        print(f"File not found: {CSV_Unified}")

if __name__ == "__main__":
    main()
