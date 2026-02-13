import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR_ROOT = os.path.join(BASE_DIR, "../")
VIS_DIR = os.path.join(BASE_DIR, "weight_sensitivity")

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

# Configurations (Must match run_weight_sensitivity.py)
# Configurations (Must match run_weight_sensitivity.py)
# Configurations (Must match run_weight_sensitivity.py)
CONFIGS = {
    "physics_low":      "Low Physics",
    "physics_baseline": "Baseline",
    "physics_high":     "High Physics",
    "ssim_low":         "Low SSIM",
    "ssim_high":        "High SSIM"
}

TASKS = ["stress", "subsidence"]

# --- SCI Style Helper ---
def set_sci_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 9 
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['axes.grid'] = False # Ablation script handles grid manually or relies on defaults? Reverted script has no ax.grid() call but polar plots have default grids. Let's keep it simple.
    plt.rcParams['svg.fonttype'] = 'none' 
    plt.rcParams['lines.linewidth'] = 1.0

def load_results():
    results = []
    
    for task in TASKS:
        for config_key, config_label in CONFIGS.items():
            dir_name = f"trained_models_{task}_sens_{config_key}"
            metrics_path = os.path.join(MODELS_DIR_ROOT, dir_name, "val_metrics.json")
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        data = json.load(f)
                    
                    entry = {
                        "Task": task.capitalize(),
                        "Config": config_label,
                        "Epoch": data.get("epoch"),
                        "Total Loss": data.get("val_loss"),
                        "MSE": data.get("val_mse"),
                        "SSIM": 1.0 - data.get("val_ssim", 0), # Convert Loss to Score
                        "Physics Loss": data.get("val_arch"),
                        "Evolution Loss": data.get("val_evo")
                    }
                    results.append(entry)
                except Exception as e:
                    print(f"Error loading {metrics_path}: {e}")
            else:
                print(f"Missing: {metrics_path}")

    return pd.DataFrame(results)

def plot_bar_metrics(df):
    if df.empty: return

    set_style = set_sci_style()
    
    # We focus on MSE and Total Loss for the main bar plot comparison
    # But let's do a grouped bar plot for critical metrics: MSE, Physics, Evolution
    
    metrics_to_plot = ["MSE", "Physics Loss", "Evolution Loss"]
    
    # Layout: 2 rows (Stress, Subsidence), 3 cols (Metrics) 
    # Or 1 row per task with grouped bars? 
    # Let's do 1 figure per task to keep it clean for papers (8.5cm or 17cm width)
    
    # Figure width: 17cm ~ 6.7 inches
    fig_width = 17 / 2.54 
    fig_height = 8 / 2.54
    
    for task in ["Stress", "Subsidence"]:
        task_df = df[df["Task"] == task].copy()
        if task_df.empty: continue
        
        # Melt for seaborn
        task_df_melted = task_df.melt(id_vars=["Config"], value_vars=metrics_to_plot, var_name="Metric", value_name="Value")
        
        # Order configs
        task_df_melted["Config"] = pd.Categorical(task_df_melted["Config"], categories=CONFIGS.values(), ordered=True)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Log Scale might be needed if values differ vastly? 
        # Checking data ranges... typically MSE is small, Physics can be large.
        # Let's use separate subplots for each metric to strictly compare weights
        
        g = sns.FacetGrid(task_df_melted, col="Metric", sharey=False, height=3, aspect=1.0)
        g.map_dataframe(sns.barplot, x="Config", y="Value", hue="Config", palette="viridis", edgecolor="black", linewidth=0.8, legend=False)
        
        g.set_titles("{col_name}")
        g.set_axis_labels("", "Loss Value")
        
        for ax in g.axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f"{task} Inversion - Sensitivity Analysis", fontsize=12, fontweight='bold')
        
        out_path = os.path.join(VIS_DIR, f"bar_sensitivity_{task.lower()}.png")
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.savefig(out_path.replace(".png", ".svg"), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

def plot_radar_charts(df):
    if df.empty: return
    set_sci_style()

    metrics = ["MSE", "SSIM", "Physics Loss", "Evolution Loss"]
    
    # Define metric directions: 1 for Higher is Better, -1 for Lower is Better
    metric_directions = {
        "MSE": -1,
        "SSIM": 1, 
        "Physics Loss": -1,
        "Evolution Loss": -1
    }
    
    # Ablation Style Colors & Markers
    colors = ['#FF4500', '#4169E1', '#32CD32', '#FFD700', '#8A2BE2', '#A52A2A', '#00CED1'] 
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for i, task in enumerate(["Stress", "Subsidence"]):
        task_df = df[df["Task"] == task].copy()
        if task_df.empty: continue
        
        task_df.set_index("Config", inplace=True)
        data = task_df[metrics]
        
        # Define Gradient Groups
        gradients = {
            "Physics Sensitivity": ["Low Physics", "Baseline", "High Physics"],
            "SSIM Sensitivity":    ["Low SSIM", "Baseline", "High SSIM"]
        }
        
        # Iterate over Gradient Groups first, then Normalize locally
        for gradient_name, config_list in gradients.items():
            # Filter Data for this specific plot
            valid_configs = [c for c in config_list if c in data.index]
            if not valid_configs: continue
            
            subset_data = data.loc[valid_configs]
            
            # Normalize Locally using Ratio (Value / Best)
            # Best is always 1.0 (Outer Edge)
            data_norm = pd.DataFrame(index=subset_data.index, columns=subset_data.columns)
            for m in metrics:
                vals = subset_data[m].values
                v_min, v_max = np.min(vals), np.max(vals)
                direction = metric_directions[m]
                
                if direction == 1: # Higher is Better (e.g. SSIM)
                    # Ratio: Val / Max (Best is Max => 1.0)
                    data_norm[m] = vals / (v_max + 1e-9)
                else: # Lower is Better (e.g. MSE, Loss)
                    # Ratio: Min / Val (Best is Min => 1.0)
                    data_norm[m] = (v_min + 1e-9) / (vals + 1e-9)

            # 8cm = 3.15 inches
            fig, ax = plt.subplots(figsize=(3.15, 3.15), subplot_kw=dict(polar=True))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            metric_labels = metrics 
            # Custom Label Placement
            plt.xticks([]) # Hide default
            
            for ang, label in zip(angles[:-1], metric_labels):
                # Determine rotation and alignment
                deg = np.degrees(ang)
                if abs(deg - 0) < 1: # Right (MSE)
                    rot = -90
                    ha = 'center'; va = 'bottom' # Anchor text baseline to radius
                    pad_dist = 1.1 
                elif abs(deg - 180) < 1: # Left (Physics)
                    rot = 90
                    ha = 'center'; va = 'bottom' # Anchor text baseline to radius
                    pad_dist = 1.1
                else:
                    rot = 0
                    ha = 'center'; va = 'center'
                    pad_dist = 1.15
                    
                ax.text(ang, pad_dist, label, rotation=rot, ha=ha, va=va, fontsize=10, fontweight='bold')
                
            ax.tick_params(pad=0) # Reset pad
            
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            plt.ylim(0, 1.0)
            
            for idx, config_name in enumerate(valid_configs):
                values = data_norm.loc[config_name].tolist()
                values += values[:1]
                
                # Style
                style_idx = idx % len(colors)
                if config_name == "Baseline":
                     color = 'red'; marker = '*'
                elif "Low" in config_name:
                     color = '#4169E1'; marker = 'o' # Blue for Low
                elif "High" in config_name:
                     color = '#32CD32'; marker = '^' # Green for High
                else:
                     color = colors[style_idx]; marker = markers[style_idx]
                
                ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=config_name, color=color, marker=marker, markersize=4)
                ax.fill(angles, values, color=color, alpha=0.1)
                
            # Legend
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, frameon=False)
            
            out_name = f"radar_{gradient_name.split()[0].lower()}_sensitivity_{task.lower()}.svg"
            output_path = os.path.join(VIS_DIR, out_name)
            plt.tight_layout()
            plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
            plt.savefig(output_path.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {output_path}")

def main():
    print("Loading Sensitivity Results...")
    df = load_results()
    
    if df.empty:
        print("No results found.")
        return

    print("Generating Professional Bar Plots...")
    plot_bar_metrics(df)
    
    print("Generating Professional Radar Plots...")
    plot_radar_charts(df)
    
    # Save summary CSV
    df.to_csv(os.path.join(VIS_DIR, "sensitivity_summary.csv"), index=False)
    print(f"Summary saved.")

if __name__ == "__main__":
    main()
