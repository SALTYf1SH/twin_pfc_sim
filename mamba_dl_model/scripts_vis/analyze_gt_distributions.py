
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_STRESS = os.path.join(BASE_DIR, "../../final_dataset_stress")
DATA_DIR_SUBSIDENCE = os.path.join(BASE_DIR, "../../final_dataset")
VIS_DIR = os.path.join(BASE_DIR, "../dataset_vis")

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

# Mappings derived from `parameter_sampler_hqh.py` (Stress - 17 features)
FEATURE_MAP_STRESS = {
    0: "Sandstone Modulus",
    1: "Sandstone Tensile Str.",
    2: "Sandstone Cohesion",
    3: "Sandstone Stiffness Ratio",
    4: "Gritstone Modulus",
    5: "Gritstone Tensile Str.",
    6: "Gritstone Cohesion",
    7: "Gritstone Stiffness Ratio",
    8: "Mudstone Modulus",
    9: "Mudstone Tensile Str.",
    10: "Mudstone Cohesion",
    11: "Coal Modulus",
    12: "Coal Tensile Str.",
    13: "Coal Cohesion",
    14: "Main Key Stratum Thickness",
    15: "Coal Seam Thickness",
    16: "Immediate Floor Thickness"
}

# Mappings derived from `parameter_sampler.py` (Subsidence - 11 features)
FEATURE_MAP_SUBSIDENCE = {
    0: "Sandstone Modulus",
    1: "Sandstone Tensile Str.",
    2: "Sandstone Cohesion",
    3: "Sandstone Stiffness Ratio",
    4: "Mudstone Modulus",
    5: "Mudstone Tensile Str.",
    6: "Mudstone Cohesion",
    7: "Mudstone Stiffness Ratio",
    8: "Main Key Stratum Thickness",
    9: "Primary Key Stratum Thickness",
    10: "Coal Seam Thickness"
}

def set_sci_style():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11 
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.grid'] = True 
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['svg.fonttype'] = 'none' # Ensure text is editable in SVG

def get_feature_color(feature_name):
    # Okabe-Ito Color Palette (Colorblind Safe)
    name_lower = feature_name.lower()
    if "sandstone" in name_lower: return "#D55E00" # Vermillion
    if "gritstone" in name_lower: return "#E69F00" # Orange
    if "mudstone" in name_lower: return "#009E73"  # Bluish Green
    if "coal" in name_lower: return "#CC79A7"      # Reddish Purple (distinct from others)
    if "thickness" in name_lower: return "#56B4E9" # Sky Blue
    return "#999999" # Grey

def plot_grouped_half_violin(ax, df, cols, group_name):
    # Data preparation
    data = []
    labels = []
    colors = []
    
    # Heuristic to simplify labels based on group name
    for c in cols:
        val = df[c].dropna().values
        data.append(val)
        colors.append(get_feature_color(c))
        
        # Simplified label
        lbl = c
        if "Modulus" in group_name: lbl = lbl.replace("Modulus", "").strip()
        if "Strength" in group_name: lbl = lbl.replace("Tensile Str.", "Tensile Strength").strip()
        if "Ratio" in group_name: lbl = lbl.replace("Stiffness Ratio", "").strip()
        if "Thickness" in group_name: lbl = lbl.replace("Thickness", "").strip()
        
        labels.append(lbl)

    # Plot Violin at positions
    positions = np.arange(1, len(cols) + 1)
    # Using 'points' usually requires more data points for smooth graphs, bandwidth estimation can help
    parts = ax.violinplot(data, positions=positions, vert=True, 
                          showmeans=False, showmedians=False, showextrema=False, widths=0.7)
    
    # Style the violin bodies
    for i, b in enumerate(parts['bodies']):
        # Half-violin clip
        pos = positions[i]
        path = b.get_paths()[0]
        verts = path.vertices
        # Clip x < pos to make it right-sided half-violin
        verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
        
        color = colors[i]
        b.set_facecolor(color)
        b.set_alpha(0.85)
        b.set_edgecolor(None) # Remove hard outline for softer look
        
    # Add statistics (Box plot style overlay)
    for i, d in enumerate(data):
        pos = positions[i]
        q1, q2, q3 = np.percentile(d, [25, 50, 75])
        
        # Line for range (whisker-ish, but maybe just IQR is cleaner for this density)
        # Let's do a thin line for min-max (excluding outliers for aesthetics? Or just full range?)
        # For simplicity and beauty: IQR line + Median point
        ax.vlines(pos, q1, q3, color='black', linestyle='-', lw=2, alpha=0.8)
        ax.scatter(pos, q2, marker='o', color='white', s=25, edgecolor='black', zorder=3)

    ax.set_xticks(positions)
    # Enforce consistent rotation for all subplots
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

    ax.set_ylabel(group_name, fontsize=12, fontweight='bold')
    
    # Cleaner grid handled by style function

def identify_groups(columns):
    groups = {
        "Elastic Modulus (Pa)": [],
        "Strength Parameters (Pa)": [], # Cohesion & Tensile
        "Stiffness Ratios": [],
        "Layer Thickness (m)": [],
        "Others": []
    }
    
    for c in columns:
        if "Modulus" in c:
            groups["Elastic Modulus (Pa)"].append(c)
        elif "Cohesion" in c or "Tensile" in c:
            groups["Strength Parameters (Pa)"].append(c)
        elif "Ratio" in c:
            groups["Stiffness Ratios"].append(c)
        elif "Thickness" in c:
            groups["Layer Thickness (m)"].append(c)
        else:
            groups["Others"].append(c)
            
    return {k: v for k, v in groups.items() if v}

def process_dataset(name, data_dir, feature_count, feature_map):
    print(f"\nProcessing {name} Dataset...")
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files: return
    
    # Load Data
    data_list = []
    print(f"Loading {len(files)} files...")
    # Cap at 500 files for speed if needed, but distribution should be stable
    # User implies we are finalizing, so use full or large subset.
    # Using full or reasonable limit.
    for f in tqdm(files): 
        try:
            with np.load(f) as d:
                data_list.append(d['x'][:feature_count])
        except: pass
    
    if not data_list: return
    df = pd.DataFrame(data_list)
    
    # Rename
    new_cols = {i: feature_map.get(i, f"Feature {i}") for i in range(feature_count)}
    df = df.rename(columns=new_cols)
    
    # Stats
    df.describe().transpose().to_csv(os.path.join(VIS_DIR, f"{name.lower()}_parameter_stats.csv"))
    
    # Grouping
    groups = identify_groups(df.columns)
    
    # Plotting
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    
    set_sci_style()
    
    # 1. Sort groups by size (descending) to optimize packing
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 2. Bin Packing (Greedy)
    MAX_WIDTH = 8
    rows = []
    current_row = []
    current_width = 0
    
    for g_name, g_cols in sorted_groups:
        w = len(g_cols)
        if current_width + w <= MAX_WIDTH:
            current_row.append((g_name, g_cols))
            current_width += w
        else:
            rows.append(current_row)
            current_row = [(g_name, g_cols)]
            current_width = w
    if current_row:
        rows.append(current_row)
        
    # 3. Setup Figure
    n_rows = len(rows)
    fig_height = n_rows * 3.5 # Increased height per row to prevent vertical overlap
    fig_width = 17 / 2.54
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Increased hspace for vertical breathing room
    outer_gs = gridspec.GridSpec(n_rows, 1, height_ratios=[1]*n_rows, hspace=0.6)
    
    legend_ax = None
    
    for r_idx, row_items in enumerate(rows):
        # Calculate width ratios
        ratios = [len(cols) for _, cols in row_items]
        total_occupied = sum(ratios)
        if total_occupied < MAX_WIDTH:
            ratios.append(MAX_WIDTH - total_occupied) # Spacer
            
        # Increased wspace for horizontal breathing room between subplots
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, len(ratios), subplot_spec=outer_gs[r_idx], 
                                                    width_ratios=ratios, wspace=0.4) 
        
        for c_idx, (g_name, g_cols) in enumerate(row_items):
            ax = fig.add_subplot(inner_gs[c_idx])
            plot_grouped_half_violin(ax, df, g_cols, g_name)
            # Reset X-limit to be local to this subplot (1..N)
            # Since physical width is proportional to N, unit scale is preserved.
            ax.set_xlim(0.5, len(g_cols) + 0.5)
            
        # Potentially use the spacer for legend if this is the last row or has space
        if r_idx == n_rows - 1 and len(ratios) > len(row_items):
            legend_ax = fig.add_subplot(inner_gs[-1])
            legend_ax.axis('off')

    # 4. Add Legend
    # Define colors manually or grab from helper
    legend_patches = [
        mpatches.Patch(color='#D55E00', label='Sandstone'),
        mpatches.Patch(color='#E69F00', label='Gritstone'),
        mpatches.Patch(color='#009E73', label='Mudstone'),
        mpatches.Patch(color='#CC79A7', label='Coal'),
        mpatches.Patch(color='#56B4E9', label='Thickness')
    ]
    
    if legend_ax:
        legend_ax.legend(handles=legend_patches, loc='center', title="Parameter Type", frameon=False, fontsize=10)
    else:
        # Fallback if no space in grid (unlikely with this heuristic)
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5)

    # Removed tight_layout() to respect manual spacing
    plt.savefig(os.path.join(VIS_DIR, f"{name.lower()}_parameter_distributions.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(VIS_DIR, f"{name.lower()}_parameter_distributions.svg"), format='svg', bbox_inches='tight')
    print(os.path.join(VIS_DIR, f"{name.lower()}_parameter_distributions.png"))
    plt.close()
    
    # Correlation (Preserve existing logic)
    # Correlation (Improved)
    plt.figure(figsize=(fig_width, fig_width * 0.85))
    corr = df.corr()
    
    # Shorten names for heatmap
    short_cols = [c.replace("Modulus", "Mod").replace("Sandstone", "SS").replace("Gritstone", "GS").replace("Mudstone", "MS").replace("Tensile Str.", "Tensile").replace("Thickness", "Thick") for c in df.columns]
    
    # Mask the upper triangle for cleaner look
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Choose a divergent palette (bluish for negative, reddish for positive) or viridis
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=short_cols, yticklabels=short_cols,
                annot=True, fmt=".2f", annot_kws={"size": 8}) # Annot stays but smaller
                
    plt.title(f"{name} Correlation Matrix", fontsize=12, pad=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{name.lower()}_correlation_matrix.png"), dpi=300)
    plt.savefig(os.path.join(VIS_DIR, f"{name.lower()}_correlation_matrix.svg"), format='svg')
    plt.close()
    print(f"Saved plots for {name}")

def main():
    # Process Stress (17 features)
    process_dataset("Stress", DATA_DIR_STRESS, 17, FEATURE_MAP_STRESS)
    
    # Process Subsidence (11 features)
    process_dataset("Subsidence", DATA_DIR_SUBSIDENCE, 11, FEATURE_MAP_SUBSIDENCE)

if __name__ == "__main__":
    main()
