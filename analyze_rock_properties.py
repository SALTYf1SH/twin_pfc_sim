# -*- coding: utf-8 -*-
"""
Analyzes and visualizes the distribution of rock mechanics properties from the
processed npz dataset.

This script iterates through the npz files in the final_dataset directory,
extracts the unique rock property vectors (the first 11 elements of the 'x' array),
calculates descriptive statistics, and generates plots (histograms and box plots)
to visualize the parameter distributions.

How to run:
python analyze_rock_properties_v2.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
DATASET_DIR = "final_dataset"
OUTPUT_STATS_FILE = "rock_properties_statistics.txt"
OUTPUT_HIST_PLOT = "rock_properties_histograms.png"
OUTPUT_BOX_PLOT = "rock_properties_boxplots.png"
# --- 新增：SVG 格式输出 ---
OUTPUT_HIST_PLOT_SVG = "rock_properties_histograms.svg"
OUTPUT_BOX_PLOT_SVG = "rock_properties_boxplots.svg"


# Define the names for the 11 properties based on data_extractor.py
PROPERTY_NAMES = [
    # Sandstone (TYPE 0)
    'sandstone_emod',
    'sandstone_pb_ten',
    'sandstone_pb_coh',
    'sandstone_kratio',
    # Mudstone/Siltstone (TYPE 2)
    'mudstone_emod',
    'mudstone_pb_ten',
    'mudstone_pb_coh',
    'mudstone_kratio',
    # Key Stratum Thicknesses
    'main_ks_thickness',
    'primary_ks_thickness',
    'coal_seam_thickness'
]

# --- V3: Define groups based on physical property type ---
# This grouping allows us to plot boxplots on different subplots
# with independent Y-axes, solving the scale difference issue.
# --- 更新：中文标题 ---
PROPERTY_GROUPS = {
    '弹性模量 (emod)': [
        'sandstone_emod',
        'mudstone_emod',
    ],
    '强度 (抗拉强度 & 粘聚力)': [
        'sandstone_pb_ten',
        'sandstone_pb_coh',
        'mudstone_pb_ten',
        'mudstone_pb_coh',
    ],
    '摩擦相关 (kratio)': [
        'sandstone_kratio',
        'mudstone_kratio',
    ],
    '关键岩层厚度': [
        'main_ks_thickness',
        'primary_ks_thickness',
        'coal_seam_thickness'
    ]
}


def analyze_properties():
    """Main function to perform the analysis and plotting."""
    print("======================================================")
    print("       Rock Properties Distribution Analyzer        ")
    print("======================================================")

    # --- 1. Data Extraction and Deduplication ---
    npz_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not npz_files:
        print(f"FATAL: No .npz files found in '{DATASET_DIR}'. Please generate the dataset first.")
        return

    print(f"Found {len(npz_files)} total .npz files. Extracting unique simulations...")

    processed_simulations = set()
    rock_properties_data = []

    for f in tqdm(npz_files, desc="Processing files"):
        # Extract the base simulation name (e.g., ..._sample_0001) from the filename
        base_name = os.path.basename(f).split('_step_')[0]
        
        if base_name not in processed_simulations:
            try:
                with np.load(f) as data:
                    # The rock properties are the first 11 elements of the 'x' vector
                    properties = data['x'][:11]
                    rock_properties_data.append(properties)
                processed_simulations.add(base_name)
            except Exception as e:
                print(f"\nWARNING: Could not process file {f}. Reason: {e}")

    if not rock_properties_data:
        print("FATAL: No valid rock property data could be extracted.")
        return

    print(f"\nExtracted {len(rock_properties_data)} unique simulation parameter sets.")

    # --- 2. Statistical Analysis using Pandas ---
    df = pd.DataFrame(rock_properties_data, columns=PROPERTY_NAMES)

    print("\nCalculating descriptive statistics...")
    stats_summary = df.describe()
    print(stats_summary)

    # Save the statistics to a text file
    with open(OUTPUT_STATS_FILE, 'w', encoding='utf-8') as f: # 确保 UTF-8 编码
        f.write("岩石属性描述性统计\n")
        f.write("============================================\n")
        f.write(stats_summary.to_string())
    print(f"\nStatistics summary saved to '{OUTPUT_STATS_FILE}'")

    # --- 3. Visualization ---
    
    # --- 设置 Matplotlib 字体和样式 ---
    # 1. 设置默认字体为 'SimSun' (宋体) 来支持中文
    #    注意: 这也会将数字和英文字母默认设置为 'SimSun' 字体。
    # 2. 解决负号显示问题
    # 3. 增大基础字号
    # 4. --- 新增：设置 SVG 保存时文字为可编辑文本 ---
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False 
        plt.rcParams['font.size'] = 14 # 设置基础字号
        plt.rcParams['svg.fonttype'] = 'none' # 确保 SVG 中的文字是文本, 而不是路径
    except Exception as e:
        print(f"警告：设置中文字体 'SimSun' 失败。可能需要手动安装字体。错误: {e}")
        print("将继续使用默认英文字体。")

    
    # Plot 1: Histograms (Unchanged logic, updated text and style)
    print(f"\nGenerating histograms plot... ({OUTPUT_HIST_PLOT} and {OUTPUT_HIST_PLOT_SVG})")
    # 调整 Figure 尺寸使其更紧凑
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    
    for i, col in enumerate(df.columns):
        df[col].hist(ax=axes[i], bins=20, grid=False, edgecolor='black')
        axes[i].set_title(col, fontsize=12) # 增大子图标题字号
        axes[i].tick_params(axis='x', rotation=45, labelsize=10) # 调整刻度字号
        axes[i].tick_params(axis='y', labelsize=10)

    # Hide any unused subplots
    for i in range(len(df.columns), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('岩石属性分布直方图', fontsize=20) # 翻译并增大总标题
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_HIST_PLOT)
    plt.savefig(OUTPUT_HIST_PLOT_SVG, format='svg') # --- 新增：保存 SVG ---
    plt.close(fig)

    # --- Plot 2: Box Plots (Optimized with Subplots) ---
    print(f"Generating grouped box plots... ({OUTPUT_BOX_PLOT} and {OUTPUT_BOX_PLOT_SVG})")
    
    n_groups = len(PROPERTY_GROUPS)
    # 调整 Figure 尺寸使其更紧凑 (高度 * 4.5)
    fig, axes = plt.subplots(n_groups, 1, figsize=(10, 4.5 * n_groups))

    # Ensure 'axes' is always an array, even if n_groups is 1
    if n_groups == 1:
        axes = [axes]

    # Iterate through the defined groups and plot on each subplot
    for ax, (group_title, group_columns) in zip(axes, PROPERTY_GROUPS.items()):
        # Select the subset of the DataFrame for this group
        df_group = df[group_columns]
        
        # Create the boxplot on the specific axis 'ax'
        # The Y-axis scale will be automatically determined for THIS group only.
        df_group.boxplot(ax=ax, grid=False)
        
        ax.set_title(group_title, fontsize=16) # 增大子图标题字号
        ax.set_ylabel('参数值', fontsize=14) # 翻译 Y 轴标签并设置字号
        ax.tick_params(axis='x', rotation=30, labelsize=12) # 增大刻度字号
        ax.tick_params(axis='y', labelsize=12) # 增大刻度字号

    fig.suptitle('岩石属性箱形图对比 (按物理类型分组)', fontsize=20) # 翻译并增大总标题
    # Use tight_layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_BOX_PLOT)
    plt.savefig(OUTPUT_BOX_PLOT_SVG, format='svg') # --- 新增：保存 SVG ---
    plt.close(fig)

    print("\n======================================================")
    print("Analysis complete.")
    print("======================================================")

if __name__ == "__main__":
    # Set a more professional plot style (plt.style.use 可能会覆盖 rcParams)
    # plt.style.use('seaborn-v0_8-whitegrid') # 样式表可能会覆盖字体设置，将其移到 analyze_properties 内部或在设置字体后调用
    
    # 最佳实践是将样式设置放在函数内部，在设置字体之前
    plt.style.use('seaborn-v0_8-whitegrid')
    analyze_properties()

