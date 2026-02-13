# -*- coding: utf-8 -*-
"""
Analysis of Coupling between Subsidence and Fracture Intensity

This script loads the processed .npz dataset and analyzes the statistical
relationship (correlation) between surface subsidence characteristics and
the resulting fracture field intensity.

It calculates:
1.  Subsidence Basin Depth (max absolute subsidence) from the 'x' vector.
2.  Mean Fracture Intensity from the 'y' vector.
3.  The Pearson Correlation Coefficient (PCC) between these two metrics
    across the entire dataset.
4.  Generates a scatter plot to visualize the coupling.
"""

import os
import glob
import numpy as np
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# --- Configuration ---

# 1. 默认的数据集目录
DEFAULT_DATASET_DIR = "final_dataset"

# 2. X 向量中静态参数（岩石属性）的数量
# (这必须与 process_new_data.py 中的 get_simulation_parameters 一致)
NUM_STATIC_PARAMS = 11

def analyze_correlation(dataset_dir):
    """
    Main analysis function.
    """
    print("======================================================")
    print("      Subsidence & Fracture Coupling Analysis       ")
    print("======================================================")
    print(f"Loading dataset from: {dataset_dir}")

    # --- 1. 查找所有 .npz 文件 ---
    search_path = os.path.join(dataset_dir, "*.npz")
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"FATAL ERROR: No .npz files found in '{dataset_dir}'.")
        print("Please run the data processing script first.")
        return

    print(f"Found {len(all_files)} data samples to analyze.")

    # --- 2. 遍历文件并提取特征 ---
    basin_depths = []
    fracture_means = []

    for filepath in tqdm(all_files, desc="Analyzing samples"):
        try:
            with np.load(filepath) as data:
                x = data['x']
                y = data['y']

                # a. 计算裂隙场强度均值 (Mean Fracture Intensity)
                # y 是 (64*64,) 的扁平化数组
                fracture_mean = np.mean(y)

                # b. 计算沉降盆地深度 (Subsidence Basin Depth)
                # x 是 [11_params, subsidence_vector]
                subsidence_vector = x[NUM_STATIC_PARAMS:]
                
                # 沉降值为负，因此"深度"是最小值的绝对值
                basin_depth = np.abs(np.min(subsidence_vector))

                # 排除无效数据（例如，如果某个步骤没有沉降或裂隙）
                if basin_depth > 1e-6 and fracture_mean > 1e-6:
                    basin_depths.append(basin_depth)
                    fracture_means.append(fracture_mean)

        except Exception as e:
            print(f"\nWarning: Could not load or process file {os.path.basename(filepath)}. Error: {e}")

    if not basin_depths:
        print("FATAL ERROR: No valid data could be extracted. Analysis aborted.")
        return
        
    print(f"\nSuccessfully processed {len(basin_depths)} valid samples.")

    # --- 3. 计算相关系数 ---
    # 转换为 numpy 数组
    depths_array = np.array(basin_depths)
    means_array = np.array(fracture_means)

    # 计算皮尔逊相关系数 (PCC) 和 p-value
    correlation, p_value = scipy.stats.pearsonr(depths_array, means_array)

    print("\n======================================================")
    print("               Correlation Results                ")
    print("======================================================")
    print(f"  指标 1: 沉降盆地深度 (最大沉降绝对值)")
    print(f"  指标 2: 裂隙场强度均值 (KDE 均值)")
    print("------------------------------------------------------")
    print(f"  皮尔逊相关系数 (r): {correlation:.6f}")
    print(f"  P-value:              {p_value:.4e}")
    if p_value < 0.05:
        print("  结论: 相关性在统计上显著 (p < 0.05)。")
    else:
        print("  结论: 相关性在统计上不显著 (p >= 0.05)。")
    print("======================================================")

    # --- 4. 可视化 ---
    print("\nGenerating correlation scatter plot...")
    
    # 设置 Matplotlib 字体 (与您之前的请求一致)
    plt.rcParams['font.sans-serif'] = ['SimSun'] # 中文字体设置为宋体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    plt.rcParams['font.size'] = 14
    font_tnr = {'family': 'Times New Roman', 'size': 14}
    font_title = {'family': 'SimSun', 'size': 18}
    font_label = {'family': 'SimSun', 'size': 16}

    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(depths_array, means_array, alpha=0.3, s=15, label="数据样本")
    
    # 添加回归线 (可选, 但很有用)
    try:
        m, b = np.polyfit(depths_array, means_array, 1)
        plt.plot(depths_array, m*depths_array + b, color='red', linestyle='--', label=f'线性拟合 (r={correlation:.3f})')
    except np.linalg.LinAlgError:
        print("Warning: Linear fit failed.")

    # 设置图表元素
    plt.title(f"沉降深度与裂隙强度的耦合关系 (N={len(depths_array)})", **font_title)
    plt.xlabel("沉降盆地深度 (m)", **font_label)
    plt.ylabel("裂隙场强度均值", **font_label)
    
    # 设置刻度字体
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(12)
        
    plt.legend(prop={'family': 'SimSun', 'size': 12})
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(dataset_dir, "subsidence_vs_fracture_correlation.png")
    plt.savefig(save_path, dpi=300)
    print(f"Correlation plot saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze coupling between subsidence and fracture intensity.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default=DEFAULT_DATASET_DIR, 
        help=f"Directory containing the .npz dataset (default: {DEFAULT_DATASET_DIR})"
    )
    args = parser.parse_args()
    
    analyze_correlation(args.dataset_dir)