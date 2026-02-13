# -*- coding: utf-8 -*-
"""
DEBUG SCRIPT V2 (FIXED): Fix Axes Orientation & Cumulative Mask & Reshape.

Fixes:
1. Reshape Check: Handles flattened inputs (4096,) -> (64, 64).
2. Transpose GT: Ensures Data (X, Y) matches Image (Row, Col).
3. Cumulative Mask: Covers rear goaf.
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset") 
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "subsidence_para/subsidence_physics_params.json")

# Physical Constants
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
OUTPUT_SIZE = 64

class ActivityArchMaskGenerator(nn.Module):
    def __init__(self, output_size=64):
        super(ActivityArchMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        
        # 网格生成 (物理坐标对齐)
        # dim 0: Y (高度) 
        # dim 1: X (长度)
        y_vals = torch.linspace(0, MODEL_HEIGHT_M, self.H)
        x_vals = torch.linspace(0, MODEL_LENGTH_M, self.W)
        
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def forward(self, mining_distance, h_max, w_arch, beta, lag):
        d = mining_distance
        
        # 1. 计算当前发育高度 (Logistic Growth)
        curr_H = h_max * np.tanh(d / 100.0)
        
        # 2. 定义关键坐标点
        # xc_front: 前方拱心 (跟随采煤面移动)
        # xc_rear:  后方拱心 (固定在开切眼附近，确保从 x=0 开始起坡)
        xc_front = d - lag
        xc_rear = w_arch  # 假设后方起坡的半宽也是 w_arch，这样 x=0 时高度为 0
        
        # 3. 计算“前方”下降曲线 (Front Curve)
        # valid for x > xc_front
        x_term_front = (self.xx - xc_front) / (w_arch + 1e-6)
        arch_front = curr_H * torch.pow((1 - x_term_front.pow(2)).clamp(min=0), beta)
        
        # 4. 计算“后方”上升曲线 (Rear Curve) [新增逻辑]
        # valid for x < xc_rear
        x_term_rear = (self.xx - xc_rear) / (w_arch + 1e-6)
        arch_rear = curr_H * torch.pow((1 - x_term_rear.pow(2)).clamp(min=0), beta)
        
        # 5. 构建包络线 (Envelope)
        # 逻辑：默认高度是平顶 (curr_H)
        # 如果在最左边 (x < xc_rear)，被后方曲线削减
        # 如果在最右边 (x > xc_front)，被前方曲线削减
        
        height_limit = torch.full_like(self.yy, curr_H) # 初始化为平顶
        
        # 左侧切角 (Slope Up)
        mask_rear = (self.xx < xc_rear)
        height_limit[mask_rear] = arch_rear[mask_rear]
        
        # 右侧切角 (Slope Down)
        mask_front = (self.xx > xc_front)
        height_limit[mask_front] = arch_front[mask_front]
        
        # 6. 生成二值 Mask
        # 物理限制: Y <= 高度包络线 且 X >= 0
        mask = (self.yy <= height_limit) & (self.xx >= 0)
        
        return mask.float(), xc_front

def debug_visualization():
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    with open(PARAMS_JSON_PATH, 'r') as f: 
        physics_params = json.load(f)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_files), 5, replace=False)
    sample_indices.sort() 
    
    mask_gen = ActivityArchMaskGenerator(output_size=64)
    
    plt.figure(figsize=(20, 6))
    
    for i, idx in enumerate(sample_indices):
        filepath = all_files[idx]
        filename = os.path.basename(filepath)
        
        s_pos = filename.rfind("sample"); s_id = int(filename[s_pos+7 : s_pos+11])
        st_pos = filename.rfind("step"); st_id = int(filename[st_pos+5 : st_pos+8])
        mining_dist = st_id * STEP_DISTANCE_M
        
        # --- 1. 加载并修复数据形状 ---
        with np.load(filepath) as data:
            gt_img = data['y']
            
        # [关键修复] 如果是一维数组 (4096,)，先变成二维 (64, 64)
        if gt_img.ndim == 1:
            gt_img = gt_img.reshape(64, 64)
            
        # --- 2. 坐标系对齐 (转置) ---
        # 物理坐标 (X, Y) -> 图像坐标 (Row=Y, Col=X)
        gt_img = gt_img.T  
        
        # Load Params
        params = physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        
        # Generate Mask
        mask, xc = mask_gen(mining_dist, params['h_max'], params['width'], params['beta'], params['lag'])
        mask_np = mask.numpy()
        
        # Calc PCR
        intersection = (gt_img * mask_np).sum()
        total_energy = gt_img.sum() + 1e-6
        pcr = intersection / total_energy
        
        # Plotting
        ax = plt.subplot(1, 5, i+1)
        
        # Ground Truth (Gray)
        # origin='lower' puts (0,0) at bottom-left corner
        ax.imshow(gt_img, cmap='gray_r', origin='lower', extent=[0, 500, 0, 150], alpha=0.6)
        
        # Mask (Red Overlay)
        masked_overlay = np.ma.masked_where(mask_np < 0.5, mask_np)
        ax.imshow(masked_overlay, cmap='Reds', origin='lower', extent=[0, 500, 0, 150], alpha=0.5, vmin=0, vmax=1)
        
        ax.set_title(f"Step {st_id}\n(d={mining_dist:.0f}m)\nPCR: {pcr:.3f}")
        ax.axvline(x=mining_dist, color='blue', linestyle='--', label='Face')
        ax.axvline(x=xc, color='green', linestyle=':', label='Center')
        
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 150)
        
    plt.tight_layout()
    save_path = "debug_mask_check_v2_fixed.png"
    plt.savefig(save_path)
    print(f"Debug image saved to '{save_path}'. Please verify the RED mask covers the GRAY cracks.")

if __name__ == "__main__":
    debug_visualization()