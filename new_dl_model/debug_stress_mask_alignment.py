# -*- coding: utf-8 -*-
"""
DEBUG SCRIPT V3 (Physics Shape & Boundary Fix)

Fixes:
1. Geometry: Left side is now a SLOPED arch (Open-off cut), not a vertical wall.
2. Boundary Artifacts: Adds valid ROI mask to ignore simulation boundary errors.
3. Orientation: Ensures correct Transpose (.T).
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
# [请确认指向应力数据集]
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress") 
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

# Physical Constants
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
OUTPUT_SIZE = 64

# --- 1. 修正后的 Mask 生成器 (双端拱坡) ---

class TheoryConsistentMaskGenerator(nn.Module):
    def __init__(self, output_size=64, ks_sigma=5.0):
        super(TheoryConsistentMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        self.ks_sigma = ks_sigma
        
        # 物理坐标
        y_vals = torch.linspace(0, MODEL_HEIGHT_M, self.H) 
        x_vals = torch.linspace(0, MODEL_LENGTH_M, self.W)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def solve_height_ode(self, dist, h_max, k, ks_h, ks_b):
        """ODE 求解高度 (保持不变)"""
        if not torch.is_tensor(dist): dist = torch.tensor(dist)
        max_d = dist.item()
        if max_d < 1.0: max_d = 1.0
        steps = int(max_d / 1.0) + 2
        
        h_curr = 0.0
        h_trace = [h_curr]
        ks_h = torch.tensor(ks_h); ks_b = torch.tensor(ks_b)
        
        for _ in range(steps):
            diff = h_curr - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum()
            dh = k * (h_max - h_curr) / (1.0 + inhibition)
            h_curr = h_curr + dh
            if h_curr > h_max: h_curr = h_max
            h_trace.append(h_curr)
            
        idx = int(dist.item())
        if idx >= len(h_trace): idx = len(h_trace) - 1
        return h_trace[idx]

    def forward(self, mining_distance, params):
        h_max = params['h_max']; w_arch = params['width']; beta = params['beta']
        lag = params['lag']; k = params.get('k_growth', 0.02)
        ks_h = params.get('ks_heights', [35.0, 85.0])
        ks_b = params.get('ks_betas', [5.0, 8.0])
        
        # 1. 求解当前最大高度
        curr_H = self.solve_height_ode(mining_distance, h_max, k, ks_h, ks_b)
        
        # 2. 定义关键位置
        # xc_front: 工作面后方的拱心
        # xc_rear:  开切眼处的拱心 (固定在左侧 w_arch 处，保证 x=0 时高度为0)
        xc_front = mining_distance - lag
        xc_rear = w_arch 
        
        # 3. 计算前方曲线 (工作面)
        x_term_front = (self.xx - xc_front) / (w_arch + 1e-6)
        arch_front = curr_H * torch.pow((1 - x_term_front.pow(2)).clamp(min=0), beta)
        
        # 4. 计算后方曲线 (开切眼 - 解决垂直墙问题)
        # 这里使用同样的椭圆方程，中心在 w_arch，使得 x=0 时高度为0
        x_term_rear = (self.xx - xc_rear) / (w_arch + 1e-6)
        arch_rear = curr_H * torch.pow((1 - x_term_rear.pow(2)).clamp(min=0), beta)
        
        # 5. 组合包络线
        # 逻辑：
        # - 如果在 xc_rear 和 xc_front 之间 -> 平顶 (采空区)
        # - 如果在左边 -> 后方曲线
        # - 如果在右边 -> 前方曲线
        
        height_limit = torch.full_like(self.xx, curr_H) # 先假设全是平顶
        
        # 左侧削坡
        mask_rear = (self.xx < xc_rear)
        height_limit[mask_rear] = arch_rear[mask_rear]
        
        # 右侧削坡
        mask_front = (self.xx > xc_front)
        height_limit[mask_front] = arch_front[mask_front]
        
        # 6. 生成 Mask
        mask = (self.yy <= height_limit).float()
        
        # --- [NEW] 边界效应屏蔽 Mask (ROI) ---
        # 强制忽略左右 40m 的区域 (模拟边界效应)
        roi_mask = torch.ones_like(mask)
        roi_mask[:, :int(40 / (MODEL_LENGTH_M/OUTPUT_SIZE))] = 0 # 左边界屏蔽
        roi_mask[:, -int(40 / (MODEL_LENGTH_M/OUTPUT_SIZE)):] = 0 # 右边界屏蔽
        
        return mask, roi_mask

# --- 2. Visualization Logic ---

def debug_visualization():
    print(f"Checking Stress Dataset: {DATASET_DIR}")
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    if os.path.exists(PARAMS_JSON_PATH):
        with open(PARAMS_JSON_PATH, 'r') as f: 
            physics_params = json.load(f)
    else:
        physics_params = {}
    
    # 挑选几个比较典型的“坏”样本进行观察 (根据您图片中的 ID)
    target_ids = [14, 67, 137, 187, 192] 
    sample_files = []
    
    for t_id in target_ids:
        # 模糊匹配
        found = [f for f in all_files if f"sample_{t_id:04d}" in f]
        if found:
            # 选一个步数较大的
            sample_files.append(found[len(found)//2]) 
    
    # 如果找不到特定样本，随机补足
    while len(sample_files) < 5:
        sample_files.append(all_files[np.random.randint(len(all_files))])
    
    mask_gen = TheoryConsistentMaskGenerator(output_size=64)
    
    plt.figure(figsize=(20, 6))
    
    for i, filepath in enumerate(sample_files[:5]):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); s_id = int(filename[s_pos+7 : s_pos+11])
        st_pos = filename.rfind("step"); st_id = int(filename[st_pos+5 : st_pos+8])
        mining_dist = st_id * STEP_DISTANCE_M
        
        with np.load(filepath) as data:
            gt_img = data['y']
            
        if gt_img.ndim == 1: gt_img = gt_img.reshape(64, 64)
        gt_img = gt_img.T # Correct Orientation
        
        # Params
        default_params = {
            "h_max": 100.0, "width": 180.0, "beta": 3.0, "lag": 20.0, # width 调大一点以覆盖应力
            "k_growth": 0.02, "ks_heights": [35.0, 85.0], "ks_betas": [5.0, 8.0]
        }
        params = physics_params.get(str(s_id), default_params)
        
        # Generate Mask & ROI
        mask, roi = mask_gen(mining_dist, params)
        mask_np = mask.numpy()
        roi_np = roi.numpy()
        
        # Plot
        ax = plt.subplot(1, 5, i+1)
        
        # 1. GT (Gray)
        ax.imshow(gt_img, cmap='gray_r', origin='lower', extent=[0, 500, 0, 150], alpha=0.6)
        
        # 2. Mask (Red Overlay) - 现在左边应该有坡度了
        masked_overlay = np.ma.masked_where(mask_np < 0.5, mask_np)
        ax.imshow(masked_overlay, cmap='Reds', origin='lower', extent=[0, 500, 0, 150], alpha=0.4, vmin=0, vmax=1)
        
        # 3. ROI (Green Hatches) - 显示被忽略的边界区域
        roi_overlay = np.ma.masked_where(roi_np > 0.5, np.ones_like(roi_np)) # 只显示 0 的部分
        ax.imshow(roi_overlay, cmap='Greens', origin='lower', extent=[0, 500, 0, 150], alpha=0.3, vmin=0, vmax=1)
        
        # Calc PCR (Consider ROI)
        # 只在 ROI 区域内计算 PCR
        valid_gt = gt_img * roi_np
        valid_mask = mask_np * roi_np
        
        # 简单的二值化 PCR
        gt_bin = (valid_gt > 0.1).astype(float)
        intersection = (gt_bin * valid_mask).sum()
        total_energy = gt_bin.sum() + 1e-6
        pcr = intersection / total_energy
        
        ax.set_title(f"Sample {s_id} Step {st_id}\n(d={mining_dist:.0f}m)\nValid-PCR: {pcr:.3f}")
        ax.axvline(x=mining_dist, color='blue', linestyle='--', label='Face')
        
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 150)
        
    plt.tight_layout()
    plt.savefig("debug_stress_v3.png")
    print("Debug image saved: debug_stress_v3.png")
    print("Green areas = Ignored Boundary Artifacts")
    print("Red area = New Arch with Left Slope")

if __name__ == "__main__":
    debug_visualization()