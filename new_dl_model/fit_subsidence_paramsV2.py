# -*- coding: utf-8 -*-
"""
fit_subsidence_params_ks.py (Key Strata Aware Version)
沉降反演模型-物理参数反算脚本 (含关键层动力学)

功能：
基于沉降数据集 (Ground Truth 裂隙场)，反演控制裂隙发育的动力学参数。
该脚本完全遵循论文第四章理论模型，通过求解微分方程来拟合参数。

对应理论公式：
1. 几何形态 (Eq. 3): y = H * [1 - ((x-xc)/W)^2]^beta
2. 高度演化 (Eq. 4): dH/dt 受关键层 (KS) 抑制

输出：
subsidence_para/subsidence_physics_params.json
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- 0. 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# [适配] 指向沉降数据集
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset") 
# [适配] 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "subsidence_para")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "subsidence_physics_params.json")

# 物理场景常数
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
IMG_SIZE = 64

# --- 1. 可微物理模型 (Differentiable Physics Model) ---

class KeyStrataFitter(nn.Module):
    """
    内嵌 ODE 求解器的拟合模型。
    用于自动识别 H_max, Width, Lag 以及 关键层位置(y_ks) 和 强度(beta_ks)。
    """
    def __init__(self):
        super().__init__()
        
        # --- A. 基础几何参数 ---
        # H_max: 理论最大高度 (50~150m)
        self.h_max = nn.Parameter(torch.tensor(100.0))
        # Width: 拱半宽 (50~120m)
        self.width = nn.Parameter(torch.tensor(94.0))
        # Shape Beta: 拱形指数 (对应 Eq. 3)
        self.shape_beta = nn.Parameter(torch.tensor(3.0))
        # Lag: 滞后距离
        self.lag = nn.Parameter(torch.tensor(20.0))
        
        # --- B. 动力学参数 (Eq. 4) ---
        # k: 基础生长速率
        self.k_growth = nn.Parameter(torch.tensor(0.02))
        
        # --- C. 关键层参数 (Key Strata) ---
        # 假设存在两层主要关键层，初始化在常见层位 (35m, 85m)
        self.ks_heights = nn.Parameter(torch.tensor([35.0, 85.0]))
        # 关键层阻滞系数 beta_i (越大阻力越大)
        self.ks_betas = nn.Parameter(torch.tensor([5.0, 8.0]))
        # 关键层影响范围 sigma (固定)
        self.ks_sigma = 5.0 

        # 坐标网格缓存
        y_vals = torch.linspace(0, MODEL_HEIGHT_M, IMG_SIZE)
        x_vals = torch.linspace(0, MODEL_LENGTH_M, IMG_SIZE)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def get_constrained_params(self):
        """物理约束限制，防止参数优化溢出"""
        h_max = torch.clamp(self.h_max, 50.0, 145.0)
        width = torch.clamp(self.width, 40.0, 150.0)
        beta = torch.clamp(self.shape_beta, 1.5, 10.0)
        lag = torch.clamp(self.lag, 0.0, 60.0)
        k = torch.clamp(self.k_growth, 0.005, 0.1)
        
        ks_h = torch.clamp(self.ks_heights, 10.0, 140.0)
        ks_b = torch.relu(self.ks_betas) # 阻力必须为正
        
        return h_max, width, beta, lag, k, ks_h, ks_b

    def solve_height_evolution(self, dist_seq):
        """
        [核心算法] 使用欧拉积分求解微分方程 Eq. 4
        dH/dx = k * (Hmax - H) / (1 + sum(beta * exp(...)))
        """
        h_max, _, _, _, k, ks_h, ks_b = self.get_constrained_params()
        
        # 积分设置
        dx = 1.0 # 步长 1米
        max_dist = dist_seq[-1]
        steps = int(max_dist.item() / dx)
        
        h_evolution = [torch.tensor(0.0).to(dist_seq.device)]
        curr_h = torch.tensor(0.0).to(dist_seq.device)
        
        for _ in range(steps):
            # 1. 计算关键层阻滞项 (Denominator of Eq. 4)
            diff = curr_h - ks_h
            # 高斯函数表示关键层的影响范围
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum()
            
            # 2. 计算导数
            dh = k * (h_max - curr_h) / (1.0 + inhibition)
            
            # 3. 更新高度
            curr_h = curr_h + dh * dx
            if curr_h > h_max: curr_h = h_max
            
            h_evolution.append(curr_h)
            
        # 映射回输入的 step 距离点
        indices = (dist_seq / dx).long().clamp(max=len(h_evolution)-1)
        h_evolution_tensor = torch.stack(h_evolution)
        
        return h_evolution_tensor[indices]

    def forward(self, dist_seq):
        """
        生成对应距离序列的 Mask 序列
        """
        h_max, width, beta, lag, _, _, _ = self.get_constrained_params()
        
        # 1. 求解 ODE 得到动态高度历史
        curr_H_seq = self.solve_height_evolution(dist_seq)
        
        # 2. 批量生成几何 Mask
        xc = (dist_seq - lag).view(-1, 1, 1)
        curr_H = curr_H_seq.view(-1, 1, 1)
        
        # 相对坐标
        x_term = (self.xx.unsqueeze(0) - xc) / (width + 1e-6)
        
        # 前方拱形曲线 (Eq. 3)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta)
        
        # 定义区域 (累积 Mask 逻辑)
        # 后方 (Rear): x <= xc (保持当前高度，模拟采空区压实前的状态)
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        # 前方 (Front): x > xc (遵循拱形曲线)
        is_front = (self.xx.unsqueeze(0) > xc)
        
        # 组合高度场
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(len(dist_seq), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        # 生成软 Mask (Sigmoid)
        y_diff = height_limit - self.yy.unsqueeze(0)
        masks = torch.sigmoid(y_diff * 0.5)
        
        return masks.unsqueeze(1)

# --- 2. 主拟合逻辑 ---

def fit_subsidence_dataset_params():
    print("=== Subsidence Parameter Fitting (Equation 4: KS-Dynamics) ===")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 1. 加载数据
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    # 按 Sample ID 分组
    samples_dict = {}
    for f in all_files:
        try:
            name = os.path.basename(f)
            s_idx = name.rfind("sample") + 7
            s_id = int(name[s_idx : s_idx+4])
            if s_id not in samples_dict: samples_dict[s_id] = []
            samples_dict[s_id].append(f)
        except: continue
        
    print(f"Fitting {len(samples_dict)} samples...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted_params = {}

    # 2. 逐样本优化
    for s_id, files in tqdm(samples_dict.items()):
        # 按 Step 排序 (必须，为了 ODE 求解)
        files.sort(key=lambda x: int(os.path.basename(x).split('step_')[1].split('.')[0]))
        
        gt_list = []
        dist_list = []
        
        for f in files:
            step_id = int(os.path.basename(f).split('step_')[1].split('.')[0])
            dist = step_id * STEP_DISTANCE_M
            
            with np.load(f) as data:
                y = data['y'].astype(np.float32)
                # 形状修正
                if y.ndim == 1: y = y.reshape(IMG_SIZE, IMG_SIZE)
                # 坐标系修正 (Transpose)
                y = y.T 
                # 二值化 (提取裂隙骨架)
                y = (y > 0.1).astype(np.float32)
                
            gt_list.append(torch.from_numpy(y))
            dist_list.append(dist)
            
        if not gt_list: continue
        
        gt_tensor = torch.stack(gt_list).to(device).unsqueeze(1)
        dist_tensor = torch.tensor(dist_list).to(device).float()
        
        # 初始化拟合器
        fitter = KeyStrataFitter().to(device)
        optimizer = optim.Adam(fitter.parameters(), lr=0.1)
        
        # 优化循环
        for _ in range(100):
            optimizer.zero_grad()
            pred_masks = fitter(dist_tensor)
            loss = nn.MSELoss()(pred_masks, gt_tensor)
            loss.backward()
            optimizer.step()
            
        # 保存参数
        h, w, b, l, k, ksh, ksb = fitter.get_constrained_params()
        
        fitted_params[str(s_id)] = {
            "h_max": round(h.item(), 2),
            "width": round(w.item(), 2),
            "beta": round(b.item(), 2),
            "lag": round(l.item(), 2),
            "k_growth": round(k.item(), 4),
            "ks_heights": ksh.tolist(),
            "ks_betas": ksb.tolist()
        }
        
    # 写入 JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(fitted_params, f, indent=4)
        
    print(f"Success! Parameters saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    fit_subsidence_dataset_params()