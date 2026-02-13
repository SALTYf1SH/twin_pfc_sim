# -*- coding: utf-8 -*-
"""
fit_stress_params_ks.py (Key Strata Aware Version)
应力反演模型-物理参数反算脚本 (含关键层动力学)

功能：
基于离散元仿真数据 (GT)，反演控制裂隙发育的动力学参数。
核心差异：使用微分方程求解器替代简单的 tanh 函数，以识别关键层位置和强度。

反演参数包括：
1. 几何参数: H_max, Width, Shape_Beta, Lag
2. 动力学参数: Growth_Rate (k)
3. 关键层参数: KS_Height (层位), KS_Strength (阻滞系数 beta)

输出：
stress_para/stress_physics_params.json
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
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")
OUTPUT_DIR = os.path.join(BASE_DIR, "stress_para")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "stress_physics_params.json")

# 物理场景常数
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
IMG_SIZE = 64

# 关键层预设数量 (假设有2层主要关键层)
NUM_KEY_STRATA = 2 

# --- 1. 可微物理模型 (Differentiable Physics Model) ---

class KeyStrataFitter(nn.Module):
    """
    内嵌 ODE 求解器的拟合模型。
    完全遵循论文公式 (3) 和 (4)。
    """
    def __init__(self):
        super().__init__()
        
        # --- A. 基础几何参数 ---
        # H_max: 理论最大高度 (50~150m)
        self.h_max = nn.Parameter(torch.tensor(100.0))
        # Width: 拱宽 (50~120m)
        self.width = nn.Parameter(torch.tensor(94.0))
        # Shape Beta: 拱形指数 (1.0~8.0)
        self.shape_beta = nn.Parameter(torch.tensor(3.0))
        # Lag: 滞后距离 (10~50m)
        self.lag = nn.Parameter(torch.tensor(20.0))
        
        # --- B. 动力学参数 (Eq. 4) ---
        # k: 基础生长速率 (0.01 ~ 0.1)
        self.k_growth = nn.Parameter(torch.tensor(0.02))
        
        # --- C. 关键层参数 (Key Strata) ---
        # KS Heights: 关键层高度 (初始化在 30m 和 80m)
        self.ks_heights = nn.Parameter(torch.tensor([35.0, 85.0]))
        # KS Strengths: 阻滞系数 beta_i (初始化为 5.0)
        self.ks_betas = nn.Parameter(torch.tensor([5.0, 8.0]))
        # KS Sigma: 影响范围 (固定为 5.0m，避免过拟合，也可以设为可训练)
        self.ks_sigma = 5.0 

        # 坐标网格缓存
        y_vals = torch.linspace(0, MODEL_HEIGHT_M, IMG_SIZE)
        x_vals = torch.linspace(0, MODEL_LENGTH_M, IMG_SIZE)
        self.y_grid, self.x_grid = torch.meshgrid(y_vals, x_vals, indexing='ij')
        self.register_buffer('yy', self.y_grid)
        self.register_buffer('xx', self.x_grid)

    def get_constrained_params(self):
        """获取受物理约束的参数（防止优化溢出）"""
        h_max = torch.clamp(self.h_max, 50.0, 145.0)
        width = torch.clamp(self.width, 40.0, 150.0)
        beta = torch.clamp(self.shape_beta, 1.5, 10.0)
        lag = torch.clamp(self.lag, 0.0, 60.0)
        k = torch.clamp(self.k_growth, 0.005, 0.1)
        
        # 关键层高度限制在模型范围内
        ks_h = torch.clamp(self.ks_heights, 10.0, 140.0)
        # 阻滞系数必须非负
        ks_b = torch.relu(self.ks_betas)
        
        return h_max, width, beta, lag, k, ks_h, ks_b

    def solve_height_evolution(self, dist_seq):
        """
        [核心] 求解微分方程 Eq. 4
        dist_seq: 一个 Sample 的所有推进步距离 [d1, d2, ..., dn]
        Return: 对应每一步的高度 H_seq
        """
        h_max, _, _, _, k, ks_h, ks_b = self.get_constrained_params()
        
        # 转换为张量计算
        current_H = 0.0
        h_history = []
        
        # 积分步长 (米)
        dx = 1.0 
        
        # 为了支持反向传播，我们需要模拟从 0 到 max_dist 的过程
        max_dist = dist_seq[-1]
        steps = int(max_dist.item() / dx)
        
        # 记录每米的 H 值
        h_evolution = [torch.tensor(0.0).to(dist_seq.device)]
        
        # 欧拉积分循环 (PyTorch 循环虽然慢，但支持自动微分)
        curr_h_tensor = torch.tensor(0.0).to(dist_seq.device)
        
        for _ in range(steps):
            # 1. 计算分母阻滞项: Sum(beta * exp(...))
            # ks_h: [2], curr_h: scalar
            diff = curr_h_tensor - ks_h
            inhibition = (ks_b * torch.exp(-(diff**2) / (2 * self.ks_sigma**2))).sum()
            
            # 2. 计算导数 dH/dx
            # Eq. 4: dH = k * (Hmax - H) / (1 + Sum(...))
            dh = k * (h_max - curr_h_tensor) / (1.0 + inhibition)
            
            # 3. 更新
            curr_h_tensor = curr_h_tensor + dh * dx
            
            # 物理截断
            if curr_h_tensor > h_max: curr_h_tensor = h_max
            
            h_evolution.append(curr_h_tensor)
            
        # 将 discrete steps 映射回 input dist_seq
        # dist_seq (e.g. 10, 20...) -> index (10, 20...)
        indices = (dist_seq / dx).long().clamp(max=len(h_evolution)-1)
        h_evolution_tensor = torch.stack(h_evolution)
        
        return h_evolution_tensor[indices]

    def forward(self, dist_seq):
        """
        Args:
            dist_seq: [Batch] 推进距离序列
        Returns:
            masks: [Batch, 1, 64, 64] 预测的掩膜序列
        """
        h_max, width, beta, lag, _, _, _ = self.get_constrained_params()
        
        # 1. 求解 ODE 得到每一步的高度 H(t)
        # 这是一个包含整个动力学历史的向量
        curr_H_seq = self.solve_height_evolution(dist_seq)
        
        # 2. 批量生成 Mask (利用 Broadcasting)
        # xc: [Batch, 1, 1]
        xc = (dist_seq - lag).view(-1, 1, 1)
        curr_H = curr_H_seq.view(-1, 1, 1)
        
        # 相对坐标 X
        x_term = (self.xx.unsqueeze(0) - xc) / (width + 1e-6)
        
        # 前方拱形曲线 (Eq. 3)
        arch_curve = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta)
        
        # 定义区域 (Cumulative Logic)
        # Rear: x <= xc
        is_rear = (self.xx.unsqueeze(0) <= xc) & (self.xx.unsqueeze(0) >= 0)
        # Front: x > xc
        is_front = (self.xx.unsqueeze(0) > xc)
        
        # 组合高度场
        # Rear: 平顶 (curr_H)
        # Front: 曲线 (arch_curve)
        height_limit = torch.zeros_like(self.xx.unsqueeze(0).expand(len(dist_seq), -1, -1))
        height_limit = torch.where(is_rear, curr_H, height_limit)
        height_limit = torch.where(is_front, arch_curve, height_limit)
        
        # Sigmoid Soft Mask
        y_diff = height_limit - self.yy.unsqueeze(0)
        masks = torch.sigmoid(y_diff * 0.5)
        
        return masks.unsqueeze(1)

# --- 2. 主拟合逻辑 ---

def fit_stress_dataset_params():
    print("=== Physics Parameter Fitting (Equation 4: KS-Dynamics) ===")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 1. 加载数据
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    # 按 Sample 分组
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

    # 2. 逐个 Sample 拟合
    for s_id, files in tqdm(samples_dict.items()):
        # 收集序列数据
        # 必须按 Step 排序，因为 ODE 需要时间顺序
        files.sort(key=lambda x: int(os.path.basename(x).split('step_')[1].split('.')[0]))
        
        gt_list = []
        dist_list = []
        
        for f in files:
            step_id = int(os.path.basename(f).split('step_')[1].split('.')[0])
            dist = step_id * STEP_DISTANCE_M
            
            with np.load(f) as data:
                y = data['y'].astype(np.float32)
                if y.ndim == 1: y = y.reshape(IMG_SIZE, IMG_SIZE)
                # 转置修正 (根据之前的 Debug)
                y = y.T 
                # 二值化
                y = (y > 0.1).astype(np.float32)
                
            gt_list.append(torch.from_numpy(y))
            dist_list.append(dist)
            
        if not gt_list: continue
        
        gt_tensor = torch.stack(gt_list).to(device).unsqueeze(1) # [T, 1, 64, 64]
        dist_tensor = torch.tensor(dist_list).to(device).float()
        
        # 初始化模型
        fitter = KeyStrataFitter().to(device)
        optimizer = optim.Adam(fitter.parameters(), lr=0.1) # 较高的 LR
        
        # 优化循环
        # 增加 Epochs，因为 ODE 求解更复杂
        for _ in range(100):
            optimizer.zero_grad()
            pred_masks = fitter(dist_tensor)
            loss = nn.MSELoss()(pred_masks, gt_tensor)
            loss.backward()
            optimizer.step()
            
        # 提取参数
        h, w, b, l, k, ksh, ksb = fitter.get_constrained_params()
        
        fitted_params[str(s_id)] = {
            "h_max": round(h.item(), 2),
            "width": round(w.item(), 2),
            "beta": round(b.item(), 2),
            "lag": round(l.item(), 2),
            # 新增动力学参数
            "k_growth": round(k.item(), 4),
            "ks_heights": ksh.tolist(),
            "ks_betas": ksb.tolist()
        }
        
    # 保存
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(fitted_params, f, indent=4)
        
    print(f"Saved params with KS dynamics to: {OUTPUT_JSON}")

if __name__ == "__main__":
    fit_stress_dataset_params()