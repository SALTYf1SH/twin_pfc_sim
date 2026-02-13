# -*- coding: utf-8 -*-
"""
fit_stress_params.py
应力反演模型-物理参数反算脚本

功能：
针对应力数据集 (final_dataset_stress) 中的每一个工况 (Sample)，
根据其全生命周期的裂隙场演化数据 (Ground Truth)，
反向拟合第四章理论模型的最佳参数 (H_max, Width, Beta, Lag)。

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

# 获取当前脚本所在目录 (new_dl_model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径 (应力数据)
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")

# 输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, "stress_para")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "stress_physics_params.json")

# 物理场景参数 (与 PFC 模拟一致)
MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
IMG_SIZE = 64

# --- 1. 单样本参数拟合器 (PyTorch Module) ---

class SingleSampleArchFitter(nn.Module):
    """
    微型可微模型，用于拟合单个 Sample 的物理参数。
    通过最小化 (理论Arch掩膜 - 真实裂隙场) 的差异来寻找最优参数。
    """
    def __init__(self):
        super().__init__()
        # 定义待拟合参数 (初始化为第四章的经验均值)
        # 使用 nn.Parameter 使其可训练
        self.h_max_raw = nn.Parameter(torch.tensor(100.0)) 
        self.width_raw = nn.Parameter(torch.tensor(94.0))
        self.beta_raw = nn.Parameter(torch.tensor(7.5))
        self.lag_raw = nn.Parameter(torch.tensor(20.0))
        
        # 预计算坐标网格
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, IMG_SIZE), 
            torch.linspace(0, MODEL_LENGTH_M, IMG_SIZE),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def get_params(self):
        """获取受物理约束限制的参数值"""
        # 限制参数在合理物理范围内，防止优化飞出边界
        h_max = torch.clamp(self.h_max_raw, 50.0, 150.0)  # 发育高度 50~150m
        width = torch.clamp(self.width_raw, 50.0, 200.0)  # 拱宽 50~200m
        beta = torch.clamp(self.beta_raw, 1.0, 15.0)      # 形状指数 1~15
        lag = torch.clamp(self.lag_raw, 0.0, 100.0)       # 滞后距离 0~100m
        return h_max, width, beta, lag

    def forward(self, mining_dist):
        """根据当前开采距离生成理论裂隙分布概率图"""
        h_max, width, beta, lag = self.get_params()
        
        # 1. 计算理论活动拱中心 xc = d - Lag
        xc = mining_dist - lag
        
        # 2. 计算当前发育高度 H(t) (简化为 S 型增长)
        curr_H = h_max * torch.tanh(mining_dist / 100.0)
        
        # 3. 生成理论掩膜 (依据论文公式 4.3)
        x_term = (self.x_grid - xc) / (width + 1e-6)
        in_arch = (x_term.abs() <= 1.0).float()
        
        # 计算拱形边界高度 y_b
        y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * in_arch
        
        # 生成 Soft Mask (Sigmoid 平滑边界)
        mask = torch.sigmoid((y_boundary - self.y_grid) * 0.5)
        return mask

# --- 2. 主拟合逻辑 ---

def fit_stress_dataset_params():
    print("======================================================")
    print("   Stress Model - Physics Parameter Identification    ")
    print("======================================================")

    # 0. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 1. 整理数据：按 Sample ID 分组
    if not os.path.exists(DATASET_DIR):
        print(f"Fatal: Dataset not found at {DATASET_DIR}")
        return

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    samples_dict = {} 
    
    print("Grouping files by Sample ID...")
    for f in all_files:
        # 解析文件名: ...sample_XXXX_step...
        name = os.path.basename(f)
        try:
            s_idx = name.rfind("sample") + 7
            sample_id = int(name[s_idx : s_idx+4])
            
            if sample_id not in samples_dict:
                samples_dict[sample_id] = []
            samples_dict[sample_id].append(f)
        except: continue
    
    print(f"Found {len(samples_dict)} unique samples.")

    # 结果字典
    fitted_params = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start fitting on {device}...")

    # 2. 循环拟合每个 Sample
    for s_id, files in tqdm(samples_dict.items(), desc="Fitting Samples"):
        # 收集该 Sample 所有 Step 的数据
        step_data = []
        mining_dists = []
        
        for f in files:
            # 解析 Step ID 计算距离
            st_idx = os.path.basename(f).rfind("step") + 5
            step_id = int(os.path.basename(f)[st_idx : st_idx+3])
            dist = step_id * STEP_DISTANCE_M
            
            # 加载真实裂隙场 GT
            with np.load(f) as data:
                y = data['y'].astype(np.float32)
                
                # ==========================================
                # [关键修复]: Reshape 4096 -> 64x64
                # ==========================================
                if y.ndim == 1 and y.shape[0] == IMG_SIZE * IMG_SIZE:
                    y = y.reshape(IMG_SIZE, IMG_SIZE)
                
                # 二值化处理：突出裂隙区域
                y = (y > 0.1).astype(np.float32) 
            
            step_data.append(torch.from_numpy(y))
            mining_dists.append(dist)
            
        if not step_data: continue

        # Stack 后形状: [Steps, 64, 64]
        # Unsqueeze 后: [Steps, 1, 64, 64] (适配模型输入)
        gt_batch = torch.stack(step_data).to(device).unsqueeze(1) 
        dist_batch = torch.tensor(mining_dists).to(device).float()
        
        # 初始化拟合器
        fitter = SingleSampleArchFitter().to(device)
        optimizer = optim.Adam(fitter.parameters(), lr=0.5) 
        
        # 优化循环
        for _ in range(50):
            optimizer.zero_grad()
            
            # 预测每一帧的理论 Mask
            pred_masks = []
            for d in dist_batch:
                pred_masks.append(fitter(d))
            pred_batch = torch.stack(pred_masks).unsqueeze(1)
            
            # 计算 Loss
            loss = nn.MSELoss()(pred_batch, gt_batch)
            
            loss.backward()
            optimizer.step()
            
        # 获取最佳参数
        h, w, b, l = fitter.get_params()
        
        # 保存结果
        fitted_params[str(s_id)] = {
            "h_max": round(h.item(), 2),
            "width": round(w.item(), 2),
            "beta":  round(b.item(), 2),
            "lag":   round(l.item(), 2)
        }
        
    # 3. 保存结果
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(fitted_params, f, indent=4)
    
    print("-" * 50)
    print(f"Fitting Complete!")
    print(f"Parameters saved to: {OUTPUT_JSON}")
    print("-" * 50)

if __name__ == "__main__":
    fit_stress_dataset_params()