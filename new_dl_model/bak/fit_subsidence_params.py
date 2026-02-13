# -*- coding: utf-8 -*-
"""
fit_subsidence_params.py (FIXED: Shape Mismatch)
沉降反演模型-物理参数反算脚本
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
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "subsidence_para")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "subsidence_physics_params.json")

MODEL_LENGTH_M = 500.0
MODEL_HEIGHT_M = 150.0
STEP_DISTANCE_M = 10.0
IMG_SIZE = 64

# --- 1. 单样本参数拟合器 (不变) ---

class SingleSampleArchFitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_max_raw = nn.Parameter(torch.tensor(100.0)) 
        self.width_raw = nn.Parameter(torch.tensor(94.0))
        self.beta_raw = nn.Parameter(torch.tensor(7.5))
        self.lag_raw = nn.Parameter(torch.tensor(20.0))
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, IMG_SIZE), 
            torch.linspace(0, MODEL_LENGTH_M, IMG_SIZE),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def get_params(self):
        h_max = torch.clamp(self.h_max_raw, 50.0, 150.0)
        width = torch.clamp(self.width_raw, 50.0, 200.0)
        beta = torch.clamp(self.beta_raw, 1.0, 15.0)
        lag = torch.clamp(self.lag_raw, 0.0, 100.0)
        return h_max, width, beta, lag

    def forward(self, mining_dist):
        h_max, width, beta, lag = self.get_params()
        xc = mining_dist - lag
        curr_H = h_max * torch.tanh(mining_dist / 100.0)
        x_term = (self.x_grid - xc) / (width + 1e-6)
        in_arch = (x_term.abs() <= 1.0).float()
        y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * in_arch
        mask = torch.sigmoid((y_boundary - self.y_grid) * 0.5)
        return mask

# --- 2. 主拟合逻辑 (修复了数据加载部分) ---

def fit_dataset_params():
    print("======================================================")
    print("   Subsidence Model - Physics Parameter Identification")
    print("======================================================")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(DATASET_DIR):
        print(f"Fatal: Dataset not found at {DATASET_DIR}")
        return

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    samples_dict = {} 
    
    print("Grouping files by Sample ID...")
    for f in all_files:
        name = os.path.basename(f)
        try:
            s_idx = name.rfind("sample") + 7
            sample_id = int(name[s_idx : s_idx+4])
            if sample_id not in samples_dict:
                samples_dict[sample_id] = []
            samples_dict[sample_id].append(f)
        except: continue
    
    print(f"Found {len(samples_dict)} unique samples.")

    fitted_params = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start fitting on {device}...")

    for s_id, files in tqdm(samples_dict.items(), desc="Fitting Samples"):
        step_data = []
        mining_dists = []
        
        for f in files:
            # 解析 Step ID
            st_idx = os.path.basename(f).rfind("step") + 5
            step_id = int(os.path.basename(f)[st_idx : st_idx+3])
            dist = step_id * STEP_DISTANCE_M
            
            with np.load(f) as data:
                y = data['y'].astype(np.float32)
                
                # ==========================================
                # [CRITICAL FIX]: Reshape 4096 -> 64x64
                # ==========================================
                if y.ndim == 1 and y.shape[0] == IMG_SIZE * IMG_SIZE:
                    y = y.reshape(IMG_SIZE, IMG_SIZE)
                
                # 二值化
                y = (y > 0.1).astype(np.float32) 
            
            step_data.append(torch.from_numpy(y))
            mining_dists.append(dist)
            
        if not step_data: continue

        # 现在 Stack 后的形状是 [Steps, 64, 64]
        # Unsqueeze 后变为 [Steps, 1, 64, 64] -> 与 pred_batch 匹配
        gt_batch = torch.stack(step_data).to(device).unsqueeze(1) 
        dist_batch = torch.tensor(mining_dists).to(device).float()
        
        fitter = SingleSampleArchFitter().to(device)
        optimizer = optim.Adam(fitter.parameters(), lr=0.5) 
        
        for _ in range(50):
            optimizer.zero_grad()
            
            pred_masks = []
            for d in dist_batch:
                pred_masks.append(fitter(d))
            pred_batch = torch.stack(pred_masks).unsqueeze(1)
            
            loss = nn.MSELoss()(pred_batch, gt_batch)
            
            loss.backward()
            optimizer.step()
            
        h, w, b, l = fitter.get_params()
        
        fitted_params[str(s_id)] = {
            "h_max": round(h.item(), 2),
            "width": round(w.item(), 2),
            "beta":  round(b.item(), 2),
            "lag":   round(l.item(), 2)
        }
        
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(fitted_params, f, indent=4)
    
    print(f"Fitting Complete! Parameters saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    fit_dataset_params()