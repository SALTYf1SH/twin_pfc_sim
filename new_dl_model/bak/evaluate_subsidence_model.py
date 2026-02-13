# -*- coding: utf-8 -*-
"""
Physics-Informed Model Evaluation Script (适用于新模型架构)

该脚本专为 'new_dl_model' 目录下的物理感知反演模型设计。
它不仅评估数据的拟合程度 (MSE/SSIM)，还评估物理规律的遵循程度 (Evo Error)。

主要功能：
1. 加载最新的双支路模型 (含 Sigmoid 输出层)。
2. 使用 SequentialFractureDataset 加载时序关联数据 (t, t-1)。
3. 计算物理一致性指标：演化误差 (Evolution Error)。
4. 执行统计假设检验 (对数正态分布, 异常值分析)。
5. 生成包含文件名和物理指标的可视化对比图。

使用方法:
# 1. 少量样本可视化 (检查效果)
python new_dl_model/evaluate_model.py --num_samples 5

# 2. 全量统计评估 (生成直方图和统计数据)
python new_dl_model/evaluate_model.py --all
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --- 依赖库检查 ---
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("WARNING: scikit-image not found. SSIM metrics will be skipped.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not found. Statistical tests will be skipped.")

# --- 0. 配置参数 ---

# 假设脚本位于 new_dl_model/ 目录下，数据在上一级目录的 final_dataset/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset") 
# 默认模型路径 (请根据实际训练输出修改)
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "trained_models_physics_informed/best_physics_model.pth")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "../trained_models_physics_informed/best_physics_model.pth")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
STATIC_FEATURES = 11
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42

# --- 1. 模型定义 (需与训练代码完全一致) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size,
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
        # 静态支路
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 动态支路
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 融合头 (包含 Sigmoid)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
            nn.Sigmoid() # 关键：训练脚本中加入了Sigmoid
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        static_out = self.static_branch(x_static)
        
        x_dynamic = x_dynamic.unsqueeze(-1)
        dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
        dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
        dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
        dynamic_out = dynamic_transformed.mean(dim=1)
        
        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

# --- 2. 数据处理 (支持 sample_step 解析) ---

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    """
    支持时序解析的数据集。
    返回: (x_t, y_t, y_prev, filename)
    """
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        self.index_map = {}
        
        # 构建索引 (sample_id, step_id) -> file_idx
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError:
                continue

    def _parse_filename(self, filepath):
        # 解析格式: ...sample_XXXX_step_XXX.npz
        filename = os.path.basename(filepath)
        
        s_pos = filename.rfind("sample")
        if s_pos == -1: raise ValueError
        s_start = s_pos + len("sample") + 1
        s_id = int(filename[s_start : s_start + 4])
        
        st_pos = filename.rfind("step")
        if st_pos == -1: raise ValueError
        st_start = st_pos + len("step") + 1
        st_id = int(filename[st_start : st_start + 3])
        
        return s_id, st_id

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        
        # 加载当前 T
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            
        # 加载 T-1 (用于评估演化误差)
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
            else:
                y_prev = y_t.clone() # 缺失回退: 假设不变
        else:
            y_prev = torch.zeros_like(y_t) # 初始步前为0

        if self.transform:
            x_t = self.transform(x_t)
            
        return x_t, y_t, y_prev, os.path.basename(curr_path)

# --- 3. 评估主逻辑 ---

def evaluate(args):
    print("========================================================")
    print("   Physics-Informed Model Evaluation (New DL Model)     ")
    print("========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    
    # 1. 数据准备
    if not os.path.exists(DATASET_DIR):
        print(f"Fatal: Dataset directory not found at {DATASET_DIR}")
        return

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print("Fatal: No .npz files found.")
        return

    # 复现训练时的划分
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    print(f"Validation Set: {len(val_files)} samples")

    # 归一化统计量 (为快速度，这里仅采样前200个训练样本计算)
    print("Calculating normalization stats...")
    temp_dataset = SequentialFractureDataset(train_files[:200])
    temp_loader = DataLoader(temp_dataset, batch_size=100)
    all_x = [x for x, _, _, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    print("Stats ready.")

    # 2. 模型加载
    # 动态获取特征维度
    with np.load(val_files[0]) as f:
        total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES
    
    model = DualBranchModel(STATIC_FEATURES, dynamic_feats, OUTPUT_HEIGHT*OUTPUT_WIDTH).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # 3. 执行评估
    val_dataset = SequentialFractureDataset(val_files, transform=transform)
    
    if args.all:
        # ---------------- 模式 A: 全量统计分析 ----------------
        print("\n--- Running Statistical Evaluation on All Samples ---")
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        all_mse = []
        all_mae = []
        all_evo = [] # 物理演化误差
        all_ssim = []
        
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_mae = nn.L1Loss(reduction='none')
        
        with torch.no_grad():
            for inputs, targets, targets_prev, _ in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets_prev = targets_prev.to(device)
                
                outputs = model(inputs)
                
                # A. 信息保真度 (MSE, MAE)
                mse_batch = criterion_mse(outputs, targets).mean(dim=1)
                all_mse.extend(mse_batch.cpu().numpy())
                
                mae_batch = criterion_mae(outputs, targets).mean(dim=1)
                all_mae.extend(mae_batch.cpu().numpy())
                
                # B. 物理一致性 (Evo Error)
                # 衡量: 预测增量 vs 真实增量
                pred_delta = outputs - targets_prev.view(outputs.shape)
                gt_delta = targets - targets_prev.view(targets.shape)
                evo_batch = criterion_mse(pred_delta, gt_delta).mean(dim=1)
                all_evo.extend(evo_batch.cpu().numpy())

                # C. 拓扑结构 (SSIM)
                if SKIMAGE_AVAILABLE:
                    out_imgs = outputs.cpu().numpy().reshape(-1, 64, 64)
                    tgt_imgs = targets.cpu().numpy().reshape(-1, 64, 64)
                    for p, t in zip(out_imgs, tgt_imgs):
                        dr = max(t.max()-t.min(), 1e-6)
                        all_ssim.append(ssim(t, p, data_range=dr))

        # 结果汇总
        mse_arr = np.array(all_mse)
        print("\n======================================================")
        print("                Statistical Summary                   ")
        print("======================================================")
        print(f"Samples Evaluated: {len(mse_arr)}")
        print(f"  [Information Fidelity]")
        print(f"    Mean MSE: {np.mean(mse_arr):.6f}")
        print(f"    Mean MAE: {np.mean(all_mae):.6f}")
        
        print(f"  [Physics Consistency]")
        print(f"    Mean Evo Error: {np.mean(all_evo):.6f}")
        print(f"    (衡量模型对动态演化增量的预测能力)")
        
        if all_ssim:
            print(f"  [Topological Structure]")
            print(f"    Mean SSIM: {np.mean(all_ssim):.6f}")

        # 绘制直方图 (支持中文显示)
        plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mse_arr, bins=100, color='#4c72b0', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title('MSE 分布直方图 (验证集)', fontsize=14)
        ax.set_xlabel('均方误差 (MSE)', fontsize=12)
        ax.set_ylabel('样本频数', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        hist_path = os.path.join(args.output_dir, "mse_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"\nSaved histogram to: {hist_path}")

        # 假设检验 (Log-Normality)
        if SCIPY_AVAILABLE:
            print("\n--- Hypothesis Testing (Log-Normal) ---")
            log_mse = np.log(mse_arr + 1e-12)
            # 采样前5000个点进行Shapiro测试(数据量太大时p值不准)
            stat, p = stats.shapiro(log_mse[:5000])
            print(f"Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p:.4g}")
            if p > 0.05:
                print(">> 结论: 数据符合对数正态分布 (Fail to reject H0)")
            else:
                print(">> 结论: 数据显著偏离对数正态分布 (Reject H0)")

    else:
        # ---------------- 模式 B: 样本可视化 ----------------
        print(f"\n--- Visualizing {args.num_samples} Random Samples ---")
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
        idx = 0
        criterion_mse = nn.MSELoss()
        
        with torch.no_grad():
            for inputs, targets, targets_prev, fnames in val_loader:
                if idx >= args.num_samples: break
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # 计算单样本指标
                mse = criterion_mse(outputs, targets).item()
                
                # Reshape & Rotate (适配地质坐标系)
                pred_img = outputs.cpu().numpy().reshape(64, 64)
                gt_img = targets.cpu().numpy().reshape(64, 64)
                
                # 逆时针旋转90度 (视PFC导出习惯而定)
                pred_img = np.rot90(pred_img)
                gt_img = np.rot90(gt_img)
                
                # 绘图
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                plt.suptitle(f"File: {fnames[0]}\nMSE: {mse:.5f}", fontsize=14)
                
                # Ground Truth
                im0 = axes[0].imshow(gt_img, cmap='jet', vmin=0, vmax=1)
                axes[0].set_title("Ground Truth (Real)")
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], fraction=0.046)
                
                # Prediction
                im1 = axes[1].imshow(pred_img, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title("Model Prediction")
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                
                # Difference
                diff = np.abs(gt_img - pred_img)
                im2 = axes[2].imshow(diff, cmap='plasma') # 使用Plasma色图突出误差
                axes[2].set_title("Absolute Difference")
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                
                save_path = os.path.join(args.output_dir, f"sample_vis_{idx}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization: {save_path}")
                
                idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Physics-Informed DL Model")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pth file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of visualization samples")
    parser.add_argument("--all", action="store_true", help="Run full statistical analysis on validation set")
    
    args = parser.parse_args()
    evaluate(args)