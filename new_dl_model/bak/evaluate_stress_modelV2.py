# -*- coding: utf-8 -*-
"""
Physics-Informed STRESS Model Evaluation Script

该脚本专为 '应力-裂隙场' 物理感知反演模型设计。
配置已适配:
1. 应力数据集 (17个静态特征)。
2. 线性输出模型 (无 Sigmoid)。
3. 物理演化误差评估 (Evo Error).

用法示例:
# 1. 少量样本可视化
python new_dl_model/evaluate_stress_model.py --num_samples 5

# 2. 全量统计评估
python new_dl_model/evaluate_stress_model.py --all
"""

import os
import glob
import json
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 数据集路径 (应力数据)
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress") 

# 2. 物理参数路径 (保持 Dataset 结构一致)
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

# 3. 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "trained_models_stress_physics/best_stress_physics_model.pth")

# 4. 输出路径
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results_stress")

# 5. 模型特征参数
STATIC_FEATURES = 17      # 应力模型为 17 个静态参数
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42
STEP_DISTANCE_M = 10.0

# --- 1. 模型定义 (线性输出，无 Sigmoid) ---

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
        
        # 融合头 (无 Sigmoid)
        fusion_input_size = 32 + d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
            # No Sigmoid: 对应训练脚本的修改
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

# --- 2. 数据处理 ---

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
    """
    适配应力模型的数据集。
    加载物理参数以保持接口一致。
    """
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        
        # 加载物理参数 (保持与训练一致)
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f:
                self.physics_params = json.load(f)
        else:
            # 评估时如果找不到参数文件，给默认值
            self.physics_params = {} 

        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError:
                continue

    def _parse_filename(self, filepath):
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
            
        # 加载 T-1
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
            else:
                y_prev = y_t.clone()
        else:
            y_prev = torch.zeros_like(y_t)

        # 获取参数 (Placeholder for evaluation)
        params = self.physics_params.get(str(s_id), {
            "h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0
        })
        phys_vec = torch.tensor([
            params['h_max'], params['width'], params['beta'], params['lag']
        ], dtype=torch.float32)

        if self.transform:
            x_t = self.transform(x_t)
            
        return x_t, y_t, y_prev, os.path.basename(curr_path)

# --- 3. 评估主逻辑 ---

def evaluate(args):
    print("========================================================")
    print("   Physics-Informed STRESS Model Evaluation             ")
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

    # 复现训练划分
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    
    print(f"Validation Set: {len(val_files)} samples")

    # 归一化统计量
    print("Calculating normalization stats...")
    # 使用空的 json path 加载器计算均值
    temp_dataset = SequentialStressDataset(train_files[:200], "") 
    temp_loader = DataLoader(temp_dataset, batch_size=100)
    all_x = [x for x, _, _, _ in temp_loader]
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    print("Stats ready.")

    # 2. 模型加载
    with np.load(val_files[0]) as f:
        total_feats = f['x'].shape[0]
    dynamic_feats = total_feats - STATIC_FEATURES # 17
    
    print(f"Model Config: Static={STATIC_FEATURES}, Dynamic={dynamic_feats}")
    
    model = DualBranchModel(STATIC_FEATURES, dynamic_feats, OUTPUT_HEIGHT*OUTPUT_WIDTH).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # 3. 执行评估
    val_dataset = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform=transform)
    
    if args.all:
        # ---------------- 模式 A: 全量统计分析 ----------------
        print("\n--- Running Statistical Evaluation on All Samples ---")
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        all_mse, all_mae, all_ssim, all_evo = [], [], [], []
        
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_mae = nn.L1Loss(reduction='none')
        
        with torch.no_grad():
            for inputs, targets, targets_prev, _ in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets_prev = targets_prev.to(device)
                
                outputs = model(inputs)
                
                # A. 信息保真度
                mse_batch = criterion_mse(outputs, targets).mean(dim=1)
                all_mse.extend(mse_batch.cpu().numpy())
                
                mae_batch = criterion_mae(outputs, targets).mean(dim=1)
                all_mae.extend(mae_batch.cpu().numpy())
                
                # B. 物理一致性 (Evo Error)
                pred_delta = outputs - targets_prev.view(outputs.shape)
                gt_delta = targets - targets_prev.view(targets.shape)
                evo_batch = criterion_mse(pred_delta, gt_delta).mean(dim=1)
                all_evo.extend(evo_batch.cpu().numpy())

                # C. 拓扑结构 (需先 Clip 到 0-1)
                if SKIMAGE_AVAILABLE:
                    # Linear output -> Clip -> SSIM
                    pred_imgs = torch.clamp(outputs, 0.0, 1.0).cpu().numpy().reshape(-1, 64, 64)
                    tgt_imgs = targets.cpu().numpy().reshape(-1, 64, 64)
                    
                    for p, t in zip(pred_imgs, tgt_imgs):
                        dr = max(t.max()-t.min(), 1e-6)
                        all_ssim.append(ssim(t, p, data_range=dr))

        # 结果汇总
        mse_arr = np.array(all_mse)
        print("\n======================================================")
        print("             Statistical Summary (Stress)             ")
        print("======================================================")
        print(f"Samples Evaluated: {len(mse_arr)}")
        print(f"  [Information Fidelity]")
        print(f"    Mean MSE: {np.mean(mse_arr):.6f}")
        print(f"    Mean MAE: {np.mean(all_mae):.6f}")
        
        print(f"  [Physics Consistency]")
        print(f"    Mean Evo Error: {np.mean(all_evo):.6f}")
        
        if all_ssim:
            print(f"  [Topological Structure]")
            print(f"    Mean SSIM: {np.mean(all_ssim):.6f}")

        # 绘制直方图
        plt.rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # 使用粉色调区分应力模型
        ax.hist(mse_arr, bins=100, color='#e377c2', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title('MSE Distribution (Stress Model Validation)', fontsize=14)
        ax.set_xlabel('MSE', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        hist_path = os.path.join(args.output_dir, "stress_mse_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"\nSaved histogram to: {hist_path}")

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
                
                mse = criterion_mse(outputs, targets).item()
                
                # Reshape & Rotate & Clip
                # 关键：预测值可能超出 [0,1]，显示前先 Clip，防止噪点
                pred_raw = outputs.cpu().numpy().reshape(64, 64)
                pred_img = np.clip(pred_raw, 0.0, 1.0)
                
                gt_img = targets.cpu().numpy().reshape(64, 64)
                
                # 旋转
                pred_img = np.rot90(pred_img)
                gt_img = np.rot90(gt_img)
                
                # 绘图
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                plt.suptitle(f"Stress-Based | File: {fnames[0]}\nMSE: {mse:.5f}", fontsize=14)
                
                im0 = axes[0].imshow(gt_img, cmap='jet', vmin=0, vmax=1)
                axes[0].set_title("Ground Truth")
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], fraction=0.046)
                
                im1 = axes[1].imshow(pred_img, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title("Prediction (Clipped)")
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                
                diff = np.abs(gt_img - pred_img)
                im2 = axes[2].imshow(diff, cmap='plasma')
                axes[2].set_title("Difference")
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                
                save_path = os.path.join(args.output_dir, f"stress_sample_{idx}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization: {save_path}")
                
                idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Physics-Informed Stress Model")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to .pth file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of visualization samples")
    parser.add_argument("--all", action="store_true", help="Run full statistical analysis")
    
    args = parser.parse_args()
    evaluate(args)