# -*- coding: utf-8 -*-
"""
Batch Evaluation Script (Fixed: True Evolution Error)

关键改进：
1. 计算 'True_Evo_Error'：
   通过 model(x_t) - model(x_t-1) 计算模型自身的动力学增量，
   并与真实增量 (y_t - y_t-1) 对比。
   这是验证模型是否学会"演化规律"的唯一标准。

2. 预期结果：
   Full Dual-Branch 模型在 True_Evo_Error 上应显著优于 No Evo 模型。
"""

import os
import glob
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# --- 0. 基础配置 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../final_dataset_stress")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models_stress_ablation")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results_stress")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_summary_fixed.csv")
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "stress_para/stress_physics_params.json")

STATIC_FEATURES = 17
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
TRAIN_VAL_SPLIT_RATIO = 0.9
RANDOM_SEED = 42
STEP_DISTANCE_M = 10.0
BATCH_SIZE = 64
MODEL_LENGTH_M = 500.0    
MODEL_HEIGHT_M = 150.0    

EXPERIMENTS = [
    {"name": "Full Dual-Branch", "ablation": "full",         "branch": "dual"},
    {"name": "Static Only",      "ablation": "full",         "branch": "static_only"},
    {"name": "Dynamic Only",     "ablation": "full",         "branch": "dynamic_only"},
    {"name": "Baseline (MSE)",   "ablation": "baseline",     "branch": "dual"},
    {"name": "No SSIM",          "ablation": "no_ssim",      "branch": "dual"},
    {"name": "No Arch Prior",    "ablation": "no_arch",      "branch": "dual"},
    {"name": "No Evolution",     "ablation": "no_evo",       "branch": "dual"},
    {"name": "No TV",            "ablation": "no_tv",        "branch": "dual"},
]

# --- 1. 辅助类 ---

class ActivityArchMaskGenerator(nn.Module):
    def __init__(self, output_size=64):
        super(ActivityArchMaskGenerator, self).__init__()
        self.H, self.W = output_size, output_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, MODEL_HEIGHT_M, self.H), 
            torch.linspace(0, MODEL_LENGTH_M, self.W),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def forward(self, mining_distances, phys_params):
        batch_size = mining_distances.size(0)
        masks = []
        for i in range(batch_size):
            d = mining_distances[i]
            h_max = phys_params[i, 0]; w_arch = phys_params[i, 1]
            beta = phys_params[i, 2]; lag = phys_params[i, 3]
            
            xc = d - lag 
            curr_H = h_max * torch.tanh(d / 100.0)
            x_term = (self.x_grid - xc) / (w_arch + 1e-6)
            y_boundary = curr_H * torch.pow((1 - x_term.pow(2)).clamp(min=0), beta) * (x_term.abs() <= 1.0).float()
            spatial_mask = torch.sigmoid((y_boundary - self.y_grid) * 0.5) 
            masks.append(spatial_mask)
        return torch.stack(masks).unsqueeze(1)

# --- 2. 模型定义 (保持一致) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)

class DualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(DualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode 
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        if self.branch_mode in ['dual', 'static_only']: static_out = self.static_branch(x_static)
        else: static_out = torch.zeros(x.size(0), 32, device=x.device)

        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic = x_dynamic.unsqueeze(-1)
            dynamic_embedded = self.dynamic_embedder(x_dynamic) * math.sqrt(self.d_model)
            dynamic_pos_encoded = self.pos_encoder(dynamic_embedded)
            dynamic_transformed = self.transformer_encoder(dynamic_pos_encoded)
            dynamic_out = dynamic_transformed.mean(dim=1)
        else: dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        return self.fusion_head(fused)

# --- 3. 数据加载 (新增 x_prev) ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialStressDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}
        self.index_map = {}
        for idx, filepath in enumerate(self.file_list):
            try:
                s_id, st_id = self._parse_filename(filepath)
                self.index_map[(s_id, st_id)] = idx
            except ValueError: continue

    def _parse_filename(self, filepath):
        filename = os.path.basename(filepath)
        s_pos = filename.rfind("sample"); s_start = s_pos + 7; s_id = int(filename[s_start : s_start + 4])
        st_pos = filename.rfind("step"); st_start = st_pos + 5; st_id = int(filename[st_start : st_start + 3])
        return s_id, st_id

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        s_id, st_id = self._parse_filename(curr_path)
        mining_dist = st_id * STEP_DISTANCE_M
        
        # Load Current T
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
        
        # Load Previous T-1
        # [NEW] 我们需要 x_prev 来让模型预测 pred_prev，从而计算真正的动力学误差
        if st_id > 1:
            prev_idx = self.index_map.get((s_id, st_id - 1))
            if prev_idx is not None:
                with np.load(self.file_list[prev_idx]) as data:
                    y_prev = torch.from_numpy(data['y'].astype(np.float32))
                    x_prev = torch.from_numpy(data['x'].astype(np.float32))
            else:
                y_prev = y_t.clone()
                x_prev = x_t.clone()
        else:
            y_prev = torch.zeros_like(y_t)
            x_prev = torch.zeros_like(x_t) # 初始状态输入为0

        # 获取参数
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)
        
        if self.transform:
            x_t = self.transform(x_t)
            # 如果不是零张量(step>1)，也需要标准化 x_prev
            if st_id > 1:
                x_prev = self.transform(x_prev)
        
        return x_t, x_prev, y_t, y_prev, np.float32(mining_dist), phys_vec

# --- 4. 核心评估函数 ---

def evaluate_single_experiment(exp_config, val_loader, device, feature_dims):
    ablation = exp_config['ablation']
    branch_mode = exp_config['branch']
    model_name = f"best_stress_{ablation}_{branch_mode}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    print(f"Evaluating: {exp_config['name']} ...")
    if not os.path.exists(model_path):
        print(f"  [Error] Model not found: {model_name}"); return None

    model = DualBranchModel(
        static_size=feature_dims['static'], dynamic_size=feature_dims['dynamic'],
        output_size=OUTPUT_HEIGHT*OUTPUT_WIDTH, branch_mode=branch_mode
    ).to(device)
    
    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except: print("  [Error] Load failed"); return None
    
    model.eval()
    all_mse, all_ssim, all_pcr, all_true_evo = [], [], [], []
    criterion_mse = nn.MSELoss(reduction='none')
    mask_generator = ActivityArchMaskGenerator(output_size=64).to(device)
    
    with torch.no_grad():
        for x_t, x_prev, y_t, y_prev, dists, phys_params in val_loader:
            x_t = x_t.to(device); x_prev = x_prev.to(device)
            y_t = y_t.to(device); y_prev = y_prev.to(device)
            dists = dists.to(device); phys_params = phys_params.to(device)
            
            # 1. 双次推理 (Twin Inference)
            pred_t = model(x_t)
            pred_prev = model(x_prev) # 模型自己对上一步的预测
            
            # 2. 基础指标
            all_mse.extend(criterion_mse(pred_t, y_t).mean(dim=1).cpu().numpy())
            
            # 3. 真实演化误差 (True Evo Error)
            # 比较: (模型预测的增量) vs (真实物理增量)
            # Delta_Pred = Pred_t - Pred_prev
            # Delta_GT   = y_t - y_prev
            # Loss = || Delta_Pred - Delta_GT ||
            model_delta = pred_t - pred_prev
            true_delta = y_t - y_prev
            evo_err = criterion_mse(model_delta, true_delta).mean(dim=1)
            all_true_evo.extend(evo_err.cpu().numpy())
            
            # 4. 形状处理 & SSIM
            pred_imgs = pred_t.view(-1, 1, 64, 64)
            pred_clamped = torch.clamp(pred_imgs, 0.0, 1.0)
            
            if SKIMAGE_AVAILABLE:
                tgt_imgs = y_t.view(-1, 1, 64, 64).cpu().numpy()
                np_preds = pred_clamped.cpu().numpy()
                for p, t in zip(np_preds, tgt_imgs):
                    dr = max(t.max()-t.min(), 1e-6)
                    all_ssim.append(ssim(t.squeeze(), p.squeeze(), data_range=dr))
            
            # 5. PCR
            arch_masks = mask_generator(dists, phys_params)
            masked = (pred_clamped * arch_masks).sum(dim=(1,2,3))
            total = pred_clamped.sum(dim=(1,2,3)) + 1e-6
            all_pcr.extend((masked/total).cpu().numpy())
            
    results = {
        "Model": exp_config['name'],
        "MSE": np.mean(all_mse),
        "SSIM": np.mean(all_ssim) if all_ssim else 0.0,
        "True_Evo_Error": np.mean(all_true_evo), # 关键新指标
        "PCR": np.mean(all_pcr)
    }
    print(f"  -> MSE:{results['MSE']:.5f}, Evo:{results['True_Evo_Error']:.5f}, PCR:{results['PCR']:.4f}")
    return results

# --- 6. 主程序 ---

def main():
    print("========================================================")
    print("   Batch Ablation Evaluation (Stress) - FIXED EVO       ")
    print("========================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found"); return
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_files)
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    print(f"Validation Set: {len(val_files)} samples")
    
    # Stats
    print("Calculating stats...")
    temp_loader = DataLoader(SequentialStressDataset(train_files[:200], ""), batch_size=100)
    all_x = [x for x, _, _, _, _, _ in temp_loader] # 解包6个返回值
    x_tensor = torch.cat(all_x, dim=0)
    mean = x_tensor.mean(dim=0); std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    
    val_dataset = SequentialStressDataset(val_files, PARAMS_JSON_PATH, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    feature_dims = {'static': STATIC_FEATURES, 'dynamic': total_feats - STATIC_FEATURES}
    
    all_results = []
    for exp in EXPERIMENTS:
        res = evaluate_single_experiment(exp, val_loader, device, feature_dims)
        if res: all_results.append(res)
            
    print("\nWriting summary to CSV...")
    fieldnames = ["Model", "MSE", "SSIM", "True_Evo_Error", "PCR"]
    
    with open(SUMMARY_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(res)
            
    print(f"Done! Summary saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()