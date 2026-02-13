import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import math

# [Mamba] Import
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it to use Mamba architecture.")
    Mamba = None

# --- Configuration ---
# SCI Formatting
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # Standard academic font size
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['svg.fonttype'] = 'none' # Ensure text is editable in SVG
# Max width usually ~17-18cm for full page width in journals. 
# 1 inch = 2.54 cm. 17 cm = 6.7 inches.
FIG_WIDTH_CM = 17 
FIG_WIDTH_INCH = FIG_WIDTH_CM / 2.54

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "../robustness_results_subsidence")
# PHYSICS_DIR = os.path.join(BASE_DIR, "../trained_models_subsidence_physics_mamba")
# if not os.path.exists(MODEL_DIR) and os.path.exists(PHYSICS_DIR):
#     MODEL_DIR = PHYSICS_DIR

PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

# PHYSICS_DIR = os.path.join(BASE_DIR, "../trained_models_subsidence_physics_mamba")
# if not os.path.exists(MODEL_DIR) and os.path.exists(PHYSICS_DIR):
#     MODEL_DIR = PHYSICS_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "../visualization_results_subsidence")

# Paths for Baselines (Adjusted based on previous analysis)
BASELINE_DIR = os.path.join(BASE_DIR, "../robustness_results_subsidence_baselines")
TRANSFORMER_DIR = os.path.join(BASE_DIR, "../robustness_results_subsidence_baselines") # Assuming Transformer is also here
# PHYSICS_MODEL_DIR = os.path.join(BASE_DIR, "../../new_dl_model/trained_models_subsidence_physics")

STATIC_FEATURES = 11
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
IMG_SIZE = 64
OUTPUT_FEATURES = OUTPUT_HEIGHT * OUTPUT_WIDTH
STEP_DISTANCE_M = 10.0
MODEL_HEIGHT_M = 150.0
MODEL_LENGTH_M = 500.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (Self-Contained) ---

class DualBranchMambaModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, branch_mode='dual',
                 d_model=128, n_layers=2, dropout=0.1): 
        super(DualBranchMambaModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        self.branch_mode = branch_mode
        self.static_branch = nn.Sequential(
            nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.dynamic_embedder = nn.Linear(1, d_model)
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        
        fusion_input_size = 32 + d_model 
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size)
        )

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]
        x_dynamic = x[:, self.static_feature_size:]
        
        if self.branch_mode in ['dual', 'static_only']:
            static_out = self.static_branch(x_static)
        else:
            static_out = torch.zeros(x.size(0), 32, device=x.device)
        
        if self.branch_mode in ['dual', 'dynamic_only']:
            x_dynamic_seq = x_dynamic.unsqueeze(-1)
            x_mamba = self.dynamic_embedder(x_dynamic_seq)
            if hasattr(self, 'mamba_layers'):
                for layer, norm in zip(self.mamba_layers, self.norms):
                    residual = x_mamba
                    x_mamba = norm(x_mamba)
                    x_mamba = layer(x_mamba)
                    x_mamba = residual + x_mamba
                dynamic_out = x_mamba.mean(dim=1)
            else:
                dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        output = self.fusion_head(fused)
        return output

class TransformerDualBranchModel(nn.Module):
    def __init__(self, static_size, dynamic_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerDualBranchModel, self).__init__()
        self.static_feature_size = static_size
        self.d_model = d_model
        
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

        self.static_branch = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU())
        self.dynamic_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=dynamic_size)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fusion_head = nn.Sequential(nn.Linear(32 + d_model, 1024), nn.ReLU(), nn.Dropout(dropout), nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, output_size))

    def forward(self, x):
        x_static = x[:, :self.static_feature_size]; x_dynamic = x[:, self.static_feature_size:]
        static_out = self.static_branch(x_static)
        dynamic_embedded = self.dynamic_embedder(x_dynamic.unsqueeze(-1)) * math.sqrt(self.d_model)
        dynamic_out = self.transformer_encoder(self.pos_encoder(dynamic_embedded)).mean(dim=1)
        return self.fusion_head(torch.cat((static_out, dynamic_out), dim=1))

class DeepCNNBaseline(nn.Module):
    def __init__(self, input_size, output_size=4096):
        super(DeepCNNBaseline, self).__init__()
        self.init_channels = 128; self.init_size = 8
        self.fc = nn.Linear(input_size, self.init_channels * self.init_size * self.init_size)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    def forward(self, x):
        x = self.fc(x).view(-1, self.init_channels, self.init_size, self.init_size)
        return self.conv_blocks(x).view(x.size(0), -1)

class BiLSTMBaseline(nn.Module):
    def __init__(self, dynamic_len, static_len, output_size=4096):
        super(BiLSTMBaseline, self).__init__()
        self.static_len = static_len; self.dynamic_len = dynamic_len
        self.lstm = nn.LSTM(1 + static_len, 256, 2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, output_size))
    def forward(self, x):
        x_static = x[:, :self.static_len]; x_dynamic = x[:, self.static_len:]
        seq_input = torch.cat([x_dynamic.unsqueeze(-1), x_static.unsqueeze(1).repeat(1, self.dynamic_len, 1)], dim=2)
        lstm_out, _ = self.lstm(seq_input)
        return self.decoder(torch.mean(lstm_out, dim=1))

# --- Data Loading ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialFractureDataset(Dataset):
    def __init__(self, npz_file_list, params_json_path, transform=None):
        self.file_list = npz_file_list
        self.transform = transform
        if os.path.exists(params_json_path):
            with open(params_json_path, 'r') as f: self.physics_params = json.load(f)
        else: self.physics_params = {}

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
        
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T # Physical alignment
        
        # Params
        params = self.physics_params.get(str(s_id), {"h_max": 100.0, "width": 94.0, "beta": 7.5, "lag": 20.0})
        phys_vec = torch.tensor([params['h_max'], params['width'], params['beta'], params['lag']], dtype=torch.float32)

        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, s_id, st_id, np.float32(mining_dist), phys_vec

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
            
            spatial_mask = (self.y_grid <= y_boundary).float()
            masks.append(spatial_mask)
            
        return torch.stack(masks).unsqueeze(1)

# --- Main Visualization Function ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4, help="Number of rows (samples) to plot")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset & Stats
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
    if not all_files: print("No data."); return
    
    # Split
    np.random.seed(53); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    
    # Stats
    stats_path = os.path.join(MODEL_DIR, "subsidence_stats_seed53.pt")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "subsidence_stats_ablation.pt")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(MODEL_DIR, "subsidence_stats.pt")
        
    if os.path.exists(stats_path):
        print(f"Loading stats from {stats_path}")
        stats = torch.load(stats_path)
        transform = NormalizeTransform(stats['mean'], stats['std'])
    else:
        print(f"Stats not found at {stats_path}. Cannot normalize correctly."); return

    dataset = SequentialFractureDataset(val_files, PARAMS_JSON_PATH, transform)
    # Loader (Batch size 1 for easy iterating and sorting)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Setup Models
    with np.load(val_files[0]) as f: total_feats = f['x'].shape[0]
    st_dim = STATIC_FEATURES; dyn_dim = total_feats - STATIC_FEATURES
    
    # Load Proposed
    model_mamba = DualBranchMambaModel(st_dim, dyn_dim, OUTPUT_FEATURES, 'dual').to(DEVICE)
    mamba_path = os.path.join(MODEL_DIR, "best_subsidence_full_dual_seed53.pth")
    # if not os.path.exists(mamba_path):
    #     mamba_path = os.path.join(MODEL_DIR, "best_subsidence_full_dual.pth") # Fallback to default name
        
    # Fallback to Physics Directory explicitly if still missing
    # if not os.path.exists(mamba_path):
    #     mamba_path = os.path.join(PHYSICS_DIR, "best_subsidence_physics_model.pth")
        
    if not os.path.exists(mamba_path): print(f"Mamba model missing at {mamba_path}"); return
    try:
        model_mamba.load_state_dict(torch.load(mamba_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Warning: Failed to load Mamba weights strictly (likely due to missing mamba_ssm). Loading with strict=False.")
        model_mamba.load_state_dict(torch.load(mamba_path, map_location=DEVICE), strict=False)
    model_mamba.eval()
    
    # Load Baselines
    models = [
        {"name": "Proposed", "model": model_mamba, "type": "Mamba"},
    ]
    
    # Transformer
    tr_path = os.path.join(TRANSFORMER_DIR, "best_baseline_subsidence_TRANSFORMER_seed53.pth")
    if os.path.exists(tr_path):
        model_tr = TransformerDualBranchModel(st_dim, dyn_dim, OUTPUT_FEATURES).to(DEVICE)
        model_tr.load_state_dict(torch.load(tr_path, map_location=DEVICE))
        model_tr.eval()
        models.append({"name": "Transformer", "model": model_tr, "type": "Transformer"})
    
    # CNN
    cnn_path = os.path.join(BASELINE_DIR, "best_subsidence_CNN_seed53.pth")
    if not os.path.exists(cnn_path): cnn_path = os.path.join(BASELINE_DIR, "best_baseline_subsidence_CNN_seed53.pth")
    if os.path.exists(cnn_path):
        model_cnn = DeepCNNBaseline(st_dim + dyn_dim).to(DEVICE)
        model_cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        model_cnn.eval()
        models.append({"name": "CNN", "model": model_cnn, "type": "CNN"})
    
    # LSTM
    lstm_path = os.path.join(BASELINE_DIR, "best_subsidence_LSTM_seed53.pth")
    if not os.path.exists(lstm_path): lstm_path = os.path.join(BASELINE_DIR, "best_baseline_subsidence_LSTM_seed53.pth")
    if os.path.exists(lstm_path):
        model_lstm = BiLSTMBaseline(dyn_dim, st_dim).to(DEVICE)
        model_lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        model_lstm.eval()
        models.append({"name": "LSTM", "model": model_lstm, "type": "LSTM"})
        
    print(f"Comparison Models: {[m['name'] for m in models]}")
    
    # 3. Filter Samples (Best MSE on Proposed AND PCR > 0.6 per Step Group)
    # Automatic Grouping Logic
    print("Scanning dataset for max steps...")
    max_step = 0
    for _, _, _, st_id, _, _ in loader:
        if st_id.item() > max_step: max_step = st_id.item()
    
    print(f"Max Step found: {max_step}. Plotting {args.samples} samples.")
    
    # Define Intervals
    # e.g. Max 54, Samples 4 -> [1-13, 14-27, 28-40, 41-54] approximately
    step_per_group = math.ceil(max_step / args.samples)
    groups = []
    for i in range(args.samples):
        start = i * step_per_group + 1
        end = min((i + 1) * step_per_group, max_step)
        groups.append({'range': (start, end), 'best_sample': None, 'best_mse': float('inf'), 
                       'fallback_sample': None, 'fallback_mse': float('inf')})
        
    print(f"Step Groups: {[g['range'] for g in groups]}")

    criterion = nn.MSELoss()
    mask_generator = ActivityArchMaskGenerator(output_size=64).to(DEVICE)
    
    print("Screening samples by group (PCR > 0.6)...")
    
    with torch.no_grad():
        for x, y, s_id, st_id, dist, phys_params in loader:
            current_step = st_id.item()
            
            # Find which group this belongs to
            target_group = None
            for g in groups:
                if g['range'][0] <= current_step <= g['range'][1]:
                    target_group = g
                    break
            
            if target_group is None: continue
            
            x = x.to(DEVICE); y = y.to(DEVICE)
            dist = dist.to(DEVICE); phys_params = phys_params.to(DEVICE)
            
            # 1. Proposed Prediction & MSE (Calculate first to enable fallback)
            pred_mamba = model_mamba(x)
            mse = criterion(pred_mamba, y.view(1, -1)).item()
            
            if math.isnan(mse) or math.isinf(mse): continue

            # 2. Always update fallback (Just best MSE seen so far)
            if mse < target_group['fallback_mse']:
                 target_group['fallback_mse'] = mse
                 target_group['fallback_sample'] = {
                    'x': x.cpu(), 'y': y.cpu(), 'mse': mse,
                    'info': (s_id.item(), st_id.item())
                }
            
            # 3. PCR Check (on Ground Truth)
            arch_masks = mask_generator(dist, phys_params)
            gt_bin = (y.view(1, 1, 64, 64) > 0.05).float()
            gt_pcr = (gt_bin * arch_masks).sum() / (gt_bin.sum() + 1e-6)
            
            # 4. Filter Logic
            # Group 0: No PCR check (subsidence potentially too small)
            # Group 1: Relaxed
            # Others: Strict
            
            group_idx = groups.index(target_group)
            if group_idx == 0: threshold = -1.0 # Pass all
            elif group_idx == 1: threshold = 0.4
            else: threshold = 0.6
            
            if gt_pcr <= threshold: continue
            
            # Subsidence Magnitude Check
            if group_idx > 0 and y.sum() <= 1.0: continue
            
            # 5. Update Best in Group (Valid Only - passed all filters)
            if mse < target_group['best_mse']:
                target_group['best_mse'] = mse
                target_group['best_sample'] = {
                    'x': x.cpu(), 'y': y.cpu(), 'mse': mse,
                    'info': (s_id.item(), st_id.item())
                }
                
    # Collect Results (Prefer Valid, else Fallback)
    top_samples = []
    for i, g in enumerate(groups):
        if g['best_sample'] is not None:
            top_samples.append(g['best_sample'])
        elif g['fallback_sample'] is not None:
            print(f"Group {i} ({g['range']}) used fallback (PCR failed or empty).")
            top_samples.append(g['fallback_sample'])
        else:
            print(f"Group {i} ({g['range']}) has NO sample (neither best nor fallback).")
    
    if not top_samples: print("No valid samples found."); return
    
    # 4. Plotting
    print(f"Plotting {len(top_samples)} samples...")
    
    # Grid: Rows = Samples. Cols = GT + Models
    n_rows = len(top_samples)
    n_cols = 1 + len(models) # GT + Models
    
    # Height per row ~ 4cm
    fig_height = (FIG_WIDTH_INCH / n_cols) * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_INCH, fig_height), squeeze=False)
    
    # Color limits (Common for visual comparison)
    vmin, vmax = 0.0, 1.0 # Subsidence usually 0-1, maybe slightly more
    
    for r, sample in enumerate(top_samples):
        x_in = sample['x'].to(DEVICE)
        y_gt = sample['y'].squeeze().numpy()
        
        # 1. Plot GT
        ax_gt = axes[r, 0]
        # [Fix] origin='lower' to fix axis orientation
        im = ax_gt.imshow(y_gt, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        # if r == 0: ax_gt.set_title("Ground Truth", fontweight='bold')
        ax_gt.set_ylabel(f"({r+1})", rotation=0, fontweight='bold', labelpad=10, va='center')
        if r == n_rows - 1: ax_gt.set_xlabel("(a) Ground Truth", fontweight='bold')
        
        # 2. Plot Models
        for c, m_cfg in enumerate(models):
            ax = axes[r, c+1]
            model = m_cfg['model']
            with torch.no_grad():
                pred = model(x_in).view(64, 64).cpu().numpy()
            
            im = ax.imshow(pred, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xticks([]); ax.set_yticks([])
            # if r == 0: ax.set_title(m_cfg['name'], fontweight='bold')
            if r == n_rows - 1: ax.set_xlabel(f"({chr(98+c)}) {m_cfg['name']}", fontweight='bold', fontsize=9)
            
    plt.tight_layout(pad=0.5, w_pad=1.5, h_pad=0.5)
    # Global Colorbar
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Fracture Density", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    
    save_path = os.path.join(OUTPUT_DIR, "comparison_results_sci.svg")
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    # Also save PNG for checking
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
