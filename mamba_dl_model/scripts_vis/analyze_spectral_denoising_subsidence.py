
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import ndimage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Configuration ---
# SCI Formatting (Width <= 17cm ~ 6.7 inch)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9 
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['svg.fonttype'] = 'none' # Ensure text is editable in SVG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../../final_dataset") # Subsidence Dataset
MODEL_DIR_PHYSICS = os.path.join(BASE_DIR, "../trained_models_subsidence_physics_mamba")
MODEL_DIR_ABLATION = os.path.join(BASE_DIR, "../trained_models_subsidence_ablation_mamba")
OUTPUT_DIR = os.path.join(BASE_DIR, "../visualization_results_subsidence") # Reusing radar dir or create new? Let's use visualization_results_subsidence
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    
PARAMS_JSON_PATH = os.path.join(BASE_DIR, "../subsidence_para/subsidence_physics_params.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64
STATIC_FEATURES = 11 # Subsidence has 11 static features
BATCH_SIZE = 64

# --- Dataset & Model Definitions ---

class NormalizeTransform:
    def __init__(self, mean, std): self.mean = mean; self.std = std
    def __call__(self, x): return (x - self.mean) / (self.std + 1e-8)

class SequentialSubsidenceAnalysisDataset(Dataset):
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        with np.load(curr_path) as data:
            x_t = torch.from_numpy(data['x'].astype(np.float32))
            y_t = torch.from_numpy(data['y'].astype(np.float32))
            if y_t.ndim == 1: y_t = y_t.reshape(IMG_SIZE, IMG_SIZE)
            y_t = y_t.T # [CRITICAL] Subsidence requires Transpose for correct interaction
            
        if self.transform: x_t = self.transform(x_t)
        return x_t, y_t, curr_path

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

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
                    x_mamba = norm(x_mamba + layer(x_mamba)) 
                dynamic_out = x_mamba.mean(dim=1)
            else: dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)
        else:
            dynamic_out = torch.zeros(x.size(0), self.d_model, device=x.device)

        fused = torch.cat((static_out, dynamic_out), dim=1)
        return self.fusion_head(fused)

# --- Spectral Analysis Functions ---

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)
    return radialprofile

def calculate_psd(images):
    # images: [B, H, W] numpy array
    psds = []
    for img in images:
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        center = (img.shape[1]//2, img.shape[0]//2)
        profile = radial_profile(magnitude_spectrum, center)
        psds.append(profile)
        
    return np.mean(psds, axis=0) # Average over batch

# --- Heuristic Quality Check (Replacing Complex PCR) ---
def calculate_gt_quality_proxy(img_np):
    """
    Calculates the ratio of high-frequency energy in the GT image.
    High Ratio = Noisy GT (Bad). Low Ratio = Clean GT (Good).
    Score = 1 - Ratio (Higher is Better).
    """
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    
    h, w = img_np.shape
    center = (w//2, h//2)
    Y, X = np.indices((h,w))
    r = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    
    # High freq > 20
    energy_high = np.sum(mag[r > 20])
    energy_total = np.sum(mag)
    ratio = energy_high / (energy_total + 1e-8)
    
    return 1.0 - ratio

# --- Main ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    import glob
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files: print("No data found."); return
    
    # Use Validation Set (Last 10%)
    np.random.seed(42); np.random.shuffle(all_files)
    val_files = all_files[int(0.9 * len(all_files)):]
    print(f"Total Validation Samples: {len(val_files)}")
    
    # Calculate Stats for Transform
    print("Calculating Stats...")
    temp_loader = DataLoader(SequentialSubsidenceAnalysisDataset(all_files[:100]), batch_size=128)
    all_x = []
    for x, _, _ in temp_loader: all_x.append(x)
    x_tensor = torch.cat(all_x, dim=0)
    transform = NormalizeTransform(x_tensor.mean(dim=0), x_tensor.std(dim=0))
    
    # 2. Filter Samples (Top 50% Quality Proxy)
    print("Filtering Top 50% Quality Samples...")
    
    dataset_full = SequentialSubsidenceAnalysisDataset(val_files, transform=None) 
    qualities = []
    
    indices = []
    for i in tqdm(range(len(val_files))):
        _, gt, _ = dataset_full[i]
        gt_np = gt.numpy()
        score = calculate_gt_quality_proxy(gt_np)
        qualities.append((i, score))
        
    # Sort by Score Descending
    qualities.sort(key=lambda x: x[1], reverse=True)
    
    # Top 50%
    top_k = int(len(qualities) * 0.5)
    top_indices = [q[0] for q in qualities[:top_k]]
    filtered_files = [val_files[i] for i in top_indices]
    
    print(f"Selected {len(filtered_files)} High-Quality Samples for Analysis.")
    
    # 3. Prepare Final Loader
    dataset = SequentialSubsidenceAnalysisDataset(filtered_files, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get Dims
    with np.load(filtered_files[0]) as f: total_feats = f['x'].shape[0]
    static_dim = STATIC_FEATURES
    dynamic_dim = total_feats - STATIC_FEATURES
    output_dim = 64 * 64
    
    # 4. Load Models
    print("Loading Models...")
    
    model_proposed = DualBranchMambaModel(static_dim, dynamic_dim, output_dim, branch_mode='dual').to(DEVICE)
    path_proposed = os.path.join(MODEL_DIR_PHYSICS, "best_subsidence_physics_model.pth")
    if os.path.exists(path_proposed): model_proposed.load_state_dict(torch.load(path_proposed, map_location=DEVICE))
    else: print(f"Warning: Proposed Model not found at {path_proposed}")
    
    model_nophy = DualBranchMambaModel(static_dim, dynamic_dim, output_dim, branch_mode='dual').to(DEVICE)
    path_nophy = os.path.join(MODEL_DIR_ABLATION, "best_subsidence_no_physics_dual.pth") # Subsidence Ablation
    if os.path.exists(path_nophy): model_nophy.load_state_dict(torch.load(path_nophy, map_location=DEVICE))
    else: print(f"Warning: NoPhysics Model not found at {path_nophy}")

    model_proposed.eval()
    model_nophy.eval()
    
    # 5. Inference & Spectral Analysis
    all_gt_psd = []
    all_proposed_psd = []
    all_nophy_psd = []
    
    with torch.no_grad():
        for x, y, _ in tqdm(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            pred_proposed = model_proposed(x).view(-1, 64, 64).cpu().numpy()
            pred_nophy = model_nophy(x).view(-1, 64, 64).cpu().numpy()
            gt = y.cpu().numpy()
            
            all_gt_psd.append(calculate_psd(gt))
            all_proposed_psd.append(calculate_psd(pred_proposed))
            all_nophy_psd.append(calculate_psd(pred_nophy))
            
    avg_gt_psd = np.mean(all_gt_psd, axis=0)
    avg_proposed_psd = np.mean(all_proposed_psd, axis=0)
    avg_nophy_psd = np.mean(all_nophy_psd, axis=0)
    
    # 6. Normalize PSD (Shift to 0 at DC)
    norm_gt = avg_gt_psd - avg_gt_psd[0]
    norm_proposed = avg_proposed_psd - avg_proposed_psd[0]
    norm_nophy = avg_nophy_psd - avg_nophy_psd[0]
    
    freqs = np.arange(len(norm_gt))
    
    # Plotting (Width <= 17cm -> 6.7 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 3.0))
    
    # Subplot 1: Normalized Spectrum
    ax1.plot(freqs, norm_gt, 'k-', linewidth=1.0, alpha=0.8, label='Simulation GT')
    ax1.plot(freqs, norm_nophy, 'r--', linewidth=1.5, label='No Physics')
    ax1.plot(freqs, norm_proposed, 'b-', linewidth=1.5, label='Proposed (Physics)')
    
    ax1.set_xlabel("Spatial Frequency\n(a) Normalized Radial Power Spectrum")
    ax1.set_ylabel("Relative Power (dB)")
    
    # Legend Bottom Left
    ax1.legend(loc='lower left', frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Inset: Zoom High Freq (Subsidence might be cleaner, but let's check >25 first? Keep >40 for consistency unless plot is weird)
    # Let's start with >30 to be safe, Subsidence is usually smoother than stress.
    axins = inset_axes(ax1, width="40%", height="30%", loc=1, borderpad=2)
    axins.plot(freqs[30:], norm_gt[30:], 'k-', alpha=0.8)
    axins.plot(freqs[30:], norm_nophy[30:], 'r--', linewidth=2)
    axins.plot(freqs[30:], norm_proposed[30:], 'b-', linewidth=2)
    axins.grid(True, alpha=0.3)
    
    # Subplot 2: Difference Plot
    # Shows how much EXTRA noise NoPhysics has compared to Proposed
    diff_noise = norm_nophy - norm_proposed
    
    ax2.plot(freqs, diff_noise, 'r-', linewidth=1.5, label='Diff. Curve')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.fill_between(freqs, 0, diff_noise, where=(diff_noise > 0), color='r', alpha=0.2, label='Noise Gap')
    # ax2.fill_between(freqs, 0, diff_noise, where=(diff_noise < 0), color='b', alpha=0.2) # Removed fill for smoothing if not needed? 
    # User asked to remove Smoothing LEGEND. Did they ask to remove the FILL? 
    # "delete Smoothing legend". Fill can stay, just no label.
    ax2.fill_between(freqs, 0, diff_noise, where=(diff_noise < 0), color='b', alpha=0.2)
    
    ax2.set_xlabel("Spatial Frequency\n(b) Noise Spectrum Difference")
    ax2.set_ylabel("Power Difference (dB)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    output_path = os.path.join(OUTPUT_DIR, "subsidence_spectral_analysis_denoising.svg")
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.savefig(output_path.replace('.svg', '.png'), bbox_inches='tight')
    
    print(f"Refined Subsidence Analysis Saved to: {output_path}")

if __name__ == "__main__":
    main()
