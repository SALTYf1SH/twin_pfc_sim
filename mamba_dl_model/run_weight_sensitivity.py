import os
import subprocess
import time

# Configuration
PYTHON_EXEC = r"E:\Anaconda\envs\mamba\python.exe"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_SCRIPT = os.path.join(BASE_DIR, "scripts_train/train_stress_physics_mamba.py")
SUBSIDENCE_SCRIPT = os.path.join(BASE_DIR, "scripts_train/train_subsidence_physics_mamba.py")

# Weight Configurations (ssim, tv, arch, evo)
# Default: (0.3, 1e-5, 0.5, 0.2)
# Weight Configurations (ssim, tv, arch, evo)
# Default Baseline: SSIM=0.3, Phys=0.1 (Arch=0.1, Evo=0.05)

CONFIGS = {
    # --- Group A: Physics Sensitivity (Fixed SSIM=0.3) ---
    "physics_low":      {"ssim": 0.3, "tv": 1e-5, "arch": 0.01, "evo": 0.005}, # Low Physics
    "physics_baseline": {"ssim": 0.3, "tv": 1e-5, "arch": 0.1,  "evo": 0.05},  # Baseline (Medium)
    "physics_high":     {"ssim": 0.3, "tv": 1e-5, "arch": 0.5,  "evo": 0.2},   # High Physics (Old Baseline)

    # --- Group B: SSIM Sensitivity (Fixed Phys=Baseline) ---
    "ssim_low":         {"ssim": 0.1, "tv": 1e-5, "arch": 0.1,  "evo": 0.05},  # Low SSIM
    # "ssim_baseline":  {"ssim": 0.3, ...} -> Same as physics_baseline
    "ssim_high":        {"ssim": 0.8, "tv": 1e-5, "arch": 0.1,  "evo": 0.05}   # High SSIM
}

# Config for quick test (uncomment for real run)
# NUM_EPOCHS = 100 
NUM_EPOCHS = 100 # Reduced for faster sensitivity analysis, but still enough to see trends

def run_experiment(script_path, task_name, config_name, params):
    ablation_name = f"sens_{config_name}"
    
    # Construct Output Directory Name
    # Matches train script: trained_models_{task}_{ablation_name}
    output_dir_name = f"trained_models_{task_name.lower()}_{ablation_name}"
    output_dir = os.path.join(BASE_DIR, output_dir_name)
    
    # Check if completed
    if os.path.exists(os.path.join(output_dir, "val_metrics.json")):
        print(f"!!! Skipping {config_name} (Already completed: {output_dir}) !!!")
        return

    cmd = [
        PYTHON_EXEC, script_path,
        "--ablation_name", ablation_name,
        "--branch_mode", "dual",
        "--lambda_ssim", str(params["ssim"]),
        "--lambda_tv", str(params["tv"]),
        "--lambda_arch", str(params["arch"]),
        "--lambda_evo", str(params["evo"]),
        "--num_epochs", str(NUM_EPOCHS),
    ]
    
    print(f"[{task_name}] Running config: {config_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run synchronously
    subprocess.run(cmd, check=True)

def main():
    print("Starting Weight Sensitivity Analysis...")
    
    # 1. Stress Experiments
    print("\n--- Running Stress Inversion Experiments ---")
    for name, params in CONFIGS.items():
        try:
            run_experiment(STRESS_SCRIPT, "Stress", name, params)
        except Exception as e:
            print(f"Error running Stress {name}: {e}")

    # 2. Subsidence Experiments
    print("\n--- Running Subsidence Prediction Experiments ---")
    for name, params in CONFIGS.items():
        try:
            run_experiment(SUBSIDENCE_SCRIPT, "Subsidence", name, params)
        except Exception as e:
            print(f"Error running Subsidence {name}: {e}")
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
