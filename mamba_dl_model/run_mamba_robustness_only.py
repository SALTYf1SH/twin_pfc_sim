import os
import subprocess
import time

# 只运行 Mamba，补全缺失的训练
SEEDS = [42, 43, 44] # 42 exists but running again is safe/checks it. Or we can skip if check exists.
MODELS = ["MAMBA"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_experiment(script_name, model_type, seeds):
    print(f"\n========================================================")
    print(f"   Mamba Robustness Recovery: {script_name} [{model_type}]")
    print(f"========================================================")
    
    for seed in seeds:
        print(f"\n--- Starting Seed: {seed} ---")
        cmd = ["E:\\Anaconda\\envs\\mamba\\python.exe", os.path.join(SCRIPT_DIR, "scripts_train", script_name), 
               "--model_type", model_type, 
               "--seed", str(seed)]
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"--- Finished Seed {seed} in {(end_time - start_time)/60.0:.2f} mins ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {model_type} seed {seed}: {e}")
            
if __name__ == "__main__":
    
    # Stress Baselines
    run_experiment("train_baselines_stress_robustness.py", "MAMBA", SEEDS)
    
    # Subsidence Baselines
    run_experiment("train_baselines_subsidence_robustness.py", "MAMBA", SEEDS)
    
    print("\nMamba Baseline Training Completed.")
