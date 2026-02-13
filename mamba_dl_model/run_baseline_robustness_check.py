import os
import subprocess
import time

SEEDS = [42, 43, 44]
MODELS = ["CNN", "LSTM", "TRANSFORMER", "MAMBA"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_experiment(script_name, model_type, seeds):
    print(f"\n========================================================")
    print(f"   Robustness Check: {script_name} [{model_type}]")
    print(f"========================================================")
    
    for seed in seeds:
        print(f"\n--- Starting Seed: {seed} ---")
        cmd = ["python", os.path.join(SCRIPT_DIR, "scripts_train", script_name), 
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
    
    # Iterate through all models
    for model in MODELS:
        # Stress Baselines
        run_experiment("train_baselines_stress_robustness.py", model, SEEDS)
        
        # Subsidence Baselines
        run_experiment("train_baselines_subsidence_robustness.py", model, SEEDS)
    
    print("\nAll Baseline Robustness Checks Completed.")
