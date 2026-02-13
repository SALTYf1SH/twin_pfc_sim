import os
import subprocess
import time

SEEDS = [42, 43, 44]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ablation Configurations
# Format: (Description, Arguments List)
ABLATIONS = [
    ("No_Physics",   ["--no_physics", "--ablation_name", "no_physics"]),
    ("Static_Only",  ["--branch_mode", "static_only", "--ablation_name", "full"]), 
    ("Dynamic_Only", ["--branch_mode", "dynamic_only", "--ablation_name", "full"]),
]

def run_experiment(script_name, ablation_desc, ablation_args, seeds):
    print(f"\n========================================================")
    print(f"   Robustness Check: {script_name} [{ablation_desc}]")
    print(f"========================================================")
    
    for seed in seeds:
        print(f"\n--- Starting Seed: {seed} ---")
        
        # Use absolute path to python
        PYTHON_EXEC = r"E:\Anaconda\envs\mamba\python.exe"
        cmd = [PYTHON_EXEC, os.path.join(SCRIPT_DIR, "scripts_train", script_name)]
        cmd.extend(ablation_args)
        cmd.extend(["--seed", str(seed)])
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"--- Finished Seed {seed} in {(end_time - start_time)/60.0:.2f} mins ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {ablation_desc} seed {seed}: {e}")

if __name__ == "__main__":
    
    for desc, args in ABLATIONS:
        # Stress Ablation
        run_experiment("train_stress_robustness.py", desc, args, SEEDS)
        
        # Subsidence Ablation
        run_experiment("train_subsidence_robustness.py", desc, args, SEEDS)
    
    print("\nAll Ablation Robustness Checks Completed.")
