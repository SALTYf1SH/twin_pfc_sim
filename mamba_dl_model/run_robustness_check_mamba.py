import os
import subprocess
import time

SEEDS = [42, 43, 44]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_experiment(script_name, seeds):
    print(f"==================================================")
    print(f"   Running Robustness Check: {script_name}")
    print(f"==================================================")
    
    for seed in seeds:
        print(f"\n--- Starting Experiment with Seed: {seed} ---")
        cmd = ["python", os.path.join(SCRIPT_DIR, "scripts_train", script_name), "--seed", str(seed)]
        
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"--- Finished Seed {seed} in {(end_time - start_time)/60.0:.2f} mins ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running seed {seed}: {e}")
            
if __name__ == "__main__":
    # Stress robust check
    run_experiment("train_stress_robustness.py", SEEDS)
    
    # Subsidence robust check
    run_experiment("train_subsidence_robustness.py", SEEDS)
    
    print("\nAll Robustness Checks Completed.")
