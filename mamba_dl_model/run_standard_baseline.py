import os
import subprocess
import time

# Configuration
PYTHON_EXEC = r"E:\Anaconda\envs\mamba\python.exe"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_SCRIPT = os.path.join(BASE_DIR, "scripts_train/train_stress_physics_mamba.py")
SUBSIDENCE_SCRIPT = os.path.join(BASE_DIR, "scripts_train/train_subsidence_physics_mamba.py")

def run_training(script_path, task_name):
    print(f"\n========================================================")
    print(f"   Re-training Standard Baseline: {task_name}")
    print(f"========================================================")
    
    cmd = [PYTHON_EXEC, script_path] 
    # No extra arguments -> uses new defaults (Low Physics)
    
    try:
        start_time = time.time()
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        end_time = time.time()
        print(f"--- Finished {task_name} in {(end_time - start_time)/60.0:.2f} mins ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {task_name}: {e}")

if __name__ == "__main__":
    print("Starting Standard Baseline Re-training (Low Physics Defaults)...")
    
    # 1. Stress
    run_training(STRESS_SCRIPT, "Stress")
    
    # 2. Subsidence
    run_training(SUBSIDENCE_SCRIPT, "Subsidence")
    
    print("\nAll Standard Baselines Re-trained.")
