import subprocess
import time
import os

PYTHON_EXEC = r"E:\Anaconda\envs\mamba\python.exe"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"\n>>> Starting {script_name} <<<")
    try:
        start_time = time.time()
        subprocess.run([PYTHON_EXEC, script_path], check=True)
        print(f">>> Finished {script_name} in {(time.time() - start_time)/60.0:.2f} mins")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error in {script_name}: {e}")

if __name__ == "__main__":
    print("=== Starting Remaining Tasks Pipeline ===")
    
    # 1. Re-train Standard Baseline (New Defaults)
    run_script("run_standard_baseline.py")
    
    # 2. Run Ablation Robustness Checks
    # This runs train_stress_robustness.py and train_subsidence_robustness.py for ablations
    run_script("run_ablation_robustness_check.py")
    
    # 3. Evaluate Ablation Models
    # Evaluate Stress
    run_script("scripts_eval/evaluate_stress_ablation_mamba.py")
    # Evaluate Subsidence
    run_script("scripts_eval/evaluate_subsidence_ablation_mamba.py")
    
    # 4. Generate Radar Plots
    run_script("scripts_vis/visualize_ablation_radar_stress.py")
    run_script("scripts_vis/visualize_ablation_radar_subsidence.py")
    
    # 5. Generate Sensitivity Plots (New)
    run_script("scripts_vis/analyze_weight_sensitivity.py")
    
    print("\n=== All Remaining Tasks Completed ===")
