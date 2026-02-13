
import os
import glob
import json
import numpy as np

def check_indices(task, dataset_dir, json_path=None):
    print(f"Checking {task}...")
    # params_dict = {}
    # if json_path and os.path.exists(json_path):
    #     with open(json_path, 'r') as f:
    #         params_dict = json.load(f)
        
    files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if not files:
        print("No files found.")
        print(f"Directory was: {dataset_dir}")
        return

    # Load 50 files
    data_list = []
    # Count how many files loaded
    loaded = 0
    for f in files:
        if loaded >= 50: break
        try:
            with np.load(f) as d:
                data_list.append(d['x'])
                loaded += 1
        except: pass
    
    if not data_list: return

    data_stack = np.stack(data_list) # (50, 34)
    mins = data_stack.min(axis=0)
    maxs = data_stack.max(axis=0)
    means = data_stack.mean(axis=0)
    stds = data_stack.std(axis=0)
    
    # Print ALL 0-17 for Stress
    print(f"Feature Statistics (0-17) over {len(data_list)} samples for {task}:")
    for i in range(len(means)):
        if i >= 18: break
        
        # Highlight candidates
        marker = ""
        if stds[i] < 1e-6: marker += " [CONSTANT]"
        if 1.0 < means[i] < 50.0 and stds[i] > 0.001: marker += " [CANDIDATE: Thick?]"
            
        print(f"Feat {i:02d}: Mean={means[i]:.4f}, Std={stds[i]:.4f}, Range=[{mins[i]:.4f}, {maxs[i]:.4f}]{marker}")
    print("-" * 30)

def main():
    base = "f:/PFCprj/pfc_twin/twin_pfc_sim/mamba_dl_model"
    check_indices("Stress", os.path.join(base, "../final_dataset_stress"), "") 
    # Skip Subsidence for now
    
    # Stress
    check_indices("Stress", 
                  os.path.join(base, "../final_dataset_stress"),
                  os.path.join(base, "stress_para/stress_physics_params.json"))
                  
    # Subsidence
    check_indices("Subsidence",
                  os.path.join(base, "../final_dataset"),
                  os.path.join(base, "subsidence_para/subsidence_physics_params.json"))

if __name__ == "__main__":
    main()
