
import numpy as np
import glob
import os

def inspect(path, name):
    files = glob.glob(os.path.join(path, "*.npz"))
    if not files:
        print(f"No files found in {path}")
        return
    
    f = files[0]
    data = np.load(f)
    print(f"--- {name} Dataset Inspection ({f}) ---")
    print("Keys:", data.files)
    if 'x' in data:
        x = data['x']
        print(f"Shape of x: {x.shape}")
        # Assuming static parameters are at the beginning or end? 
        # Often static params are repeated or just 1D if it's per sample.
        # But here x is likely (steps, features) or (features,). 
        # Let's see some values.
        print("First 20 values of flattened x:", x.flatten()[:20])
        
stress_path = r"f:\PFCprj\pfc_twin\twin_pfc_sim\final_dataset_stress"
subsidence_path = r"f:\PFCprj\pfc_twin\twin_pfc_sim\final_dataset"

inspect(stress_path, "Stress")
inspect(subsidence_path, "Subsidence")
