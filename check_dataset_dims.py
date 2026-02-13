
import os
import glob
import numpy as np

def check_dataset(name, path):
    print(f"Checking {name} at {path}...")
    files = glob.glob(os.path.join(path, "*.npz"))
    if not files:
        print(f"  No .npz files found.")
        return
    
    print(f"  Found {len(files)} files.")
    with np.load(files[0]) as f:
        x = f['x']
        print(f"  Shape of x: {x.shape}")
        # x shape is commonly (sequence_length, num_features) or similar
        # If it's single step, it might be (num_features,)
        
if __name__ == "__main__":
    base_dir = "f:/PFCprj/pfc_twin/twin_pfc_sim/"
    check_dataset("final_dataset", os.path.join(base_dir, "final_dataset"))
    check_dataset("final_dataset_stress", os.path.join(base_dir, "final_dataset_stress"))
