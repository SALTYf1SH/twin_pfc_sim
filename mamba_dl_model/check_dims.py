import glob
import os
import numpy as np
from collections import Counter
from tqdm import tqdm

def check_dataset(name, path):
    print(f"\n{'='*40}")
    print(f"Checking {name} at {path}...")
    print(f"{'='*40}")
    
    files = glob.glob(os.path.join(path, "*.npz"))
    if not files:
        print(f"  No .npz files found.")
        return
    
    print(f"  Found {len(files)} files.")
    
    shapes = Counter()
    example_files = {} 
    
    print("  Scanning all files...")
    # Scan first 500 files to save time if dataset is huge, or all if needed.
    # User said "check all", so we check all.
    for f in tqdm(files):
        try:
            with np.load(f) as data:
                s = data['x'].shape
                shapes[s] += 1
                if s not in example_files:
                    example_files[s] = os.path.basename(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print("\n  Shape Distribution:")
    for shape, count in shapes.items():
        print(f"    Shape {shape}: {count} files (Example: {example_files[shape]})")
        
    sorted_files = sorted(files)
    print("\n  Checking first file (Sorted order):")
    try:
        with np.load(sorted_files[0]) as data:
             print(f"    {os.path.basename(sorted_files[0])}: {data['x'].shape}")
    except: pass
        
if __name__ == "__main__":
    base_dir = "f:/PFCprj/pfc_twin/twin_pfc_sim/"
    check_dataset("final_dataset_stress", os.path.join(base_dir, "final_dataset_stress"))
