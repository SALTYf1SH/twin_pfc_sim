
import os
import glob
import numpy as np
import torch
import json
from tqdm import tqdm

# Config
STRESS_DIR = "../final_dataset_stress"
SUB_DIR = "../final_dataset"
STEP_DISTANCE_M = 10.0
IMG_SIZE = 64

def get_files(dataset_dir):
    return glob.glob(os.path.join(os.path.dirname(__file__), dataset_dir, "*.npz"))

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    s_pos = filename.rfind("sample"); st_pos = filename.rfind("step")
    s_id = int(filename[s_pos+7 : s_pos+11])
    st_id = int(filename[st_pos+5 : st_pos+8])
    return s_id, st_id

def analyze_dataset(name, dataset_dir, is_stress=False):
    files = get_files(dataset_dir)
    if not files:
        print(f"[{name}] No files found in {dataset_dir}")
        return

    print(f"--- Analyzing {name} ({len(files)} files) ---")
    
    # Build Map
    file_map = {}
    for f in files:
        try:
            sid, stid = parse_filename(f)
            file_map[(sid, stid)] = f
        except: pass

    total_delta_sq = 0.0
    total_val_sq = 0.0
    count = 0
    max_delta = 0.0
    
    # Random sample to save time if large
    import random
    random.shuffle(files)
    sample_files = files[:500] 

    for f in tqdm(sample_files):
        sid, stid = parse_filename(f)
        if stid <= 1: continue # No prev
        
        prev_f = file_map.get((sid, stid-1))
        if not prev_f: continue

        try:
            # Load Current
            with np.load(f) as d:
                y = d['y'].astype(np.float32)
                if is_stress:
                    if y.ndim==1: y=y.reshape(64,64)
                    y = y.T
                else: 
                     # Subsidence logic from eval script
                     if y.ndim==1: y=y.reshape(64,64)
                     y = y.T

            # Load Prev
            with np.load(prev_f) as d:
                y_prev = d['y'].astype(np.float32)
                if is_stress:
                   if y_prev.ndim==1: y_prev=y_prev.reshape(64,64)
                   y_prev = y_prev.T
                else:
                   if y_prev.ndim==1: y_prev=y_prev.reshape(64,64)
                   y_prev = y_prev.T
            
            # Calc
            val_sq = np.mean(y**2)
            delta = y - y_prev
            delta_sq = np.mean(delta**2)
            
            total_val_sq += val_sq
            total_delta_sq += delta_sq
            max_delta = max(max_delta, np.max(np.abs(delta)))
            count += 1
        except Exception as e:
            pass

    if count == 0:
        print("No pairs found.")
        return

    avg_val_sq = total_val_sq / count
    avg_delta_sq = total_delta_sq / count
    
    print(f"Avg Value MS (Energy): {avg_val_sq:.6f}")
    print(f"Avg Delta MS (Change Energy): {avg_delta_sq:.8f}")
    print(f"Ratio (Change/Value): {avg_delta_sq/avg_val_sq:.6f}")
    print(f"Max Delta: {max_delta:.6f}")
    print(f"Root Mean Square Delta: {np.sqrt(avg_delta_sq):.6f}")

if __name__ == "__main__":
    analyze_dataset("SUBSIDENCE", SUB_DIR, is_stress=False)
    analyze_dataset("STRESS", STRESS_DIR, is_stress=True)
