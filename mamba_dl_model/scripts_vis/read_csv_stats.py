
import pandas as pd
import os

vis_dir = r"f:\PFCprj\pfc_twin\twin_pfc_sim\mamba_dl_model\dataset_vis"
files = ["stress_parameter_stats.csv", "subsidence_parameter_stats.csv"]

for f in files:
    path = os.path.join(vis_dir, f)
    if os.path.exists(path):
        print(f"--- {f} ---")
        try:
            df = pd.read_csv(path)
            print(df.to_string())
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
