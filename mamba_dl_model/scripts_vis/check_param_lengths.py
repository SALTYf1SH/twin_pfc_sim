
import json
import os

stress_json = r"f:\PFCprj\pfc_twin\twin_pfc_sim\mamba_dl_model\stress_para\stress_physics_params.json"
subsidence_json = r"f:\PFCprj\pfc_twin\twin_pfc_sim\mamba_dl_model\subsidence_para\subsidence_physics_params.json"

def check(path, name):
    if not os.path.exists(path):
        print(f"{name}: File not found")
        return
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Get first key
    first_key = list(data.keys())[0]
    entry = data[first_key]
    
    print(f"--- {name} Params (Sample {first_key}) ---")
    keys = ["ks_heights", "ks_betas"]
    for k in keys:
        if k in entry:
            print(f"{k}: len = {len(entry[k])}")
        else:
            print(f"{k}: Not found")

check(stress_json, "Stress")
check(subsidence_json, "Subsidence")
