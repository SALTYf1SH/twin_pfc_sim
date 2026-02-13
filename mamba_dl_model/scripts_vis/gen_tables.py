
import pandas as pd
import io

csv_path_ablation = "f:/PFCprj/pfc_twin/twin_pfc_sim/mamba_dl_model/robustness_mamba_ablation_results.csv"
csv_path_baseline = "f:/PFCprj/pfc_twin/twin_pfc_sim/mamba_dl_model/robustness_baseline_results.csv"

try:
    # Now all results (ablation + baseline) are in the single CSV
    df = pd.read_csv(csv_path_ablation)
    
    # Check if we need to rename columns (Config is standard now)
    if 'Model' in df.columns and 'Config' not in df.columns:
        df = df.rename(columns={'Model': 'Config'})
        
    # Select common columns of interest
    cols = ['Task', 'Config', 'Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper']
    df = df[cols]
    
except Exception as e:
    print(f"Error reading or processing CSVs: {e}")
    exit()

# Filter Lists
ABLATION_MODELS = [
    "full_dual", 
    "full_dynamic_only", 
    "full_static_only", 
    "no_physics_dual_no_phys", 
    "vanilla_mamba"
]

BASELINE_MODELS = [
    "full_dual", 
    "LSTM", 
    "CNN", 
    "TRANSFORMER",
    "vanilla_mamba"
]

MODEL_NAME_MAP = {
    "full_dual": "Proposed (Mamba)",
    "full_dynamic_only": "Dynamic Only",
    "full_static_only": "Static Only",
    "no_physics_dual_no_phys": "No Physics",
    "vanilla_mamba": "Vanilla Mamba",
    "LSTM": "LSTM",
    "CNN": "CNN",
    "TRANSFORMER": "Transformer"
}

METRICS = ["MSE", "MAE", "SSIM", "PCC", "PCR", "Evo"]

def generate_table(task, model_list, title, file_suffix):
    print(f"### {title}")
    header = "| Model | " + " | ".join(METRICS) + " |"
    print(header)
    print("| :--- | " + " | ".join([":---:"] * len(METRICS)) + " |")
    
    table_rows = []

    for model_key in model_list:
        display_name = MODEL_NAME_MAP.get(model_key, model_key)
        row_str = f"| **{display_name}** |"
        
        row_data = {"Model": display_name}

        for metric in METRICS:
            # Find row
            subset = df[(df['Task'] == task) & (df['Config'] == model_key) & (df['Metric'] == metric)]
            if not subset.empty:
                val_mean = subset['Mean'].values[0]
                val_std = subset['Std'].values[0]
                val_ci_low = subset['CI_Lower'].values[0]
                val_ci_high = subset['CI_Upper'].values[0]
                
                # Format logic
                # For MSE: scientific notation if too small
                if metric == "MSE":
                    if val_mean < 0.0001:
                        fmt = f"{val_mean:.2e}± {val_std:.2e} ({val_ci_low:.2e}, {val_ci_high:.2e})"
                    else:
                        fmt = f"{val_mean:.5f}± {val_std:.5f} ({val_ci_low:.5f}, {val_ci_high:.5f})"
                else:
                    fmt = f"{val_mean:.5f}± {val_std:.5f} ({val_ci_low:.5f}, {val_ci_high:.5f})"
                    
                row_str += f" {fmt} |"
                row_data[metric] = fmt
            else:
                row_str += " N/A |"
                row_data[metric] = "N/A"
        print(row_str)
        table_rows.append(row_data)
    print("\n")
    
    # Save to CSV
    # Construct DataFrame from rows
    df_table = pd.DataFrame(table_rows)
    # Ensure column order
    cols_order = ["Model"] + METRICS
    df_table = df_table[cols_order]
    
    out_csv_path = f"f:/PFCprj/pfc_twin/twin_pfc_sim/mamba_dl_model/scripts_vis/{file_suffix}.csv"
    df_table.to_csv(out_csv_path, index=False)
    print(f"Table saved to: {out_csv_path}\n")

# 1. Stress Ablation
generate_table("stress", ABLATION_MODELS, "Table 1: Stress Inversion - Ablation Study", "table1_stress_ablation")

# 2. Stress Baseline
generate_table("stress", BASELINE_MODELS, "Table 2: Stress Inversion - Baseline Comparison", "table2_stress_baseline")

# 3. Subsidence Ablation
generate_table("subsidence", ABLATION_MODELS, "Table 3: Subsidence Prediction - Ablation Study", "table3_subsidence_ablation")

# 4. Subsidence Baseline
generate_table("subsidence", BASELINE_MODELS, "Table 4: Subsidence Prediction - Baseline Comparison", "table4_subsidence_baseline")
