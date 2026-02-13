
import pandas as pd
import os

try:
    df = pd.read_csv('robustness_mamba_ablation_results.csv')
    # Filter for stress and full configuration
    stress_results = df[(df['Task'] == 'stress') & (df['Config'].isin(['full_dual', 'full', 'Proposed (Mamba)']))]
    
    if stress_results.empty:
        # Maybe config name is different, print unique configs for stress
        print("Configs found for stress:", df[df['Task'] == 'stress']['Config'].unique())
        print("All Rows for stress:")
        print(df[df['Task'] == 'stress'])
    else:
        print(stress_results)
        
    print("\nColumn headers:", df.columns.tolist())

except Exception as e:
    print(f"Error: {e}")
