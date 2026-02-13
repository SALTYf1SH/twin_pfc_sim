# -*- coding: utf-8 -*-
"""
Script to perform t-SNE visualization on the conditional vectors (rock parameters)
of the dataset.

This script will:
1. Load all data samples from the `final_dataset` directory.
2. Extract the first 11 elements (the conditional vector) from each sample.
3. Use scikit-learn's t-SNE to reduce the dimensionality of these vectors to 2D.
4. Create a scatter plot of the 2D vectors, colored by a specific parameter
   (e.g., sandstone elastic modulus) to reveal learned structures.
"""

import os
import glob
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---

# Directory where the final processed dataset is saved.
DATASET_DIR = "final_dataset"
OUTPUT_DIR = "evaluation_results"

# The index of the parameter to use for coloring the plot.
# From data_extractor.py, we know:
# 0: sandstone_emod
# 1: sandstone_pb_ten
# 2: sandstone_pb_coh
# 3: sandstone_kratio
# 4: mudstone_emod
# ... and so on.
COLOR_BY_PARAM_INDEX = 0 
COLOR_BY_PARAM_NAME = "Sandstone Elastic Modulus (emod)"

NUM_PARAMS = 11

def main():
    """
    Main function to run the t-SNE visualization process.
    """
    print("======================================================")
    print("        t-SNE Conditional Vector Visualizer         ")
    print("======================================================")

    if not os.path.isdir(DATASET_DIR):
        print(f"FATAL ERROR: Source directory '{DATASET_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Data ---
    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{DATASET_DIR}'.")
        return

    print(f"Found {len(all_files)} total samples. Loading conditional vectors...")

    # We only need the first step of each simulation sample, as the params are the same for all steps
    # This avoids unnecessary computation and memory usage.
    unique_sim_files = []
    seen_simulations = set()
    for f in all_files:
        sim_name = "_".join(os.path.basename(f).split('_')[:-2])
        if sim_name not in seen_simulations:
            seen_simulations.add(sim_name)
            unique_sim_files.append(f)
    
    print(f"Found {len(unique_sim_files)} unique simulation parameter sets.")

    conditional_vectors = []
    for file_path in tqdm(unique_sim_files, desc="Loading data"):
        with np.load(file_path) as data:
            # Extract the first NUM_PARAMS elements
            conditional_vectors.append(data['x'][:NUM_PARAMS])
    
    X = np.array(conditional_vectors)

    # --- 2. Perform t-SNE ---
    print("\nPerforming t-SNE dimensionality reduction...")
    print("This may take a few moments.")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print("t-SNE finished.")

    # --- 3. Plot Results ---
    print("Generating plot...")
    
    # Get the specific parameter values to use for coloring
    color_values = X[:, COLOR_BY_PARAM_INDEX]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=color_values, cmap='viridis', alpha=0.7)
    
    plt.title('t-SNE Visualization of Conditional Vector Space', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(COLOR_BY_PARAM_NAME)
    
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = os.path.join(OUTPUT_DIR, 'tsne_visualization.png')
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    plt.show()

    print("\n======================================================")
    print("                       Done!                        ")
    print("======================================================")

if __name__ == "__main__":
    main()
