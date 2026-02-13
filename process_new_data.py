# -*- coding: utf-8 -*-
"""
Process New Data and Merge into Final Dataset

This script processes raw PFC simulation results from a specified directory and
merges them into the existing final dataset. It is a modified version of the
original data_extractor.py, designed to be pointed at a new source directory
of raw experiment data.

It iterates through each simulation folder in the provided source directory,
and for each excavation step, it creates a paired (X, Y) sample which is then
saved as a .npz file in the project's standard final_dataset directory.

- Input (X): A 1D vector containing key rock properties and the surface subsidence vector.
- Output (Y): A 1D vector representing a flattened 2D grid of fracture intensity.

How to run:
python process_new_data.py Z:\pfc_2d_sim
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
import argparse

# --- Configuration ---


# Directory where the final processed dataset will be saved.
FINAL_DATASET_DIR = "final_dataset"

# --- Grid Configuration for Output Vector (Y) ---

# Define the resolution of the grid to represent the fracture distribution.
GRID_RESOLUTION = (64, 64)  # (Width, Height) -> Results in a 64x64 grid

# --- Helper Functions (Copied from data_extractor.py for consistency) ---

def get_simulation_parameters(sim_folder_path):
    """
    Reads the key varying parameters from the simulation's parameter file.
    This forms the first part of our input vector X.
    """
    config_path = os.path.join(sim_folder_path, "simulation_parameters.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find 'simulation_parameters.txt' in {sim_folder_path}")

    with open(config_path, 'r') as f:
        # This is a bit of a hacky way to read the JSON part of the text file
        content = f.read()
        json_str = content[content.find('{'):]
        config = json.loads(json_str)

    # Extract all 11 varied parameters to form the feature vector
    param_vector = []
    
    # Sandstone (TYPE 0) properties
    param_vector.append(config["ROCK_PARA"][0][3][1]) # sandstone_emod
    param_vector.append(config["ROCK_PARA"][0][4][1]) # sandstone_pb_ten
    param_vector.append(config["ROCK_PARA"][0][5][1]) # sandstone_pb_coh
    param_vector.append(config["ROCK_PARA"][0][7][1]) # sandstone_kratio

    # Mudstone/Siltstone (TYPE 2) properties
    param_vector.append(config["ROCK_PARA"][2][3][1]) # mudstone_emod
    param_vector.append(config["ROCK_PARA"][2][4][1]) # mudstone_pb_ten
    param_vector.append(config["ROCK_PARA"][2][5][1]) # mudstone_pb_coh
    param_vector.append(config["ROCK_PARA"][2][7][1]) # mudstone_kratio

    # Key Stratum Thicknesses
    param_vector.append(config["ROCK_LAYER_THICKNESSES"][9])  # main_ks_thickness
    param_vector.append(config["ROCK_LAYER_THICKNESSES"][13]) # primary_ks_thickness
    param_vector.append(config["ROCK_LAYER_THICKNESSES"][2])  # coal_seam_thickness
    
    return np.array(param_vector, dtype=np.float32)

def create_fracture_grid(fracture_csv_path, model_width, model_height):
    """
    Reads fracture data and uses Kernel Density Estimation (KDE) to create
    a smooth 2D grid representing fracture intensity. The aperture of each
    fracture is used as the weight for the KDE.
    """
    # Define the grid to evaluate the KDE on
    grid_x, grid_y = np.mgrid[0:GRID_RESOLUTION[0], 0:GRID_RESOLUTION[1]]

    if not os.path.exists(fracture_csv_path):
        return np.zeros(GRID_RESOLUTION, dtype=np.float32)

    fracture_df = pd.read_csv(fracture_csv_path)
    # KDE requires at least 2 points to estimate density
    if fracture_df.empty or len(fracture_df) < 2:
        return np.zeros(GRID_RESOLUTION, dtype=np.float32)

    # Define the spatial boundaries of the model
    x_min, x_max = -model_width / 2.0, model_width / 2.0
    y_min, y_max = 0, model_height # Assuming model base is at y=0

    # Get fracture positions and apertures
    x_coords = fracture_df['pos_x'].values
    y_coords = fracture_df['pos_y'].values
    apertures = fracture_df['aperture'].values

    # Normalize coordinates to the grid index space [0, GRID_RESOLUTION]
    # This is crucial for evaluating the KDE on the grid later
    norm_x = (x_coords - x_min) / (x_max - x_min) * GRID_RESOLUTION[0]
    norm_y = (y_coords - y_min) / (y_max - y_min) * GRID_RESOLUTION[1]
    
    # Combine into a (2, N) dataset as required by gaussian_kde
    dataset = np.vstack([norm_x, norm_y])
    
    # Perform weighted Kernel Density Estimation
    try:
        kde = gaussian_kde(dataset, weights=apertures)
        
        # Evaluate the KDE on the grid positions
        positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
        grid = np.reshape(kde(positions).T, GRID_RESOLUTION)

    except (np.linalg.LinAlgError, ValueError) as e:
        # This can happen if all points are co-linear or there are not enough points
        print(f"  - WARNING: KDE failed for {os.path.basename(fracture_csv_path)}. Reason: {e}. Returning zero grid.")
        return np.zeros(GRID_RESOLUTION, dtype=np.float32)

    # Normalize the resulting grid to a [0, 1] range for consistency
    if grid.max() > 0:
        grid = grid / grid.max()

    return grid.astype(np.float32)

def process_simulation_folder(sim_folder_path, output_dir):
    """
    Processes a single simulation folder, extracting all step data.
    """
    try:
        # 1. Get the fixed rock property vector for this simulation
        rock_prop_vector = get_simulation_parameters(sim_folder_path)
        
        # Load the config to get model dimensions
        with open(os.path.join(sim_folder_path, "simulation_parameters.txt"), 'r') as f:
            content = f.read()
            json_str = content[content.find('{'):]
            config = json.loads(json_str)
        model_width = config["MODEL_WIDTH"]
        model_height = sum(config["ROCK_LAYER_THICKNESSES"])

        # 2. Load the surface subsidence data for all steps
        subsidence_csv_path = os.path.join(sim_folder_path, "surface_y_disp_vs_section.csv")
        if not os.path.exists(subsidence_csv_path):
            print(f"  - WARNING: Subsidence CSV not found in {os.path.basename(sim_folder_path)}. Skipping folder.")
            return 0
        subsidence_df = pd.read_csv(subsidence_csv_path)
        # The first column is the monitoring point position, the rest are steps
        step_columns = subsidence_df.columns[1:]

        # 3. Iterate through each excavation step
        num_samples_created = 0
        for i, step_col_name in enumerate(step_columns):
            step_number = i + 1
            
            # a. Get the subsidence vector for the current step
            subsidence_vector = subsidence_df[step_col_name].values.astype(np.float32)
            
            # b. Combine to form the full input vector X
            input_vector_X = np.concatenate([rock_prop_vector, subsidence_vector])
            
            # c. Create the output vector Y from fracture data
            fracture_csv_path = os.path.join(sim_folder_path, "csv", f"fractures_step_{step_number}.csv")
            fracture_grid = create_fracture_grid(fracture_csv_path, model_width, model_height)
            output_vector_Y = fracture_grid.flatten() # Flatten the 2D grid to a 1D vector
            
            # d. Save the (X, Y) pair
            sim_name = os.path.basename(sim_folder_path)
            output_filename = f"{sim_name}_step_{step_number:03d}.npz"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Check for existence before overwriting
            if os.path.exists(output_filepath):
                print(f"  - INFO: File {output_filename} already exists. Skipping.")
                continue

            np.savez_compressed(output_filepath, x=input_vector_X, y=output_vector_Y)
            num_samples_created += 1
            
        return num_samples_created

    except Exception as e:
        print(f"  - ERROR: Failed to process folder '{os.path.basename(sim_folder_path)}'. Reason: {e}")
        return 0


def main():
    """
    Main function to run the data extraction process.
    """
    parser = argparse.ArgumentParser(description="Process new PFC simulation data and merge into the final dataset.")
    parser.add_argument("source_dir", type=str, help="The absolute path to the new directory containing raw simulation folders.")
    args = parser.parse_args()

    source_experiments_dir = args.source_dir

    print("======================================================")
    print("      Process and Merge New PFC Simulation Data       ")
    print("======================================================")
    print(f"Source Directory: {source_experiments_dir}")
    print(f"Output Directory: {FINAL_DATASET_DIR}")
    print("------------------------------------------------------")


    if not os.path.isdir(source_experiments_dir):
        print(f"FATAL ERROR: Source directory '{source_experiments_dir}' not found.")
        return

    if not os.path.exists(FINAL_DATASET_DIR):
        os.makedirs(FINAL_DATASET_DIR)
        print(f"INFO: Created output directory '{FINAL_DATASET_DIR}'")

    # Find all simulation folders in the new source directory
    sim_folders = [
        os.path.join(source_experiments_dir, d)
        for d in os.listdir(source_experiments_dir)
        if os.path.isdir(os.path.join(source_experiments_dir, d))
    ]

    if not sim_folders:
        print("WARNING: No simulation folders found to process in the source directory.")
        return

    print(f"INFO: Found {len(sim_folders)} simulation folders to process.")
    
    total_samples = 0
    # Use tqdm for a nice progress bar
    for folder_path in tqdm(sim_folders, desc="Processing New Simulations"):
        num_created = process_simulation_folder(folder_path, FINAL_DATASET_DIR)
        total_samples += num_created

    print("\n======================================================")
    print("               Processing Summary                   ")
    print("======================================================")
    print(f"Total simulation folders processed: {len(sim_folders)}")
    print(f"New (X, Y) data samples created:  {total_samples}")
    print(f"Dataset updated in: '{FINAL_DATASET_DIR}'")
    print("======================================================")


if __name__ == "__main__":
    main()
