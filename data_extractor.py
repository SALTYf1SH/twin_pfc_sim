# -*- coding: utf-8 -*-
"""
Data Extractor for PFC Simulation Results

This script processes the raw output from completed PFC simulations and transforms
it into a structured dataset suitable for training a machine learning model.
It iterates through each simulation folder, and for each excavation step, it
creates a paired (X, Y) sample.

- Input (X): A 1D vector containing key rock properties and the surface subsidence vector.
- Output (Y): A 1D vector representing a flattened 2D grid of fracture intensity.

This is Step 4 in the data generation pipeline as outlined in roadmap.md.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---

# Directory where raw simulation results are stored.
SOURCE_EXPERIMENTS_DIR = "experiments"

# Directory where the final processed dataset will be saved.
FINAL_DATASET_DIR = "final_dataset"

# --- Grid Configuration for Output Vector (Y) ---
# Define the resolution of the grid to represent the fracture distribution.
GRID_RESOLUTION = (64, 64)  # (Width, Height) -> Results in a 64x64 grid

# --- Helper Functions ---

def get_simulation_parameters(sim_folder_path):
    """
    Reads the key varying parameters from the simulation's parameter file.
    This forms the first part of our input vector X.
    """
    params_to_extract = [
        "sandstone_emod", "sandstone_pb_coh",
        "mudstone_emod", "mudstone_pb_coh",
        "main_ks_thickness", "primary_ks_thickness"
    ]
    
    # We need to find the original config file to get the exact values
    # The parameter_sampler script saves them in a unique name.
    # We can also read the generated simulation_parameters.txt
    
    # Let's find the original config file by its unique name pattern
    config_path = os.path.join(sim_folder_path, "simulation_parameters.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find 'simulation_parameters.txt' in {sim_folder_path}")

    with open(config_path, 'r') as f:
        # This is a bit of a hacky way to read the JSON part of the text file
        content = f.read()
        json_str = content[content.find('{'):]
        config = json.loads(json_str)

    # Now extract the specific values we varied
    param_vector = []
    # Sandstone (TYPE 0) emod and pb_coh
    param_vector.append(config["ROCK_PARA"][0][3][1]) # emod
    param_vector.append(config["ROCK_PARA"][0][5][1]) # pb_coh
    # Mudstone/Siltstone (TYPE 2) emod and pb_coh
    param_vector.append(config["ROCK_PARA"][2][3][1]) # emod
    param_vector.append(config["ROCK_PARA"][2][5][1]) # pb_coh
    # Key Stratum Thicknesses (indices are 0-based)
    param_vector.append(config["ROCK_LAYER_THICKNESSES"][9])  # Layer #10 (Main KS)
    param_vector.append(config["ROCK_LAYER_THICKNESSES"][13]) # Layer #14 (Primary KS)
    
    return np.array(param_vector, dtype=np.float32)


def create_fracture_grid(fracture_csv_path, model_width, model_height):
    """
    Reads fracture data and converts it into a 2D grid representing fracture intensity.
    The intensity is based on the sum of apertures in each grid cell.
    """
    grid = np.zeros(GRID_RESOLUTION, dtype=np.float32)
    
    if not os.path.exists(fracture_csv_path):
        # It's possible a step has no fractures, return an empty grid
        return grid

    fracture_df = pd.read_csv(fracture_csv_path)
    if fracture_df.empty:
        return grid

    # Define the spatial boundaries of the model
    x_min, x_max = -model_width / 2.0, model_width / 2.0
    y_min, y_max = 0, model_height # Assuming model base is at y=0

    # Get fracture positions and apertures
    x_coords = fracture_df['pos_x'].values
    y_coords = fracture_df['pos_y'].values
    apertures = fracture_df['aperture'].values

    # Calculate grid cell indices for each fracture
    # np.floor is used to handle points exactly on the max boundary
    x_indices = np.floor((x_coords - x_min) / (x_max - x_min) * GRID_RESOLUTION[0]).astype(int)
    y_indices = np.floor((y_coords - y_min) / (y_max - y_min) * GRID_RESOLUTION[1]).astype(int)

    # Clip indices to be within the valid range [0, GRID_RESOLUTION-1]
    x_indices = np.clip(x_indices, 0, GRID_RESOLUTION[0] - 1)
    y_indices = np.clip(y_indices, 0, GRID_RESOLUTION[1] - 1)

    # Add apertures to the corresponding grid cells
    # np.add.at is a special function for unbuffered in-place addition,
    # which correctly handles multiple fractures falling into the same cell.
    np.add.at(grid, (y_indices, x_indices), apertures)

    return grid


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
            print(f"  - WARNING: Subsidence CSV not found. Skipping folder.")
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
    print("======================================================")
    print("          PFC Simulation Data Extractor             ")
    print("======================================================")

    if not os.path.isdir(SOURCE_EXPERIMENTS_DIR):
        print(f"FATAL ERROR: Source directory '{SOURCE_EXPERIMENTS_DIR}' not found.")
        return

    if not os.path.exists(FINAL_DATASET_DIR):
        os.makedirs(FINAL_DATASET_DIR)
        print(f"INFO: Created output directory '{FINAL_DATASET_DIR}'")

    # Find all completed simulation folders
    sim_folders = [
        os.path.join(SOURCE_EXPERIMENTS_DIR, d)
        for d in os.listdir(SOURCE_EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(SOURCE_EXPERIMENTS_DIR, d))
    ]

    if not sim_folders:
        print("WARNING: No simulation folders found to process.")
        return

    print(f"INFO: Found {len(sim_folders)} simulation folders to process.")
    
    total_samples = 0
    # Use tqdm for a nice progress bar
    for folder_path in tqdm(sim_folders, desc="Processing Simulations"):
        num_created = process_simulation_folder(folder_path, FINAL_DATASET_DIR)
        total_samples += num_created

    print("\n======================================================")
    print("               Extraction Summary                   ")
    print("======================================================")
    print(f"Total simulation folders processed: {len(sim_folders)}")
    print(f"Total (X, Y) data samples created:  {total_samples}")
    print(f"Dataset saved in: '{FINAL_DATASET_DIR}'")
    print("======================================================")


if __name__ == "__main__":
    main()
