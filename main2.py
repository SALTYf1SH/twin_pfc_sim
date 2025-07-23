# -*- coding: utf-8 -*-
"""
Refactored PFC simulation script for staged excavation of stratified rock mass.

This script automates a multi-stage PFC2D simulation based on a user-provided
working version. It has been organized for clarity and robustness while preserving
the original simulation logic.
"""

import csv
import itasca
from itasca import ball, wall
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import hashlib

# Attempt to import utility functions from utils.py
# If this fails, ensure utils.py is in the same directory.
try:
    from utils import (run_dat_file, delete_balls_outside_area, fenceng,
                       get_avg_ball_y_disp, plot_y_displacement_heatmap)
except ImportError:
    print("FATAL ERROR: Could not import from 'utils.py'.")
    print("Please ensure 'utils.py' is in the same directory as this script.")
    exit()


# ==============================================================================
# 0. GLOBAL CONFIGURATION
# ==============================================================================
# All simulation parameters are defined here for easy modification.
CONFIG = {
    # --- Simulation Control ---
    "DETERMINISTIC_MODE": True,
    # "EXPERIMENT_NAME" will be set automatically by timestamp.
    "EXPERIMENT_NAME": "", 

    # --- Model Geometry (units: meters) ---
    "MODEL_WIDTH": 250.0,
    "ROCK_LAYER_THICKNESSES": [
        20.50, 3.00, 7.50, 30.00, 8.00, 7.50, 6.50, 6.00, 4.50, 7.00,
        9.00, 6.50, 6.50, 7.50, 2.00, 9.00, 6.00, 3.00, 3.50, 5.50
    ],

    # --- Excavation Parameters (units: meters) ---
    "LEFT_PILLAR_WIDTH": 45.0,
    "RIGHT_PILLAR_WIDTH": 45.0,
    "EXCAVATION_STEP_WIDTH": 10.0,
    "EXCAVATION_LAYER_GROUP": '11',

    # === Parameters for Equilibrium Stage (jiaojie.dat) ===
    # These will be passed to PFC using `itasca.fish.set()`.
    "EQUILIBRIUM_PARAMS_LIST": [
        # Group 1
        ('pb_modules', 1e9),
        ('emod000', 15e9),
        ('ten_', 0.75e6),
        ('coh_', 0.75e6),
        ('fric', 0.1),
        ('kratio', 2.0),
        # Group 2 (Key stratum)
        ('key_pb_modules', 3e9),
        ('key_emod000', 45e9),
        ('key_ten_', 4.5e6),
        ('key_coh_', 4.5e6),
        ('key_fric', 0.3),
        ('key_kratio', 2.0),
        # Group 3
        ('pb_modules_1', 1e8),
        ('emod111', 1e8),
        ('ten1_', 1e5),
        ('coh1_', 1e5),
        ('fric1', 0.1),
        ('kratio', 2.0),
        # Group 4 (Damping)
        ('dpnr', 0.5),
        ('dpsr', 0.0),
    ],

    # --- Solver Settings ---
    "SOLVE_CYCLES_PER_STEP": 8000, # Fixed number of cycles per excavation step
    "SOLVE_RATIO_TARGET": 1e-5,

    # --- File Paths (using os module) ---
    "BASE_SAVE_PATH": "experiments",
    "INITIAL_MODEL_SAVE": "yuya.sav",
    "EQUILIBRIUM_MODEL_SAVE": "jiaojie.sav",
    "INITIAL_MODEL_DAT": "yuya-new.dat",
    "EQUILIBRIUM_DAT": "jiaojie.dat",
}


# ==============================================================================
# SIMULATION WORKFLOW FUNCTIONS
# ==============================================================================

def setup_environment(config):
    """
    Creates directories based on a parameter hash. If a directory exists,
    it prints a warning but proceeds with the simulation.
    """
    if not config["EXPERIMENT_NAME"]:
        param_string = str(config["EQUILIBRIUM_PARAMS_LIST"])
        hasher = hashlib.md5()
        hasher.update(param_string.encode('utf-8'))
        experiment_name = hasher.hexdigest()
        config["EXPERIMENT_NAME"] = experiment_name
        
    exp_path = os.path.join(config["BASE_SAVE_PATH"], config["EXPERIMENT_NAME"])
    
    # --- START OF MODIFICATION ---
    # Inform the user if the directory already exists.
    if os.path.exists(exp_path):
        print(f"WARNING: Result folder '{exp_path}' already exists. Files may be overwritten.")
    else:
        print(f"INFO: Creating new results folder: '{exp_path}'")
    # --- END OF MODIFICATION ---  
    paths = {
        "root": exp_path,
        "img": os.path.join(exp_path, "img"),
        "sav": os.path.join(exp_path, "sav"),
        "mat": os.path.join(exp_path, "mat"),
        "csv": os.path.join(exp_path, "csv"), # MODIFIED: Added csv path
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    print(f"INFO: Results will be saved in '{exp_path}'")
    return paths

def save_parameters_to_file(config, folder_path):
    """Saves the configuration dictionary to a text file."""
    param_file_path = os.path.join(folder_path, "simulation_parameters.txt")
    with open(param_file_path, 'w') as f:
        f.write(f"Simulation Parameters for: {config['EXPERIMENT_NAME']}\n")
        f.write("="*40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"INFO: Simulation parameters saved to '{param_file_path}'")

def calculate_geology(config):
    """Calculates model height and cumulative layer heights for PFC stratification."""
    thicknesses = config["ROCK_LAYER_THICKNESSES"].copy()
    model_height = sum(thicknesses)
    thicknesses.reverse() # Reverse for bottom-up modeling
    
    cumulative_heights = []
    current_height = 0
    for thickness in thicknesses:
        current_height += thickness
        cumulative_heights.append(round(current_height, 4))
        
    print(f"INFO: Model total height calculated as {model_height:.2f} m.")
    return cumulative_heights, model_height

def run_stage_one_generation(config, paths):
    """Generates the initial particle assembly (Stage 1: Yuya)."""
    save_file = os.path.join(paths["root"], config["INITIAL_MODEL_SAVE"])
    if os.path.exists(save_file):
        print(f"INFO: Found '{save_file}'. Skipping initial particle generation.")
        itasca.command(f"model restore '{save_file}'")
    else:
        print(f"INFO: '{save_file}' not found. Generating initial particle model.")
        itasca.command("model new")
        itasca.set_deterministic(config["DETERMINISTIC_MODE"])
        run_dat_file(config["INITIAL_MODEL_DAT"])
        
        # Temporary save before cleaning stray balls
        temp_save_file = os.path.join(paths["root"], "yuya_temp.sav")
        itasca.command(f"model save '{temp_save_file}'")
        
        delete_balls_outside_area(
            x_min=wall.find('boxWallLeft4').pos_x(),
            x_max=wall.find('boxWallRight2').pos_x(),
            y_min=wall.find('boxWallBottom1').pos_y(),
            y_max=wall.find('boxWallTop3').pos_y()
        )
        itasca.command(f"model save '{save_file}'")
        print(f"SUCCESS: Initial model saved to '{save_file}'.")

def run_stage_two_equilibrium(config, layer_array, paths):
    """Performs model stratification and calculates initial equilibrium (Stage 2: Jiaojie)."""
    save_file = os.path.join(paths["root"], config["EQUILIBRIUM_MODEL_SAVE"])
    if os.path.exists(save_file):
        print(f"INFO: Found '{save_file}'. Skipping stratification and equilibrium.")
        itasca.command(f"model restore '{save_file}'")
    else:
        print(f"INFO: '{save_file}' not found. Performing stratification and equilibrium.")
        
        # Step 2a: Stratify the model
        print("--> Step 2a: Stratifying model (fenceng)...")
        initial_save_file = os.path.join(paths["root"], config["INITIAL_MODEL_SAVE"])
        run_stage_one_generation(config, paths) # This ensures the file exists
        itasca.command(f"model restore '{initial_save_file}'")
        fenceng(layer_array=layer_array)
        
        fenceng_temp_file = os.path.join(paths["root"], "fenceng_temp.sav")
        itasca.command(f"model save '{fenceng_temp_file}'")
        
        # Step 2b: Calculate initial equilibrium
        print("--> Step 2b: Calculating initial equilibrium (jiaojie)...")
        itasca.command(f"model restore '{fenceng_temp_file}'")
        
        # *** FIX: Set FISH variables before calling the DAT file. ***
        print("INFO: Setting FISH variables for jiaojie.dat...")
        if "EQUILIBRIUM_PARAMS_LIST" in config:
            for name, value in config["EQUILIBRIUM_PARAMS_LIST"]:
                itasca.fish.set(name, value)
                print(f"  -> Set FISH variable: {name} = {value}")
        
        run_dat_file(config["EQUILIBRIUM_DAT"])
        itasca.command(f"model save '{save_file}'")
        print(f"SUCCESS: Equilibrium model saved to '{save_file}'.")

def setup_monitoring_points(config, model_height):
    """Defines vertical monitoring sections on the model surface to track subsidence."""
    model_width = config["MODEL_WIDTH"]
    
    # Use theoretical model height for a stable reference point
    ypos_bottom_wall = itasca.wall.find('boxWallBottom1').pos_y()
    top_y_ref = ypos_bottom_wall + model_height
    print(f"INFO: Using theoretical top Y-position for monitoring: {top_y_ref:.2f}")

    rdmax = itasca.fish.get('rdmax')
    
    model_x_min = -model_width / 2.0
    model_x_max = model_width / 2.0
    
    # Define boundaries for monitoring sections
    section_boundaries = [model_x_min, model_x_min + config["LEFT_PILLAR_WIDTH"]]
    current_x = model_x_min + config["LEFT_PILLAR_WIDTH"]
    while current_x + config["EXCAVATION_STEP_WIDTH"] < model_x_max - config["RIGHT_PILLAR_WIDTH"]:
        current_x += config["EXCAVATION_STEP_WIDTH"]
        section_boundaries.append(current_x)
    section_boundaries.append(model_x_max)

    # Find balls near the theoretical top surface
    y_search_min = top_y_ref - (rdmax * 2.0)
    all_top_balls = [b for b in ball.list() if b.pos_y() >= y_search_min]

    # Assign balls to their respective sections
    ball_objects_dict = {}
    for i in range(len(section_boundaries) - 1):
        x_min, x_max = section_boundaries[i], section_boundaries[i+1]
        section_balls = [b for b in all_top_balls if x_min <= b.pos_x() < x_max]
        if section_balls:
            ball_objects_dict[str(i)] = section_balls
            
    if not ball_objects_dict:
        raise RuntimeError("CRITICAL ERROR: Failed to define any surface monitoring sections.")
                           
    print(f"INFO: Successfully defined {len(ball_objects_dict)} vertical monitoring sections.")
    return ball_objects_dict, section_boundaries, top_y_ref

def run_excavation_simulation(config, paths, ball_objects_dict, section_boundaries, model_top_y):
    """Runs the main excavation loop, solving and recording data at each step."""
    model_width = config["MODEL_WIDTH"]
    start_x = config["LEFT_PILLAR_WIDTH"] - (model_width / 2.0)
    end_x = (model_width / 2.0) - config["RIGHT_PILLAR_WIDTH"]
    step_width = config["EXCAVATION_STEP_WIDTH"]
    num_steps = int((end_x - start_x) / step_width)
    
    y_disps_list = {}
    
    print(f"\n--- Starting Excavation Simulation ({num_steps} steps) ---")

    for i in range(num_steps):
        excavation_pos = start_x + i * step_width
        excavation_end = excavation_pos + step_width
        print(f"--> Step {i+1}/{num_steps}: Excavating from {excavation_pos:.2f}m to {excavation_end:.2f}m...")
        
        # Delete balls in the target layer and section
        cmd = (f"ball delete range group '{config['EXCAVATION_LAYER_GROUP']}' "
               f"pos-x {excavation_pos} {excavation_end}")
        itasca.command(cmd)
        
        # Solve to new equilibrium with a fixed number of cycles
        step_interval_cycles = config["SOLVE_CYCLES_PER_STEP"]
        itasca.command(f"model solve cycle {step_interval_cycles} or ratio-average {config['SOLVE_RATIO_TARGET']}")
        
        # Save model state for this step
        save_file = os.path.join(paths["sav"], f"step_{i}.sav")
        itasca.command(f"model save '{save_file}'")
        
        # Record average vertical displacement for each monitoring section
        sec_num = len(ball_objects_dict)
        y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(k)]) for k in range(sec_num)]
        y_disps_list[excavation_pos] = y_disps
        
        # Plot and save displacement heatmap
        model_plot_height = 160
        rdmax = itasca.fish.get('rdmax')
        plot_y_displacement_heatmap(
            window_size=rdmax * 2,
            model_width=model_width,
            model_height=model_plot_height,
            name=f"{excavation_pos:.2f}",
            interpolate='nearest',
            resu_path=paths["root"]
        )
    
    print("--- Excavation Simulation Complete ---")
    return y_disps_list

def save_results(config, paths, y_disps_list, section_boundaries):
    """Plots surface subsidence curves and saves all displacement data to a CSV file."""
    # Calculate the center x-coordinate for each monitoring section
    monitoring_point_x_coords = []
    # Check if y_disps_list is not empty before accessing its keys
    if not y_disps_list:
        print("WARNING: y_disps_list is empty, cannot save results.")
        return
        
    first_step_key = list(y_disps_list.keys())[0]
    num_sections_in_data = len(y_disps_list[first_step_key])

    for i in range(len(section_boundaries) - 1):
        if i < num_sections_in_data:
             x_center = (section_boundaries[i] + section_boundaries[i+1]) / 2.0
             monitoring_point_x_coords.append(x_center)

    # Plot Surface Subsidence Curves
    plt.figure(figsize=(12, 7))
    for excavation_pos, y_disps in y_disps_list.items():
        if len(monitoring_point_x_coords) == len(y_disps):
            plt.plot(monitoring_point_x_coords, y_disps, marker='o', linestyle='-',
                     markersize=4, label=f'Excavated to {excavation_pos:.2f} m')

    plt.xlabel('Horizontal Position (m)')
    plt.xlim(-config["MODEL_WIDTH"] / 2.0, config["MODEL_WIDTH"] / 2.0)
    plt.ylabel('Vertical Displacement (m)')
    plt.title('Surface Subsidence Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_file = os.path.join(paths["img"], "surface_y_disp_vs_section.png")
    plt.savefig(plot_file, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"INFO: Subsidence plot saved to '{plot_file}'")

    # Save Data to CSV File
    csv_file = os.path.join(paths["root"], "surface_y_disp_vs_section.csv")
    excavation_steps = list(y_disps_list.keys())
    header = ['Monitoring_Point_X_Position'] + [f'Excavated_to_{step:.2f}m' for step in excavation_steps]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Transpose data: each row is a monitoring point, each column is an excavation step
        num_sections = len(monitoring_point_x_coords)
        for i in range(num_sections):
            row = [monitoring_point_x_coords[i]]
            for step in excavation_steps:
                row.append(y_disps_list[step][i])
            writer.writerow(row)
            
    print(f"INFO: Subsidence data saved to '{csv_file}'")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to run the entire PFC simulation workflow."""
    print("======================================================")
    print("      PFC Stratified Rock Excavation Simulation       ")
    print("======================================================")
    try:
        # --- Stage 0: Setup ---
        paths = setup_environment(CONFIG)
        save_parameters_to_file(CONFIG, paths["root"])
        
        layer_array, model_height = calculate_geology(CONFIG)
        itasca.command("python-reset-state false")
        itasca.command("model new")

        # --- Stage 1 & 2: Model Generation and Equilibrium ---
        run_stage_two_equilibrium(CONFIG, layer_array, paths)
        
        # --- Stage 3: Setup for Excavation ---
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")
        ball_objects_dict, section_boundaries, model_top_y = setup_monitoring_points(CONFIG, model_height)

        # --- Stage 4: Run Excavation Simulation ---
        y_disps_list = run_excavation_simulation(CONFIG, paths, ball_objects_dict, section_boundaries, model_top_y)
        
        # --- Stage 5: Post-processing and Saving ---
        if y_disps_list:
            save_results(CONFIG, paths, y_disps_list, section_boundaries)
        else:
            print("WARNING: No excavation data was generated. Skipping results processing.")
            
        print("\nSimulation finished successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR: An exception occurred during the simulation.")
        print(f"Error details: {e}")
        # Add a traceback for more detailed debugging info
        import traceback
        traceback.print_exc()
    
    print("======================================================")

if __name__ == "__main__":
    main()
