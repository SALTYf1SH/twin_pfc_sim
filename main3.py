# -*- coding: utf-8 -*-
"""
Refactored PFC simulation script for staged excavation of stratified rock mass.

This script automates a multi-stage PFC2D simulation. It reads all simulation
parameters from an external 'config.json' file for increased flexibility and
better experiment management.
"""

import csv
import itasca
from itasca import ball, wall
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import hashlib
import collections
import traceback

# Attempt to import utility functions from utils.py
# If this fails, ensure utils.py is in the same directory.
try:
    from utils import (run_dat_file, delete_balls_outside_area, fenceng,
                       get_avg_ball_y_disp, get_avg_ball_yy_stress, # <-- _MODIFIED_ Added get_avg_ball_yy_stress
                       plot_y_displacement_heatmap)
except ImportError:
    print("FATAL ERROR: Could not import from 'utils.py'.")
    print("Please ensure 'utils.py' is in the same directory as this script.")
    exit()

# ==============================================================================
# 0. CONFIGURATION LOADING
# ==============================================================================
def load_config(filepath="config_sd.json"):
    """
    Loads simulation parameters from a JSON configuration file.
    
    Args:
        filepath (str): The path to the configuration file.
    
    Returns:
        dict: A dictionary containing all simulation parameters.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"INFO: Successfully loaded configuration from '{filepath}'")
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at '{filepath}'.")
        print("Please ensure 'config.json' is in the same directory as main.py.")
        exit()
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Could not parse '{filepath}'. The file contains invalid JSON.")
        print(f"Error details: {e}")
        exit()


# ==============================================================================
# SIMULATION WORKFLOW FUNCTIONS
# ==============================================================================

def setup_environment(config):
    """
    Creates directories based on a parameter hash. If a directory exists,
    it prints a warning but proceeds with the simulation.
    """
    if not config.get("EXPERIMENT_NAME"):
        param_string = str(config.get("EQUILIBRIUM_PARAMS_LIST", ""))
        hasher = hashlib.md5()
        hasher.update(param_string.encode('utf-8'))
        experiment_name = hasher.hexdigest()
        config["EXPERIMENT_NAME"] = experiment_name
        
    exp_path = os.path.join(config["BASE_SAVE_PATH"], config["EXPERIMENT_NAME"])
    
    if os.path.exists(exp_path):
        print(f"WARNING: Result folder '{exp_path}' already exists. Files may be overwritten.")
    else:
        print(f"INFO: Creating new results folder: '{exp_path}'")
        
    paths = {
        "root": exp_path,
        "img": os.path.join(exp_path, "img"),
        "sav": os.path.join(exp_path, "sav"),
        "mat": os.path.join(exp_path, "mat"),
        "csv": os.path.join(exp_path, "csv"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    print(f"INFO: Results will be saved in '{exp_path}'")

    return paths

def save_parameters_to_file(config, folder_path):
    """Saves the configuration dictionary to a text file for record-keeping."""
    param_file_path = os.path.join(folder_path, "simulation_parameters.txt")
    with open(param_file_path, 'w') as f:
        f.write(f"Simulation Parameters for: {config['EXPERIMENT_NAME']}\n")
        f.write("="*40 + "\n")
        json.dump(config, f, indent=4)
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
        
        print("INFO: Setting FISH variables for jiaojie.dat...")
        if "EQUILIBRIUM_PARAMS_LIST" in config:
            for name, value in config["EQUILIBRIUM_PARAMS_LIST"]:
                itasca.fish.set(name, value)
                print(f"  -> Set FISH variable: {name} = {value}")
        
        run_dat_file(config["EQUILIBRIUM_DAT"])
        
        type_list = config["ROCK_TYPE"]
        para_list = config["ROCK_PARA"]
        for i in range(1,len(type_list)+1):
            para_type = type_list[i-1]
            paras = para_list[para_type]
            emod, pb_modules, kratio = 0, 0, 0
            for name, value in paras:
                if name not in ['NAME', 'TYPE']:
                    if name == 'emod': emod = value
                    if name == 'pb_modules': pb_modules = value
                    if name == 'kratio': kratio = value
                    if name in ['fric', 'pb_ten', 'pb_coh']:
                        command = f"contact property {name} {value} range group '{i}'"
                        itasca.command(command)
            command = f"contact method deform emod {emod} kratio {kratio} range group '{i}'"
            itasca.command(command)
            command = f"contact method pb_deform emod {pb_modules} kratio {kratio} range group '{i}'"
            itasca.command(command)
                    
        itasca.command(f"model save '{save_file}'")
        print(f"SUCCESS: Equilibrium model saved to '{save_file}'.")

def setup_monitoring_points(config, model_height):
    """Defines vertical monitoring sections on the model surface to track subsidence."""
    model_width = config["MODEL_WIDTH"]
    ypos_bottom_wall = itasca.wall.find('boxWallBottom1').pos_y()
    top_y_ref = ypos_bottom_wall + model_height
    print(f"INFO: Using theoretical top Y-position for monitoring: {top_y_ref:.2f}")

    rdmax = itasca.fish.get('rdmax')
    model_x_min = -model_width / 2.0
    model_x_max = model_width / 2.0
    
    section_boundaries = [model_x_min, model_x_min + config["LEFT_PILLAR_WIDTH"]]
    current_x = model_x_min + config["LEFT_PILLAR_WIDTH"]
    while current_x + config["EXCAVATION_STEP_WIDTH"] < model_x_max - config["RIGHT_PILLAR_WIDTH"]:
        current_x += config["EXCAVATION_STEP_WIDTH"]
        section_boundaries.append(current_x)
    section_boundaries.append(model_x_max)

    y_search_min = top_y_ref - (rdmax * 2.0)
    all_top_balls = [b for b in ball.list() if b.pos_y() >= y_search_min]

    ball_objects_dict = {}
    for i in range(len(section_boundaries) - 1):
        x_min, x_max = section_boundaries[i], section_boundaries[i+1]
        section_balls = [b for b in all_top_balls if x_min <= b.pos_x() < x_max]
        if section_balls:
            ball_objects_dict[str(i)] = section_balls
            
    if not ball_objects_dict and len(all_top_balls) > 0:
         print("WARNING: Monitoring sections are defined, but no top balls fall within them.")
    elif not all_top_balls:
         print("WARNING: No top balls found for surface monitoring.")
                           
    print(f"INFO: Successfully defined {len(ball_objects_dict)} vertical monitoring sections.")
    return ball_objects_dict, section_boundaries, top_y_ref

# --- _ADDED_ ---
def setup_stress_monitoring_sections(config, section_boundaries):
    """Defines horizontal monitoring sections in the coal seam floor."""
    stress_objects_dict = {}
    try:
        # 1. Identify the floor layer group
        excavation_group_str = config['EXCAVATION_LAYER_GROUP']
        floor_group_int = int(excavation_group_str) - 1
        if floor_group_int <= 0:
            raise ValueError(f"Excavation group '{excavation_group_str}' is invalid, no floor layer below it.")
        floor_group_str = str(floor_group_int)
        print(f"INFO: Setting up stress monitoring for floor layer: group '{floor_group_str}'")

        # 2. Get all balls in this layer group
        all_floor_balls = [
            b for b in ball.list() 
            if b.in_group(floor_group_str, 'layer')
        ]
        
        if not all_floor_balls:
             print(f"WARNING: No balls found in floor layer group '{floor_group_str}'. Stress monitoring will be empty.")
             return {}

        # 3. Assign balls to their respective horizontal sections
        for i in range(len(section_boundaries) - 1):
            x_min, x_max = section_boundaries[i], section_boundaries[i+1]
            section_balls = [b for b in all_floor_balls if x_min <= b.pos_x() < x_max]
            if section_balls:
                stress_objects_dict[str(i)] = section_balls
        
        print(f"INFO: Successfully defined {len(stress_objects_dict)} horizontal stress monitoring sections in the floor.")

    except Exception as e:
        print(f"CRITICAL ERROR in setup_stress_monitoring_sections: {e}")
        traceback.print_exc()
        print("WARNING: Floor stress monitoring setup failed. Will skip stress collection.")
    
    return stress_objects_dict
# --- _END_ADDED_ ---

def calculate_fragment_properties(balls):
    """Calculates geometric properties for a single fragment given its constituent balls."""
    if not balls: return {}
    num_balls = len(balls)
    radii = np.array([b.radius() for b in balls])
    areas = np.pi * radii**2
    total_area = np.sum(areas)
    if total_area == 0: return {"num_balls": num_balls, "area": 0, "centroid_x": 0, "centroid_y": 0, "orientation": 0.0}

    positions_x = np.array([b.pos_x() for b in balls])
    positions_y = np.array([b.pos_y() for b in balls])
    centroid_x = np.sum(positions_x * areas) / total_area
    centroid_y = np.sum(positions_y * areas) / total_area
    return {"num_balls": num_balls, "area": total_area, "centroid_x": centroid_x, "centroid_y": centroid_y, "orientation": 0.0}

def process_fragment_evolution(step_number, paths, previous_ball_to_fragment_map, config):
    """Analyzes fragment evolution, saves properties and genealogy, and creates a plot."""
    print(f"    -> Processing fragment evolution for step {step_number}...")
    fragments = {}
    all_balls = list(itasca.ball.list())
    if not all_balls:
        print("    -> WARNING: No balls in model to process.")
        return {}
    for ball in all_balls:
        frag_id = ball.fragment()
        if frag_id not in fragments:
            fragments[frag_id] = []
        fragments[frag_id].append(ball)

    current_ball_to_fragment_map = {b.id(): b.fragment() for b in all_balls}
    all_fragments_properties = []
    for frag_id, balls in fragments.items():
        parent_id = -1
        if previous_ball_to_fragment_map:
            parent_ids_of_balls = [previous_ball_to_fragment_map.get(b.id(), -1) for b in balls]
            if parent_ids_of_balls:
                parent_id = collections.Counter(parent_ids_of_balls).most_common(1)[0][0]
        properties = calculate_fragment_properties(balls)
        properties['fragment_id'] = frag_id
        properties['parent_id'] = parent_id
        all_fragments_properties.append(properties)

    props_csv_file = os.path.join(paths["csv"], f"fragments_properties_step_{step_number}.csv")
    try:
        with open(props_csv_file, 'w', newline='') as f:
            fieldnames = ["fragment_id", "parent_id", "num_balls", "area", "centroid_x", "centroid_y", "orientation"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_fragments_properties)
        print(f"    -> Fragment properties saved to '{props_csv_file}'")
    except IOError as e:
        print(f"    -> ERROR: Could not write fragment properties CSV. {e}")
    
    # --- Save detailed ball data for the current step ---
    balls_data = []
    for ball in all_balls:
        balls_data.append({
            'x': ball.pos_x(),
            'y': ball.pos_y(),
            'radius': ball.radius(),
            'fragment_id': ball.fragment()
        })
    
    if balls_data:
        # Use the built-in csv module to avoid external dependencies in PFC
        balls_csv_file = os.path.join(paths["csv"], f"fragments_balls_step_{step_number}.csv")
        try:
            headers = balls_data[0].keys()
            with open(balls_csv_file, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=headers)
                dict_writer.writeheader()
                dict_writer.writerows(balls_data)
            print(f"    -> Ball data saved to '{balls_csv_file}'")
        except (IOError, IndexError) as e:
            print(f"    -> ERROR: Could not write ball data CSV. {e}")

    # 5. Create visualization with boundary fragments in grey
    plot_file = os.path.join(paths["img"], f"fragments_step_{step_number}.png")

    # a. Identify boundary fragments
    model_width = config["MODEL_WIDTH"]
    x_min, x_max = -model_width / 2.0, model_width / 2.0
    boundary_threshold = 20.0 # User-defined fixed threshold of 20m
    boundary_fragment_ids = set()
    for ball in all_balls:
        if ball.pos_x() < x_min + boundary_threshold or ball.pos_x() > x_max - boundary_threshold:
            boundary_fragment_ids.add(ball.fragment())

    # b. Assign colors
    from matplotlib.colors import to_rgba
    internal_frag_ids = sorted([fid for fid in fragments.keys() if fid not in boundary_fragment_ids])
    cmap = plt.get_cmap('tab20')

    # Pre-convert all colors to RGBA tuples to ensure the list is uniform
    color_map = {fid: cmap(i % 20) for i, fid in enumerate(internal_frag_ids)}
    grey_color = to_rgba('0.75')
    black_color = to_rgba('black')

    ball_colors = []
    for ball in all_balls:
        frag_id = ball.fragment()
        if frag_id in boundary_fragment_ids:
            ball_colors.append(grey_color)
        else:
            ball_colors.append(color_map.get(frag_id, black_color))

    # c. Plotting
    plt.figure(figsize=(15, 12))
    plt.scatter([b.pos_x() for b in all_balls], 
                [b.pos_y() for b in all_balls], 
                c=ball_colors, 
                s=1, 
                alpha=0.8)

    plt.xlabel("Horizontal Position (m)")
    plt.ylabel("Vertical Position (m)")
    plt.title(f"Fragment Visualization for Step {step_number} (Boundary Fragments in Grey)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"    -> Fragment visualization plot saved to '{plot_file}'")
    except IOError as e:
        print(f"    -> ERROR: Could not save fragment plot. {e}")
    plt.close()

    return current_ball_to_fragment_map


def process_fracture_data(step_number, paths):
    """
    Extracts and saves DFN fracture data (position and aperture) for the current step.
    """
    print(f"    -> Processing DFN fracture data for step {step_number}...")
    try:
        # Check if the DFN module and fractures exist to avoid errors
        if not hasattr(itasca, 'dfn') or not hasattr(itasca.dfn, 'fracture'):
            print("    -> INFO: DFN module not available in this PFC version. Skipping.")
            return

        fracture_list = list(itasca.dfn.fracture.list())
        if not fracture_list:
            print("    -> INFO: No DFN fractures found in the model for this step.")
            return

        fracture_properties = []
        for frac in fracture_list:
            fracture_properties.append({
                "pos_x": frac.pos_x(),
                "pos_y": frac.pos_y(),
                "aperture": frac.aperture()
            })

        csv_file = os.path.join(paths["csv"], f"fractures_step_{step_number}.csv")
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ["pos_x", "pos_y", "aperture"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fracture_properties)
        print(f"    -> Successfully saved data for {len(fracture_properties)} fractures to '" + csv_file + "'")

    except Exception as e:
        print(f"    -> ERROR: An unexpected error occurred while processing fracture data: {e}")


def run_excavation_simulation(config, paths, ball_objects_dict, section_boundaries, model_top_y, stress_objects_dict): # <-- _MODIFIED_ Added stress_objects_dict
    """Runs the main excavation loop, solving and recording data at each step."""
    model_width = config["MODEL_WIDTH"]
    start_x = config["LEFT_PILLAR_WIDTH"] - (model_width / 2.0)
    end_x = (model_width / 2.0) - config["RIGHT_PILLAR_WIDTH"]
    step_width = config["EXCAVATION_STEP_WIDTH"]
    num_steps = int((end_x - start_x) / step_width)
    
    y_disps_list = {}
    stress_data_list = {} # <-- _ADDED_
    previous_ball_to_fragment_map = {}
    
    print(f"\n--- Starting Excavation Simulation ({num_steps} steps) ---")

    # Define number of sections based on boundaries, this is the master count
    num_sections = len(section_boundaries) - 1

    for i in range(num_steps):
        excavation_pos = start_x + i * step_width
        excavation_end = excavation_pos + step_width
        print(f"--> Step {i+1}/{num_steps}: Excavating from {excavation_pos:.2f}m to {excavation_end:.2f}m...")
        
        cmd = (
               f"ball delete range group '{config['EXCAVATION_LAYER_GROUP']}' "
               f"pos-x {excavation_pos} {excavation_end}"
        )
        itasca.command(cmd)
        
        itasca.command(f"model solve cycle {config['SOLVE_CYCLES_PER_STEP']} or ratio-average {config['SOLVE_RATIO_TARGET']}")
        
        save_file = os.path.join(paths["sav"], f"step_{i}.sav")
        itasca.command(f"model save '{save_file}'")
        
        # --- Data Extraction Block ---
        # 1. Process fragment evolution
        print("    -> Computing fragments...")
        itasca.command("fragment compute")
        current_map = process_fragment_evolution(i + 1, paths, previous_ball_to_fragment_map, config)
        previous_ball_to_fragment_map = current_map
        
        # 2. Process DFN fracture data
        process_fracture_data(i + 1, paths)

        # 3. Record surface displacement
        # <-- _MODIFIED_ (Bug Fix): Use .get(str(k), []) for safe access
        y_disps = [get_avg_ball_y_disp(ball_objects_dict.get(str(k), [])) for k in range(num_sections)]
        y_disps_list[excavation_pos] = y_disps
        
        # 4. Record floor stress <-- _ADDED_
        try:
            # Use .get(str(k), []) for safe access
            stress_vals = [get_avg_ball_yy_stress(stress_objects_dict.get(str(k), [])) for k in range(num_sections)]
            stress_data_list[excavation_pos] = stress_vals
        except Exception as e:
            print(f"    -> ERROR: Failed to record stress data for step {i+1}. {e}")
        
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
    return y_disps_list, stress_data_list # <-- _MODIFIED_ Return both datasets

def save_results(config, paths, y_disps_list, section_boundaries):
    """Plots surface subsidence curves and saves all displacement data to a CSV file."""
    if not y_disps_list:
        print("WARNING: y_disps_list is empty, cannot save results.")
        return
        
    monitoring_point_x_coords = []
    # <-- _MODIFIED_ (Bug Fix): Calculate x_coords based on boundaries, not on data length
    for i in range(len(section_boundaries) - 1):
         x_center = (section_boundaries[i] + section_boundaries[i+1]) / 2.0
         monitoring_point_x_coords.append(x_center)

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

    csv_file = os.path.join(paths["root"], "surface_y_disp_vs_section.csv")
    excavation_steps = list(y_disps_list.keys())
    header = ['Monitoring_Point_X_Position'] + [f'Excavated_to_{step:.2f}m' for step in excavation_steps]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        num_sections = len(monitoring_point_x_coords)
        for i in range(num_sections):
            row = [monitoring_point_x_coords[i]]
            for step in excavation_steps:
                # Ensure data exists for this step, though it should
                if step in y_disps_list and i < len(y_disps_list[step]):
                    row.append(y_disps_list[step][i])
                else:
                    row.append(None) # Add placeholder if data is missing
            writer.writerow(row)
            
    print(f"INFO: Subsidence data saved to '{csv_file}'")

# --- _ADDED_ ---
def save_stress_results(config, paths, stress_data_list, section_boundaries):
    """Plots floor stress curves and saves all stress data to a CSV file."""
    if not stress_data_list:
        print("WARNING: stress_data_list is empty, cannot save stress results.")
        return
        
    monitoring_point_x_coords = []
    for i in range(len(section_boundaries) - 1):
        x_center = (section_boundaries[i] + section_boundaries[i+1]) / 2.0
        monitoring_point_x_coords.append(x_center)

    plt.figure(figsize=(12, 7))
    for excavation_pos, stress_vals in stress_data_list.items():
        if len(monitoring_point_x_coords) == len(stress_vals):
            plt.plot(monitoring_point_x_coords, stress_vals, marker='o', linestyle='-',
                     markersize=4, label=f'Excavated to {excavation_pos:.2f} m')

    plt.xlabel('Horizontal Position (m)')
    plt.xlim(-config["MODEL_WIDTH"] / 2.0, config["MODEL_WIDTH"] / 2.0)
    plt.ylabel('Vertical Stress (Pa)')
    plt.title('Floor Vertical Stress Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_file = os.path.join(paths["img"], "floor_yy_stress_vs_section.png")
    plt.savefig(plot_file, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"INFO: Floor stress plot saved to '{plot_file}'")

    csv_file = os.path.join(paths["root"], "floor_yy_stress_vs_section.csv")
    excavation_steps = list(stress_data_list.keys())
    header = ['Monitoring_Point_X_Position'] + [f'Excavated_to_{step:.2f}m' for step in excavation_steps]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        num_sections = len(monitoring_point_x_coords)
        for i in range(num_sections):
            row = [monitoring_point_x_coords[i]]
            for step in excavation_steps:
                if step in stress_data_list and i < len(stress_data_list[step]):
                    row.append(stress_data_list[step][i])
                else:
                    row.append(None)
            writer.writerow(row)
            
    print(f"INFO: Floor stress data saved to '{csv_file}'")
# --- _END_ADDED_ ---


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
        itasca.command("model new")
        CONFIG = load_config()
        paths = setup_environment(CONFIG)
        save_parameters_to_file(CONFIG, paths["root"])
        
        layer_array, model_height = calculate_geology(CONFIG)
        itasca.command("python-reset-state false")
        layer_hights = CONFIG["ROCK_LAYER_THICKNESSES"]
        model_height = 0
        for i in range(len(layer_hights)):
            model_height += layer_hights[i]
        model_width = CONFIG["MODEL_WIDTH"]
        itasca.fish.set('height', model_height)
        itasca.fish.set('width', model_width)

        # --- Stage 1 & 2: Model Generation and Equilibrium ---
        run_stage_two_equilibrium(CONFIG, layer_array, paths)
        
        # --- Stage 3: Setup for Excavation ---
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")
        ball_objects_dict, section_boundaries, model_top_y = setup_monitoring_points(CONFIG, model_height)
        stress_objects_dict = setup_stress_monitoring_sections(CONFIG, section_boundaries) # <-- _ADDED_

        # --- Stage 4: Run Excavation Simulation ---
        # <-- _MODIFIED_ to get both datasets
        y_disps_list, stress_data_list = run_excavation_simulation(
            CONFIG, paths, 
            ball_objects_dict, section_boundaries, model_top_y, 
            stress_objects_dict
        )
        
        # --- Stage 5: Post-processing and Saving ---
        if y_disps_list:
            save_results(CONFIG, paths, y_disps_list, section_boundaries)
        else:
            print("WARNING: No excavation data was generated. Skipping results processing.")
            
        # <-- _ADDED_
        if stress_data_list:
            save_stress_results(CONFIG, paths, stress_data_list, section_boundaries)
        else:
            print("WARNING: No stress data was generated. Skipping stress results processing.")
        # --- _END_ADDED_
            
        print("\nSimulation finished successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR: An exception occurred during the simulation.")
        print(f"Error details: {e}")
        traceback.print_exc()
    
    print("======================================================")

if __name__ == "__main__":
    main()