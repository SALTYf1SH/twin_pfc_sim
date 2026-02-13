# -*- coding: utf-8 -*-
"""
Parameter Sampler for HQH Simulation (New Working Condition)

This script generates a set of varied configuration files for running batch PFC
simulations based on the 'config_hqh_base.json' file. 

It uses Latin Hypercube Sampling (LHS) to efficiently sample the parameter 
space for the 23-layer model (Excavation Group 11).
"""

import json
import os
import numpy as np
from scipy.stats import qmc
import copy

# --- Configuration ---
# 1. Base config file for the new HQH working condition
BASE_CONFIG_FILE = "config_hqh_base.json" 

# 2. New output directory for these samples
OUTPUT_DIR = "configs_to_run_hqh"

NUM_SAMPLES = 200  # Number of configuration files to generate
SAMPLING_RANGE_PERCENTAGE = 0.50  # e.g., 0.50 means +/- 50% from the base value

# --- Parameter Space Definition ---
# Paths are defined for the new 23-layer model (EXCAVATION_LAYER_GROUP: "11")
# based on the 5 rock types in 'config_hqh_base.json'.
PARAMETERS_TO_VARY = {
    # --- Material Properties ---
    # Sandstone (TYPE 0)
    "sandstone_emod": (("ROCK_PARA", 0, 3, 1), None),
    "sandstone_pb_ten": (("ROCK_PARA", 0, 4, 1), None),
    "sandstone_pb_coh": (("ROCK_PARA", 0, 5, 1), None),
    "sandstone_kratio": (("ROCK_PARA", 0, 7, 1), None),

    # Gritstone (TYPE 1) - Key Stratum Material
    "gritstone_emod": (("ROCK_PARA", 1, 3, 1), None),
    "gritstone_pb_ten": (("ROCK_PARA", 1, 4, 1), None),
    "gritstone_pb_coh": (("ROCK_PARA", 1, 5, 1), None),
    "gritstone_kratio": (("ROCK_PARA", 1, 7, 1), None),

    # Mudstone/Siltstone (TYPE 2)
    "mudstone_emod": (("ROCK_PARA", 2, 3, 1), None),
    "mudstone_pb_ten": (("ROCK_PARA", 2, 4, 1), None),
    "mudstone_pb_coh": (("ROCK_PARA", 2, 5, 1), None),

    # Coal (TYPE 3) - Excavation Material
    "coal_emod": (("ROCK_PARA", 3, 3, 1), None),
    "coal_pb_ten": (("ROCK_PARA", 3, 4, 1), None),
    "coal_pb_coh": (("ROCK_PARA", 3, 5, 1), None),

    # --- Layer Thicknesses ---
    # (L17) Main KS (30m Gritstone) - varying middle part (index 5)
    "main_ks_thickness": (("ROCK_LAYER_THICKNESSES", 5), None),
    
    # (L11) Coal Seam (3-1 Coal) - (index 12)
    "coal_seam_thickness": (("ROCK_LAYER_THICKNESSES", 12), None),
    
    # (L10) Immediate Floor (Mudstone) - (index 13)
    "immediate_floor_thickness": (("ROCK_LAYER_THICKNESSES", 13), None),
}

def get_value_from_path(config_dict, path):
    """Helper function to retrieve a value from a nested dict/list using a path tuple."""
    value = config_dict
    for key in path:
        value = value[key]
    return value

def set_value_from_path(config_dict, path, new_value):
    """Helper function to set a value in a nested dict/list using a path tuple."""
    obj = config_dict
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = new_value

def generate_samples():
    """
    Main function to perform LHS and generate configuration files.
    """
    print("======================================================")
    print("      PFC Configuration Sampler (HQH Condition)     ")
    print("======================================================")

    # 1. Load Base Configuration
    try:
        with open(BASE_CONFIG_FILE, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        print(f"INFO: Successfully loaded base configuration from '{BASE_CONFIG_FILE}'")
    except FileNotFoundError:
        print(f"FATAL ERROR: Base configuration file '{BASE_CONFIG_FILE}' not found.")
        print("Please ensure you have saved the new config as 'config_hqh_base.json'.")
        return
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not parse '{BASE_CONFIG_FILE}'. Invalid JSON.")
        return

    # 2. Create Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"INFO: Created output directory '{OUTPUT_DIR}'")

    # 3. Define Sampling Bounds
    lower_bounds = []
    upper_bounds = []
    param_names = list(PARAMETERS_TO_VARY.keys())
    
    print("\nINFO: Defining parameter sampling space...")
    for name in param_names:
        path, _ = PARAMETERS_TO_VARY[name]
        try:
            base_value = get_value_from_path(base_config, path)
            PARAMETERS_TO_VARY[name] = (path, base_value) # Store the base value
            
            lower_bound = base_value * (1 - SAMPLING_RANGE_PERCENTAGE)
            upper_bound = base_value * (1 + SAMPLING_RANGE_PERCENTAGE)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            print(f"  - {name}: Varying from {lower_bound:.2e} to {upper_bound:.2e} (Base: {base_value:.2e})")
        except Exception as e:
            print(f"FATAL ERROR: Could not find path {path} for parameter '{name}' in the base config.")
            print(f"Error: {e}")
            return

    # 4. Perform Latin Hypercube Sampling
    dimension = len(param_names)
    sampler = qmc.LatinHypercube(d=dimension)
    samples = sampler.random(n=NUM_SAMPLES)
    
    # Scale samples from [0, 1] to the defined parameter ranges
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    print(f"\nINFO: Generated {NUM_SAMPLES} samples using LHS.")

    # 5. Generate and Save New Configuration Files
    for i in range(NUM_SAMPLES):
        new_config = copy.deepcopy(base_config)
        
        # Update the experiment name to be unique
        base_name = new_config.get("EXPERIMENT_NAME", "hqh_exp")
        new_config["EXPERIMENT_NAME"] = f"{base_name}_sample_{i+1:04d}"

        for j, name in enumerate(param_names):
            path, _ = PARAMETERS_TO_VARY[name]
            new_value = scaled_samples[i, j]
            
            # For numeric types, round to a reasonable precision
            if isinstance(new_value, float):
                # Heuristic for precision: keep a few significant digits
                new_value = float(f"{new_value:.4e}")

            set_value_from_path(new_config, path, new_value)
            
        # Save the new config file
        output_filename = os.path.join(OUTPUT_DIR, f"config_sample_{i+1:04d}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=4)

    print(f"\nSUCCESS: Successfully generated {NUM_SAMPLES} configuration files in '{OUTPUT_DIR}'.")
    print("======================================================")


if __name__ == "__main__":
    generate_samples()