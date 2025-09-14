# -*- coding: utf-8 -*-
"""
Batch Runner for PFC Simulations

This script automates the process of running multiple PFC simulations based on a
set of generated configuration files. It handles setting the correct working
directory, copying configurations, calling the PFC executable, and supports
resuming from the last completed simulation.

This is Step 3 in the data generation pipeline as outlined in roadmap.md.
"""

import os
import json
import shutil
import subprocess
import time

# --- Configuration ---

# The command to execute PFC. This should be the full path to the console executable.
# IMPORTANT: Use forward slashes or double backslashes for paths.
PFC_EXECUTABLE_PATH = "D:/Tools/pfc600/exe64/pfc2d600_console.exe"

# The directory where the project and main3.py are located.
# This script will ensure PFC runs within this directory.
PROJECT_WORKING_DIR = "F:/PFCprj/pfc_twin/twin_pfc_sim"

# The Python script that PFC should call.
PFC_CALL_SCRIPT = "main3.py"

# The directory containing the generated .json configuration files.
CONFIGS_SOURCE_DIR = "configs_to_run"

# The main configuration file that main3.py reads. This file will be overwritten.
TARGET_CONFIG_FILE = "config_sd.json"

# The base directory where simulation results are stored (defined in your config).
# This is used to check if a simulation has already been completed.
BASE_RESULTS_DIR = "experiments"


def run_batch_simulations():
    """
    Main function to discover and run all PFC simulations.
    """
    print("======================================================")
    print("            PFC Batch Simulation Runner             ")
    print("======================================================")

    # 1. Verify paths
    if not os.path.exists(PFC_EXECUTABLE_PATH):
        print(f"FATAL ERROR: PFC executable not found at '{PFC_EXECUTABLE_PATH}'")
        print("Please update the PFC_EXECUTABLE_PATH variable in this script.")
        return

    if not os.path.isdir(PROJECT_WORKING_DIR):
        print(f"FATAL ERROR: Project directory not found at '{PROJECT_WORKING_DIR}'")
        return

    config_files = sorted([f for f in os.listdir(CONFIGS_SOURCE_DIR) if f.endswith('.json')])
    if not config_files:
        print(f"FATAL ERROR: No configuration files found in '{CONFIGS_SOURCE_DIR}'")
        return

    total_sims = len(config_files)
    print(f"INFO: Found {total_sims} simulation configurations to run.")

    # 2. Main execution loop
    completed_sims = 0
    skipped_sims = 0
    failed_sims = 0

    for i, config_filename in enumerate(config_files):
        start_time = time.time()
        print(f"\n--- Processing Simulation {i+1}/{total_sims} ---")
        print(f"Config file: {config_filename}")

        # a. Load the sample config to get the experiment name for checking
        sample_config_path = os.path.join(CONFIGS_SOURCE_DIR, config_filename)
        with open(sample_config_path, 'r', encoding='utf-8') as f:
            sample_config = json.load(f)
        
        experiment_name = sample_config.get("EXPERIMENT_NAME")
        if not experiment_name:
            print(f"WARNING: 'EXPERIMENT_NAME' not found in {config_filename}. Cannot check for completion.")
            # Fallback to a generic check, though less reliable
            experiment_folder_path = None
        else:
            experiment_folder_path = os.path.join(BASE_RESULTS_DIR, experiment_name)

        # b. Check for resumability: skip if the result folder already exists
        if experiment_folder_path and os.path.exists(experiment_folder_path):
            print(f"INFO: Result folder '{experiment_folder_path}' already exists.")
            print("Skipping this simulation (assuming it was completed).")
            skipped_sims += 1
            continue

        # c. Overwrite the target configuration file
        target_config_path = os.path.join(PROJECT_WORKING_DIR, TARGET_CONFIG_FILE)
        try:
            shutil.copy(sample_config_path, target_config_path)
            print(f"INFO: Copied '{config_filename}' to '{TARGET_CONFIG_FILE}'")
        except IOError as e:
            print(f"ERROR: Could not copy configuration file. Skipping. Details: {e}")
            failed_sims += 1
            continue

        # d. Construct and run the PFC command
        # We provide the absolute path to the script to be safe
        script_to_call_abs_path = os.path.join(PROJECT_WORKING_DIR, PFC_CALL_SCRIPT)
        command = [
            PFC_EXECUTABLE_PATH,
            "call",
            script_to_call_abs_path
        ]

        print(f"INFO: Executing PFC...")
        print(f"  -> Command: {' '.join(command)}")
        print(f"  -> Working Directory: {PROJECT_WORKING_DIR}")

        try:
            # The key is to set the `cwd` (current working directory) argument.
            # This tells the subprocess to run as if it were started in that folder.
            process = subprocess.run(
                command,
                cwd=PROJECT_WORKING_DIR,
                check=True,        # Raises an exception if the command returns a non-zero exit code
                capture_output=True, # Captures stdout and stderr
                text=True          # Decodes stdout/stderr as text
            )
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"SUCCESS: Simulation completed in {duration:.2f} seconds.")
            # Optional: Print stdout for debugging if needed
            # print("--- PFC Output ---")
            # print(process.stdout)
            # print("------------------")
            completed_sims += 1

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"ERROR: PFC simulation failed after {duration:.2f} seconds with exit code {e.returncode}.")
            print("--- PFC Standard Output ---")
            print(e.stdout)
            print("--- PFC Standard Error ---")
            print(e.stderr)
            print("--------------------------")
            failed_sims += 1
        except FileNotFoundError:
            print(f"FATAL ERROR: Could not execute command. Is the path '{PFC_EXECUTABLE_PATH}' correct?")
            return # Stop the whole batch if the executable is not found
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"ERROR: An unexpected error occurred after {duration:.2f} seconds: {e}")
            failed_sims += 1

    print("\n======================================================")
    print("              Batch Simulation Summary              ")
    print("======================================================")
    print(f"Total Configurations Found: {total_sims}")
    print(f"Successfully Completed:     {completed_sims}")
    print(f"Skipped (Already Done):     {skipped_sims}")
    print(f"Failed:                     {failed_sims}")
    print("======================================================")


if __name__ == "__main__":
    run_batch_simulations()
