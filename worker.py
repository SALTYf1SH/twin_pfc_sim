# -*- coding: utf-8 -*-
"""
Worker Script for Distributed PFC Simulations

This script is the core of the distributed computing system. It should be run
on every computer (worker) that will participate in the simulations, including
the main computer itself.

Each worker will autonomously pick up a simulation job from a shared directory,
execute it, and report the result until no jobs are left.
"""

import os
import json
import shutil
import subprocess
import time
import random
import socket
import sys

# --- Load Worker Configuration ---
def load_worker_config():
    config_path = 'worker_config.json'
    if not os.path.exists(config_path):
        print(f"FATAL ERROR: Worker configuration file '{config_path}' not found.")
        print("Please copy 'worker_config.json.template' to 'worker_config.json' and edit the PFC executable path.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not read or parse '{config_path}'. Error: {e}")
        sys.exit(1)

worker_config = load_worker_config()
PFC_EXECUTABLE_PATH = worker_config.get("PFC_EXECUTABLE_PATH")

if not PFC_EXECUTABLE_PATH or not os.path.exists(PFC_EXECUTABLE_PATH):
    print(f"FATAL ERROR: PFC executable path '{PFC_EXECUTABLE_PATH}' from config is invalid or does not exist.")
    sys.exit(1)

# --- Configuration ---
# The project directory is assumed to be the current working directory from which
# this script is launched. This is robust for mapped network drives.
PROJECT_WORKING_DIR = os.getcwd()

# All other paths are now relative to the current working directory.
PFC_CALL_SCRIPT = "main3.py"
TARGET_CONFIG_FILE = "config_sd.json"

# Task state directories (relative to the project directory)
DIR_TODO = os.path.join(PROJECT_WORKING_DIR, "jobs_todo")
DIR_RUNNING = os.path.join(PROJECT_WORKING_DIR, "jobs_running")
DIR_DONE = os.path.join(PROJECT_WORKING_DIR, "jobs_done")
DIR_FAILED = os.path.join(PROJECT_WORKING_DIR, "jobs_failed")

# --- Worker Settings ---
WORKER_ID = f"{socket.gethostname()}_{os.getpid()}" # Unique ID for this worker process
POLL_INTERVAL_SECONDS = 5 # How often to check for new jobs

def claim_job():
    """
    Atomically claims a job from the 'todo' directory.
    Moves a random job from 'todo' to 'running'.
    Returns the path to the claimed job config, or None if no jobs are available.
    """
    try:
        todo_files = [f for f in os.listdir(DIR_TODO) if f.endswith('.json')]
        if not todo_files:
            return None

        # Pick a random job to reduce the chance of multiple workers trying for the same file
        job_filename = random.choice(todo_files)
        
        source_path = os.path.join(DIR_TODO, job_filename)
        # Add worker ID to the running filename for better tracking
        running_filename = f"{os.path.splitext(job_filename)[0]}_{WORKER_ID}.json"
        destination_path = os.path.join(DIR_RUNNING, running_filename)

        # The atomic move operation
        os.rename(source_path, destination_path)
        
        print(f"[{WORKER_ID}] Claimed job: {job_filename}")
        return destination_path

    except FileNotFoundError:
        # This can happen if another worker claimed the file in the tiny window
        # between listing files and trying to move one. It's normal.
        return None
    except Exception as e:
        print(f"[{WORKER_ID}] ERROR: Could not claim a job. Reason: {e}")
        return None

def run_simulation(job_config_path):
    """
    Runs a single PFC simulation for the given job configuration.
    Returns True on success, False on failure.
    """
    # 1. Overwrite the target configuration file
    target_config_path = os.path.join(PROJECT_WORKING_DIR, TARGET_CONFIG_FILE)
    try:
        shutil.copy(job_config_path, target_config_path)
    except IOError as e:
        print(f"[{WORKER_ID}] ERROR: Could not copy config file. Details: {e}")
        return False

    # 2. Construct and run the PFC command via a temporary .dat file
    runner_dat_content = f'''
program directory "{PROJECT_WORKING_DIR.replace(os.path.sep, '/')}"
program call "{PFC_CALL_SCRIPT}"
'''
    runner_dat_path = os.path.join(PROJECT_WORKING_DIR, f"temp_runner_{WORKER_ID}.dat")
    with open(runner_dat_path, "w") as f:
        f.write(runner_dat_content)

    command = f'"{PFC_EXECUTABLE_PATH}" call "{runner_dat_path}"' # Corrected command string

    print(f"[{WORKER_ID}] Executing PFC simulation...")
    print(f"  -> Config: {os.path.basename(job_config_path)}")
    
    try:
        # MODIFIED FOR DEBUGGING: All output from the subprocess will now be visible in the console.
        subprocess.run(
            command,
            cwd=PROJECT_WORKING_DIR,
            check=True,
            shell=True
        )
        print(f"[{WORKER_ID}] Simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{WORKER_ID}] ERROR: PFC simulation failed!")
        # The error from PFC/main3.py should have been printed directly to the console above.
        # The e.stderr attribute might be empty now, but we check just in case.
        if e.stderr:
            print("--- PFC Standard Error (if any was captured) ---")
            print(e.stderr.decode('utf-8', errors='ignore'))
            print("-------------------------------------------------")
        return False
    finally:
        # Clean up the temporary runner file
        if os.path.exists(runner_dat_path):
            os.remove(runner_dat_path)

def main_worker_loop():
    """
    The main infinite loop for the worker process.
    """
    print("======================================================")
    print(f"      PFC Distributed Worker Started ({WORKER_ID})      ")
    print("======================================================")
    print(f" -> Watching for jobs in: {os.path.abspath(DIR_TODO)}")
    print(" -> Press Ctrl+C to stop this worker.")

    while True:
        # 1. Try to claim a job
        job_path = claim_job()

        if job_path:
            # 2. If a job was claimed, run it
            start_time = time.time()
            success = run_simulation(job_path)
            duration = time.time() - start_time
            print(f"[{WORKER_ID}] Job processing finished in {duration:.2f} seconds.")

            # 3. Move the job to the appropriate final directory
            final_dir = DIR_DONE if success else DIR_FAILED
            try:
                final_path = os.path.join(final_dir, os.path.basename(job_path))
                os.rename(job_path, final_path)
                print(f"[{WORKER_ID}] Moved job to '{final_dir}' directory.")
            except Exception as e:
                print(f"[{WORKER_ID}] CRITICAL ERROR: Could not move completed job file! Reason: {e}")
            
            # Continue to the next loop iteration immediately to check for more work
            continue

        # 4. If no job was claimed, check if all work is done
        try:
            num_todo = len(os.listdir(DIR_TODO))
            num_running = len(os.listdir(DIR_RUNNING))
            if num_todo == 0 and num_running == 0:
                print(f"\n[{WORKER_ID}] All jobs have been completed. Exiting.")
                break
        except FileNotFoundError:
            print(f"FATAL ERROR: Task directories not found. Please run setup_distributed_run.py first.")
            break
            
        # 5. Wait before polling again
        # print(f"[{WORKER_ID}] No jobs found. Waiting for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)

    print("======================================================")
    print(f"      Worker {WORKER_ID} Shutting Down      ")
    print("======================================================")


if __name__ == "__main__":
    try:
        main_worker_loop()
    except KeyboardInterrupt:
        print(f"\n[{WORKER_ID}] Worker stopped by user. Exiting.")
    except Exception as e:
        print(f"\n[{WORKER_ID}] A fatal error occurred in the worker loop: {e}")
        # Add traceback for debugging
        import traceback
        traceback.print_exc()