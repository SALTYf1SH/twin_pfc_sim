import csv
import itasca
from itasca import ball, wall
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

# ==============================================================================
# 0. Global Settings
# ==============================================================================
deterministic_mode = True  # Set to True to ensure consistent results for each run

# ==============================================================================
# 1. Define Rock Strata Geology (Unit: meters)
# ==============================================================================
# Directly define the thickness of each rock layer (Unit: meters), the order is consistent with the table you provided (from top to bottom)
thicknesses_in_meters = [
    20.50, 3.00, 7.50, 30.00, 8.00, 7.50, 6.50, 6.00, 4.50, 
    7.00, 9.00, 6.50, 6.50, 7.50, 2.00, 9.00, 6.00, 3.00, 
    3.50, 5.50
]
model_height=0
for i in range(len(thicknesses_in_meters)):
    model_height += thicknesses_in_meters[i]

# *** Correction: Invert the rock layer list ***
# Your table is from top to bottom, while modeling requires it from bottom to top. Therefore, the list order is reversed.
thicknesses_in_meters.reverse()

# Automatically calculate the cumulative height, this will be the new layer_array
# This array defines the top elevation of each stratum from bottom to top (Unit: meters)
cumulative_heights = []
current_height = 0
for thickness in thicknesses_in_meters:
    current_height += thickness
    cumulative_heights.append(round(current_height, 4))

layer_array = cumulative_heights
# Based on your data, the total model height is about 1.6m, and the model width should be set to 2.5m in the .dat file

# ==============================================================================
# 2. Set Model and Simulation Parameters (Unit: meters)
# ==============================================================================

# --- Excavation Parameters (Unit: meters) ---
# Note: The total model width is 2.5m, please ensure the excavation parameters are reasonable
subsurface_level = 0.1    # Thickness of the surface soil layer (m)

# According to user requirements, excavation leaves a 45cm pillar on the left and a 45cm pillar on the right.
left_pillar_width = 45  # Width of the left coal pillar (m)
right_pillar_width = 45 # Width of the right coal pillar (m)

# Set the width of the left coal pillar as the length of the first section
first_section_length = left_pillar_width

# Define the width of each excavation step
sec_interval = 20  # Length of subsequent excavation blocks (m)

# --- Mechanical Parameters (Important Note!) ---
# The following parameters are example values and are not specific to your particular rock strata.
# A real simulation requires parameter calibration for each rock type (e.g., medium-grained sandstone, siltstone, etc.)
# to obtain a set of microscopic parameters that match the actual rock mechanical properties.
fric = 0.05     # Friction coefficient (dimensionless)
rfric = 0.0     # Rolling friction coefficient (dimensionless)
dpnr = 0.2      # Normal damping coefficient (dimensionless)
dpsr = 0.2      # Shear damping coefficient (dimensionless)

# --- Troubleshooting: Parameter adjustment for model shrinkage issues ---
# The "collapsing into a block" issue you encountered is caused by a mismatch between particle stiffness and cohesion.
# When the cohesive force F0 is very large and the particle stiffness emod is too small,
# the particles will be overly "compressed", leading to model collapse.
#
# Solution:
# 1. Open your .dat file (likely yuya-new.dat or pingheng-linear.dat).
# 2. Find the command that defines particle stiffness, usually "ball attribute emod ..." or within the "ball contact-model" command.
# 3. Significantly increase the value of emod. For example, if it is currently emod 1e6, try changing it to emod 1e7 or emod 1e8.
# 4. Save the .dat file and re-run this Python script.
#
# Increasing stiffness allows particles to effectively resist strong cohesive forces,
# thus maintaining model stability while preserving cohesion.

F0 = 1e5      # Maximum cohesive force in the surface layer (N) -> Restored to original value
D0 = 0.2      # Cohesion range in the surface layer (m)

# --- Simulation Flow Control ---
# *** Correction: Update the excavated coal seam group name ***
# Since the rock layer order has been reversed, the layer to be excavated has changed.
# The fenceng function in utils.py names layer groups starting from '1'.
excavation_layer_group = '11' # Excavate group '11', which corresponds to the "3-1 coal" seam.


model_width = 250# Total model width

# Automatically calculate the start and end sections for excavation
# Section 0 is the left pillar, so excavation starts from section 1
excavation_start_px = left_pillar_width-(model_width/2)
excavation_end_px = (model_width/2) - right_pillar_width
excavation_step = int((excavation_end_px-excavation_start_px)/sec_interval)

step_solve_time = 3   # Define the solving time after each excavation step (seconds)

def run_simulation(**params):

    # Create result path
    resu_path = f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}_{D0}'
    
    # Create result folders
    if not os.path.exists(resu_path):
        os.makedirs(resu_path)
        os.makedirs(os.path.join(resu_path, 'img'))
        os.makedirs(os.path.join(resu_path, 'sav'))
        os.makedirs(os.path.join(resu_path, 'mat'))

    # Prevent PFC from resetting the Python state
    itasca.command("python-reset-state false")

    itasca.command("model new")

    # Define checkpoint file paths in the current directory
    yuya_sav_path = 'yuya.sav'
    jiaojie_sav_path = 'jiaojie.sav'

    # ==============================================================================
    # Stage 1: Initial Particle Model Generation (Yuya)
    # ==============================================================================
    if os.path.exists(yuya_sav_path):
        print(f"INFO: Found '{yuya_sav_path}'. Skipping initial particle generation.")
        itasca.command(f"model restore '{yuya_sav_path}'")
    else:
        print(f"INFO: '{yuya_sav_path}' not found. Generating initial particle model.")
        itasca.command("model new")
        itasca.set_deterministic(deterministic_mode)

        # Generate initial particles using the DAT file
        run_dat_file("yuya-new.dat")
        itasca.command("model save 'yuya_temp'") # Temporary save before deleting stray balls

        # Delete balls outside the defined walls
        delete_balls_outside_area(
            x_min=wall.find('boxWallLeft4').pos_x(),
            x_max=wall.find('boxWallRight2').pos_x(),
            y_min=wall.find('boxWallBottom1').pos_y(),
            y_max=wall.find('boxWallTop3').pos_y()
        )
        
        # Save the final state for this stage
        itasca.command(f"model save '{yuya_sav_path}'")
        print(f"SUCCESS: Initial model generated and saved to '{yuya_sav_path}'.")

    # ==============================================================================
    # Stage 2: Stratification and Initial Equilibrium (Jiaojie)
    # ==============================================================================
    sec_num = 0  # Initialize section number
    if os.path.exists(jiaojie_sav_path):
        print(f"INFO: Found '{jiaojie_sav_path}'. Skipping stratification and equilibrium calculation.")
        itasca.command(f"model restore '{jiaojie_sav_path}'")

        # Recover the number of sections from the loaded model's groups
        sec_num = len(layer_array)
        
        if sec_num == 0:
            raise ValueError("ERROR: Model loaded, but no section groups ('0', '1', ...) were found. Check the file or delete it to recalculate.")
        print(f"INFO: Recovered {sec_num} sections from the loaded model.")

    else:
        print(f"INFO: '{jiaojie_sav_path}' not found. Performing stratification and equilibrium.")
        
        # Step 2a: Stratify and section the model
        print("--> Step 2a: Stratifying model (fenceng)...")
        # Ensure we start from the correct 'yuya' state
        itasca.command(f"model restore '{yuya_sav_path}'") 
        sec_num = fenceng(
            layer_array=layer_array
        )
        itasca.command("model save 'fenceng_temp'") # Save the stratified state

        # Step 2b: Calculate initial equilibrium
        print("--> Step 2b: Calculating initial equilibrium (jiaojie)...")
        itasca.command("model restore 'fenceng_temp'")
        
        # Pass mechanical properties to PFC via FISH variables
        itasca.fish.set('pb_modules', 1e9)
        itasca.fish.set('emod000', 15e9)
        itasca.fish.set('ten_', 1.5e6)
        itasca.fish.set('coh_', 1.5e6)
        itasca.fish.set('fric', 0.1)
        itasca.fish.set('kratio', 2.)
        itasca.fish.set('emod111', 1e9)
        itasca.fish.set('ten_', 2e5)
        itasca.fish.set('coh_', 2e5)
        itasca.fish.set('dpnr', 0.5)
        itasca.fish.set('dpsr', 0.0)

        # Run the equilibrium calculation script
        run_dat_file("jiaojie.dat")
        
        # Save the final equilibrium state to the current directory
        itasca.command(f"model save '{jiaojie_sav_path}'")
        print(f"SUCCESS: Equilibrium model calculated and saved to '{jiaojie_sav_path}'.")
        # Get model properties (the model is already loaded or in the correct state)
    # 4. Simulate excavation
    itasca.command(f"model restore '{os.path.join(resu_path, 'sav', 'jiaojie')}'")
    
    # Reset ball attributes like displacement
    itasca.command("ball attribute velocity 0 spin 0 displacement 0")

    # Get the top balls for monitoring surface subsidence
    top_ball_pos = get_balls_max_pos(1)
    wly = top_ball_pos
    wlx = 250
    wall_up_pos_y = 0
    for i in range(len(layer_array)):
        wall_up_pos_y += layer_array[i]
    # Check if the model is excessively expanding or contracting
    if top_ball_pos > wall_up_pos_y * 1.1:
        print(f"Warning: Model is expanding, please check parameters. Current top height {top_ball_pos}, model wall height: {wall_up_pos_y}")
    if top_ball_pos < wall_up_pos_y * 0.8:
        print(f"Warning: Model is contracting, please check parameters. Current top height {top_ball_pos}, model wall height: {wall_up_pos_y}")

    rdmax = itasca.fish.get('rdmax')
    
    # Store the top balls of each section in a dictionary for subsequent displacement monitoring
    # +++ This is the CORRECTED block +++
    # Define vertical monitoring sections based on x-coordinates to correctly measure surface subsidence profile.
    # NOTE: PFC model coordinates are often centered around 0. We assume the model spans from -wlx/2 to +wlx/2.
    model_x_min = -wlx / 2
    model_x_max = wlx / 2

    # Define the boundaries of each vertical monitoring section
    section_boundaries = [model_x_min, model_x_min + left_pillar_width]
    current_x = model_x_min + left_pillar_width
    while current_x + sec_interval < model_x_max - right_pillar_width:
        current_x += sec_interval
        section_boundaries.append(current_x)
    section_boundaries.append(model_x_max)

    # Get all balls near the model's surface
    y_search_min = top_ball_pos - rdmax * 2.0  # Use a slightly larger search depth
    all_top_balls = [b for b in itasca.ball.list() if b.pos_y() >= y_search_min]

    # Assign top balls to their respective vertical sections
    ball_objects_dict = {}
    for i in range(len(section_boundaries) - 1):
        x_min = section_boundaries[i]
        x_max = section_boundaries[i+1]
        
        # Find balls within the current vertical section's x-range
        section_balls = [b for b in all_top_balls if x_min <= b.pos_x() < x_max]
        
        if section_balls: # Only add sections that actually contain monitoring balls
            ball_objects_dict[str(len(ball_objects_dict))] = section_balls
    
    sec_num = len(ball_objects_dict)
    if sec_num == 0:
        raise RuntimeError("CRITICAL ERROR: Failed to define any surface monitoring sections. "
                        "Check model generation, `wlx`, and pillar width parameters.")

    print(f"INFO: Successfully defined {sec_num} vertical monitoring sections for surface subsidence.")

    y_disps_list = {}
    
    # Start the excavation loop (using the new start/end sections)

    for i in range(0, excavation_step):
        # 'excavation_pos' represents the x-coordinate of the current excavation face
        excavation_pos = excavation_start_px + i * sec_interval
        
        # Delete balls within the specified coal layer and section
        command = "ball delete range group '11' pos-x "+str(excavation_pos)+' '+str(excavation_pos+sec_interval)
        itasca.command(command)
        # Solve/Calculate
        itasca.command(f"model solve cycle 10")
        global_timestep = itasca.timestep()
        step_interval_cycles = int(step_solve_time / global_timestep)
        step_interval_cycles = 100

        itasca.command(f"model solve cycle {step_interval_cycles} or ratio-average 1e-5")
        itasca.command(f"model save '{os.path.join(resu_path, 'sav', str(i))}'")

        # Get the average vertical displacement for each section and record it
        y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(k)]) for k in range(0, sec_num)]
        y_disps_list[excavation_pos] = y_disps

        # Plot and save the displacement contour map for the current step
        plot_y_displacement_heatmap(window_size=rdmax * 2, model_width=wlx, model_height=wly, name=f"{excavation_pos:.2f}", interpolate='nearest', resu_path=resu_path)

    # 5. Result Output
    # Plot surface subsidence curves for all excavation steps
    plt.figure(figsize=(10, 6))
    # Calculate the center x-coordinate for each monitoring section for plotting.
    # This will be used for both the plot and the CSV output.
    monitoring_point_x_coords = []
    for i in range(len(section_boundaries) - 1):
        x_center = (section_boundaries[i] + section_boundaries[i+1]) / 2.0
        monitoring_point_x_coords.append(x_center)

    # Plot surface subsidence curves for all excavation steps
    plt.figure(figsize=(12, 7))
    for excavation_pos, y_disps in y_disps_list.items():
        # Ensure the data lengths match before plotting
        if len(monitoring_point_x_coords) == len(y_disps):
             plt.plot(monitoring_point_x_coords, y_disps, marker='o', linestyle='-', markersize=4, label=f'Excavated to {excavation_pos:.2f} m')

    plt.xlabel('Working Face Position (m)')
    plt.xlim(0, int(wlx))
    plt.ylabel('Vertical Displacement (m)')
    plt.title('Surface Subsidence Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
    plt.savefig(os.path.join(resu_path, 'img', 'surface_y_disp_vs_section.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # Save subsidence data to a CSV file
    with open(os.path.join(resu_path, 'surface_y_disp_vs_section.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Monitoring Point Position (m)'] + list(np.fromiter(y_disps_list.keys(), dtype=float)))
        for section in range(0, sec_num):
            if section == 0:
                x_position = first_section_length / 2 if first_section_length > 0 else sec_interval / 2
            elif section == sec_num - 1:
                x_position = int(wlx)
            else:
                x_position = first_section_length + (section - 1) * sec_interval + sec_interval / 2 if first_section_length > 0 else (section - 1) * sec_interval + sec_interval / 2
            row = [x_position] + [y_disps_list[step][section] for step in y_disps_list]
            writer.writerow(row)
    
    print(f"Simulation complete. Results saved to {resu_path}")
    return True
    
if __name__ == "__main__":
    run_simulation(
        deterministic_mode=deterministic_mode,
        fric=fric,
        rfric=rfric,
        dpnr=dpnr,
        dpsr=dpsr,
        F0=F0,
        D0=D0,
        step_solve_time=step_solve_time,
        layer_array=layer_array,
        first_section_length=first_section_length,
        subsurface_level=subsurface_level,
        sec_interval=sec_interval,
        excavation_layer_group=excavation_layer_group,
        excavation_step = excavation_step,
        excavation_start_px = excavation_start_px,
        excavation_end_px = excavation_end_px,
        wlx = model_width,
        wly = model_height,
        
    )