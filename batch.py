import csv
import itasca
from itasca import ball, wall, ballarray
from itertools import product
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from utils import *

start_comb_idx = 0
exp_num = 100

deterministic_mode = False

# layer_array = [
#     5,  # coal layer height
# ]
# subsurface_level = 10   # set the height of the subsurface
first_section_length = 5   # set the length of the first section
sec_interval = 3   # set the length of each section

# set contact properties' ranges
# fric = 0.05    # friction coefficient, range: 0.0-1.0
# rfric = 0.0   # rolling friction coefficient, range: 0.0-1.0
# dpnr = 0.2   # normal damping coefficient, range: 0.0-1.0
# dpsr = 0.2    # shear damping coefficient, range: 0.0-1.0
# F0 = 1e5    # maximum attractive force at subsurface (N), range: 0-inf

opencut_sec = 5   # set the section where excavation starts
step_interval = 30000  # Define step interval

# Prevent Python state from resetting when issuing 'model new' or 'model restore'
itasca.command("python-reset-state false")

itasca.set_deterministic(deterministic_mode)

def run_simulation(params):
    try:
        fric, rfric, dpnr, dpsr, F0 = params
        # Create result path with contact properties and date
        resu_path = f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}'
        
        # Create main result directory and subdirectories
        if not os.path.exists(resu_path):
            os.makedirs(resu_path)
            os.makedirs(os.path.join(resu_path, 'img'))
            os.makedirs(os.path.join(resu_path, 'sav')) 
            os.makedirs(os.path.join(resu_path, 'mat'))

        # fenceng
        # itasca.command("model restore 'yuya'")
        # sec_num = fenceng(
        #     sec_interval=sec_interval, 
        #     layer_array=layer_array,
        #     first_section_length=first_section_length,
        #     subsurface_level=subsurface_level
        # )

        # pingheng
        itasca.command("model restore 'fenceng'")

        wall_up_pos_y = wall.find('boxWallTop3').pos_y()
        wlx, wly = compute_dimensions()

        # send contact properties to FISH
        itasca.fish.set('fric', fric)
        itasca.fish.set('rfric', rfric)
        itasca.fish.set('dpnr', dpnr)
        itasca.fish.set('dpsr', dpsr)
        itasca.fish.set('F0', F0)

        run_dat_file("pingheng-linear.dat")

        # kaiwa
        # Reset ball attributes
        itasca.command("ball attribute velocity 0 spin 0 displacement 0")

        # get a dict ball objects of each section near the ball at the top of the model
        top_ball_pos = get_balls_max_pos(1)
        rdmax = itasca.fish.get('rdmax')
        
        ball_objects_dict = {}
        for i in range(1, sec_num + 1):
            ball_objects_dict[str(i)] = get_balls_object_in_area(str(i), top_ball_pos-rdmax, top_ball_pos)

        # Print warning if top_ball_pos is larger than wall_up_pos_y
        if top_ball_pos > wall_up_pos_y * 1.1:
            print(f"Warning: the model is expanding, please check the model. Current top_ball_pos: {top_ball_pos}, model height: {wall_up_pos_y}")
        if top_ball_pos < wall_up_pos_y * 0.8:
            print(f"Warning: the model is shrinking, please check the model. Current top_ball_pos: {top_ball_pos}, model height: {wall_up_pos_y}")

        # delete empty list in ball_objects_dict
        ball_objects_dict = {k: v for k, v in ball_objects_dict.items() if v}
        
        sec_num = len(ball_objects_dict)

        y_disps_list = {}
        
        # Loop through sections
        for i in range(opencut_sec, sec_num-2):
            section_name = str(i)
            if first_section_length > 0:
                excavation_pos = first_section_length + (i-1) * sec_interval
            else:
                excavation_pos = i * sec_interval
            
            # Loop through all balls
            for ball_obj in list(ball.list()):  # Convert to list to avoid iterator issues
                if ball_obj.valid():  # Check if ball is still valid
                    if ball_obj.in_group('1', 'layer') and ball_obj.in_group(f'{section_name}', 'section'):
                        try:
                            ball_obj.delete()
                        except Exception as e:
                            print(f"Error deleting ball {ball_obj.id()}: {str(e)}")
            
            # Save model and solve
            if i < opencut_sec+2:
                itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-5")
            elif i < sec_num-3:
                itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-3")
            else:
                itasca.command(f"model solve ratio-average 1e-3")

            # get avg y disp of each section and plot the y disp vs section number
            y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(i)]) for i in range(1, sec_num + 1)]
            y_disps_list[excavation_pos] = y_disps

            plot_y_displacement_heatmap(window_size=rdmax * 2, model_width=wlx, model_height=wly, name=excavation_pos, interpolate='nearest', resu_path=resu_path)

        # plot every y disp vs section number
        plt.figure(figsize=(10, 6))
        for excavation_pos, y_disps in y_disps_list.items():
            plt.plot(range(1, sec_num + 1), y_disps, label=f'{excavation_pos}')
        plt.xlabel('Section Number')
        plt.ylabel('Average Y Displacement')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(resu_path, 'img', 'surface_y_disp_vs_section.png'), dpi=400, bbox_inches='tight')

        # save y_disps_list to csv
        with open(os.path.join(resu_path, 'mat', 'surface_y_disp_vs_section.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header row with step numbers
            writer.writerow(['Section'] + list(np.fromiter(y_disps_list.keys(), dtype=float)))
            # Write data rows
            for section in range(1, sec_num + 1):
                row = [section] + [y_disps_list[step][section-1] for step in y_disps_list]
                writer.writerow(row)
        
        return True
        
    except Exception as e:
        print(f"Error in simulation {resu_path}: {str(e)}")
        return False

def main(start_comb_idx, exp_num):
    # Define parameter ranges
    param_ranges = {
        'fric': np.linspace(0.0, 1.0, 4),   # 4 values from 0.0 to 1.0
        'rfric': np.linspace(0.0, 1.0, 4),  # 4 values from 0.0 to 1.0
        'dpnr': np.linspace(0.0, 1.0, 5),   # 5 values from 0.0 to 1.0
        'dpsr': np.linspace(0.0, 1.0, 5),   # 5 values from 0.0 to 1.0
        # 'F0': np.logspace(4, 6, 5)        # 5 values from 1e4 to 1e6
        'F0': [1e4, 1e6]
    }
    
    # Generate all combinations
    param_combinations = list(product(
        param_ranges['fric'],
        param_ranges['rfric'],
        param_ranges['dpnr'],
        param_ranges['dpsr'],
        param_ranges['F0']
    ))[start_comb_idx:start_comb_idx+exp_num]
    
    # Create log file
    log_file = 'grid_search.log'
    total_combinations = len(param_combinations)
    
    print(f"Starting grid search with {total_combinations} combinations")
    
    # Run all combinations
    start_time = time.time()
    for i, params in enumerate(param_combinations, 1):
        print(f"\nRunning combination {i}/{total_combinations}")
        print(f"Parameters: fric={params[0]}, rfric={params[1]}, dpnr={params[2]}, dpsr={params[3]}, F0={params[4]}")
        
        success = run_simulation(params)
        
        # Log progress
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] [INFO] Combination {i}/{total_combinations}\n")
            f.write(f"[{timestamp}] [PARAMS] fric={params[0]}, rfric={params[1]}, dpnr={params[2]}, dpsr={params[3]}, F0={params[4]}\n")
            f.write(f"[{timestamp}] [STATUS] {'Success' if success else 'Failed'}\n")
            f.write("-" * 50 + "\n")
        
        # Calculate and display estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_sim = elapsed_time / i
        remaining_time = avg_time_per_sim * (total_combinations - i)
        print(f"Estimated time remaining: {remaining_time/3600:.1f} hours")

if __name__ == "__main__":
    main(start_comb_idx, exp_num)
