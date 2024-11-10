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

deterministic_mode = False

layer_array = [
    5,  # coal layer height
]
subsurface_level = 2   # set the height of the subsurface
first_section_length = 5   # set the length of the first section
sec_interval = 3   # set the length of each section

# set contact properties' ranges
fric = 0.05    # friction coefficient, range: 0.0-1.0
rfric = 0.0   # rolling friction coefficient, range: 0.0-1.0
dpnr = 0.2   # normal damping coefficient, range: 0.0-1.0
dpsr = 0.2    # shear damping coefficient, range: 0.0-1.0
F0 = 1e5    # maximum attractive force at subsurface (N), range: 0-inf

opencut_sec = 5   # set the section where excavation starts
step_interval = 30000  # Define step interval

# Prevent Python state from resetting when issuing 'model new' or 'model restore'
itasca.command("python-reset-state false")

itasca.set_deterministic(deterministic_mode)

def run_simulation(**params):

    # Create result path with contact properties and date
    resu_path = f'experiments/exp_{fric}_{rfric}_{dpnr}_{dpsr}_{F0}'
    
    # Create main result directory and subdirectories
    if not os.path.exists(resu_path):
        os.makedirs(resu_path)
        os.makedirs(os.path.join(resu_path, 'img'))
        os.makedirs(os.path.join(resu_path, 'sav')) 
        os.makedirs(os.path.join(resu_path, 'mat'))

    # yuya
    run_dat_file("yuya-new.dat")
    itasca.command("model save 'yuya'")

    # delete balls outside the wall
    delete_balls_outside_area(
        x_min=wall.find('boxWallLeft4').pos_x(),
        x_max=wall.find('boxWallRight2').pos_x(),
        y_min=wall.find('boxWallBottom1').pos_y(),
        y_max=wall.find('boxWallTop3').pos_y()
    )

    # fenceng
    itasca.command("model restore 'yuya'")
    sec_num = fenceng(
        sec_interval=sec_interval, 
        layer_array=layer_array,
        first_section_length=first_section_length,
        subsurface_level=subsurface_level
    )
    itasca.command("model save 'fenceng'")

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
    itasca.command("model save 'pingheng'")

    # kaiwa
    # Restore the model
    itasca.command("model restore 'pingheng'")
    
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
            itasca.command(f"model save '{section_name}'")
        elif i < sec_num-3:
            itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-3")
            itasca.command(f"model save '{section_name}'")
        else:
            itasca.command(f"model solve ratio-average 1e-3")
            itasca.command(f"model save '{section_name}'")

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
    
if __name__ == "__main__":
    run_simulation(
        fric=fric, 
        rfric=rfric, 
        dpnr=dpnr, 
        dpsr=dpsr, 
        F0=F0,
        opencut_sec=opencut_sec,
        step_interval=step_interval,
        layer_array=layer_array,
        first_section_length=first_section_length,
        subsurface_level=subsurface_level,
        sec_interval=sec_interval
    )
