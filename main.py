import itasca
from itasca import ball, wall, ballarray
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

# set contact properties
fric = 0.05    # friction coefficient
rfric = 0.0   # rolling friction coefficient
dpnr = 0.2   # normal damping coefficient
dpsr = 0.2    # shear damping coefficient
F0 = 1e2    # maximum attractive force (N)

# Define step interval
step_interval = 30000

sec_interval = 3
opencut_sec = 5

layer_array = [
    5,  # coal layer height
]
subsurface_level = 5

image_save_path = 'images'
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# Prevent Python state from resetting when issuing 'model new' or 'model restore'
itasca.command("python-reset-state false")

def run_dat_file(file_path):
    try:
        # Convert path to proper format
        full_path = os.path.abspath(file_path)
        full_path = full_path.replace("\\", "/")
        
        # Execute the .dat file
        itasca.command(f'program call "{full_path}"')
        print(f"Successfully executed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error executing {file_path}: {str(e)}")
        return False

def compute_dimensions():
    wp_right = wall.find('boxWallRight2')
    wp_left = wall.find('boxWallLeft4')
    wp_up = wall.find('boxWallTop3')
    wp_down = wall.find('boxWallBottom1')
    wlx = wp_right.pos_x() - wp_left.pos_x()
    wly = wp_up.pos_y() - wp_down.pos_y()
    return wlx, wly

def set_balls_group_in_area(x_min, x_max, y_min, y_max, group_name, slot_name=None):
    # Remove existing group if it exists
    # if slot_name:
    #     command_str = f'ball group "{group_name}" slot "{slot_name}" remove'
    # else:
    #     command_str = f'ball group "{group_name}" remove'
    # itasca.command(command_str)
    
    # Get ball positions as numpy array
    positions = ballarray.pos()
    
    # Create boolean mask for balls in the area
    in_area = (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) & \
              (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max)
    
    # Set group membership using the mask
    if slot_name:
        ballarray.set_group(in_area, group_name, slot_name)
    else:
        ballarray.set_group(in_area, group_name)

def get_balls_max_pos(dim):
    '''
    Get the maximum position of balls in the given dimension

    Args:
        dim (int): Dimension index (0 for x, 1 for y)
    
    Returns:
        float: Maximum position of balls in the given dimension
    '''
    positions = ballarray.pos()
    return np.max(positions[:, dim])

def get_balls_object_in_area(group_name, y_min, y_max):
    ball_objects = []
    balls = ball.list()
    
    for ball_obj in balls:
        pos = ball_obj.pos()
        if ball_obj.in_group(group_name, 'section') and (y_min <= pos[1] <= y_max):
            ball_objects.append(ball_obj)
    
    return ball_objects

def get_avg_ball_y_disp(ball_objects):
    y_disps = [ball_obj.disp_y() for ball_obj in ball_objects]
    return np.mean(y_disps)

def get_balls_y_displacement_matrix(window_size, model_width, model_height, overlap=0.5):
    """
    Create a heatmap of y-displacements using a sliding window.
    
    Args:
        window_size (float): Size of the square sliding window
        model_width (float): Total width of the model
        model_height (float): Total height of the model
        overlap (float): Overlap ratio between windows (0 to 1), default 0.5
    
    Returns:
        tuple: (displacement_matrix, x_centers, y_centers)
    """
    # Calculate step size based on overlap
    step_size = window_size * (1 - overlap)
    
    # Calculate number of windows in each direction
    n_windows_x = int(np.ceil((model_width - window_size) / step_size)) + 1
    n_windows_y = int(np.ceil((model_height - window_size) / step_size)) + 1
    
    # Initialize matrices for results
    displacement_matrix = np.zeros((n_windows_y, n_windows_x))
    x_centers = np.zeros(n_windows_x)
    y_centers = np.zeros(n_windows_y)
    
    # Get wall positions for reference
    wp_left = wall.find('boxWallLeft4').pos_x()
    wp_bottom = wall.find('boxWallBottom1').pos_y()
    
    # Slide window across model
    # Pre-fetch all balls and their positions once
    all_balls = list(ball.list())  # Convert iterator to list once
    all_positions = np.empty((len(all_balls), 2))  # Pre-allocate arrays
    all_disps = np.empty(len(all_balls))
    
    # Fill arrays in single pass through balls
    for i, ball_obj in enumerate(all_balls):
        all_positions[i] = ball_obj.pos()
        all_disps[i] = ball_obj.disp_y()
    
    for i in range(n_windows_y):
        y_min = wp_bottom + i * step_size
        y_max = y_min + window_size
        y_centers[i] = y_min + window_size/2
        
        for j in range(n_windows_x):
            x_min = wp_left + j * step_size
            x_max = x_min + window_size
            x_centers[j] = x_min + window_size/2
            
            # Use numpy boolean indexing to find balls in window
            in_window = ((all_positions[:, 0] >= x_min) & 
                        (all_positions[:, 0] <= x_max) &
                        (all_positions[:, 1] >= y_min) & 
                        (all_positions[:, 1] <= y_max))
            
            # Calculate average displacement if balls exist in window
            if np.any(in_window):
                displacement_matrix[i, j] = np.mean(all_disps[in_window])
            else:
                displacement_matrix[i, j] = np.nan
    
    return displacement_matrix, x_centers, y_centers

# Example usage and plotting:
def plot_y_displacement_heatmap(window_size, model_width, model_height, name, save_path=".", overlap=0.5):
    """
    Create and plot the displacement heatmap.
    
    Args:
        window_size (float): Size of the square sliding window
        overlap (float): Overlap ratio between windows (0 to 1)
    """
    
    # Create heatmap data
    disp_matrix, x_centers, y_centers = get_balls_y_displacement_matrix(
        window_size, model_width, model_height, overlap
    )
    
    # Create heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        disp_matrix,
        extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
        origin='lower',
        aspect='equal',
        cmap='coolwarm_r'
    )
    
    plt.colorbar(label='Y Displacement')
    plt.clim(-6,0)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Y Displacement Heatmap: {name}')
    plt.savefig(f'{save_path}/displacement_heatmap_{name}.png')
    plt.close()

def fenceng(sec_interval, layer_array, subsurface_level=5):
        
    # Get starting x,y position
    wlx, wly = compute_dimensions()
    ypos0 = wall.find('boxWallBottom1').pos_y()
    xpos0 = wall.find('boxWallLeft4').pos_x()
    
    # Calculate number of sections based on wall width and interval
    assert wlx > sec_interval, f"Section interval must be less than wall width, Current sec_interval: {sec_interval}, wall width: {wlx}"
    sec_num = int(wlx // sec_interval)

    height_array = [0]
    height_array.extend(layer_array)
    assert height_array[-1] < wly - subsurface_level, f"Height array must be less than wall height, Current height array: {height_array}, wall height: {wly}"
    height_array.append(wly - subsurface_level)
    height_array.append(wly)

    # Create layer groups
    for i in range(1, len(height_array)):
        ypos_up = ypos0 + height_array[i]
        ypos_down = ypos0 if i == 1 else ypos0 + height_array[i-1]
        
        # Assign balls to layer groups
        set_balls_group_in_area(
            x_min=wall.find('boxWallLeft4').pos_x(),
            x_max=wall.find('boxWallRight2').pos_x(),
            y_min=ypos_down,
            y_max=ypos_up,
            group_name=str(i) if i != len(height_array)-1 else 'subsurface',
            slot_name='layer'
        )
    
    # Create section groups
    for j in range(1, sec_num + 1):
        xpos_right = xpos0 + j * sec_interval
        xpos_left = xpos0 + (j-1) * sec_interval
        
        # Assign balls to section groups
        set_balls_group_in_area(
            x_min=xpos_left,
            x_max=xpos_right,
            y_min=wall.find('boxWallBottom1').pos_y(),
            y_max=wall.find('boxWallTop3').pos_y(),
            group_name=str(j),
            slot_name='section'
        )
    
    # Handle remaining section if needed
    if xpos_right < (wlx / 2 - itasca.fish.get('rdmax')):
        xpos_right = xpos0 + wlx
        xpos_left = xpos0 + sec_num * sec_interval
        
        # Assign balls to extra section group
        set_balls_group_in_area(
            x_min=xpos_left,
            x_max=xpos_right,
            y_min=wall.find('boxWallBottom1').pos_y(),
            y_max=wall.find('boxWallTop3').pos_y(),
            group_name=str(sec_num + 1),
            slot_name='section'
        )
        
        return sec_num + 1
    
    return sec_num

if __name__ == "__main__":
    # yuya
    run_dat_file("yuya-new.dat")

    # fenceng
    itasca.command("model restore 'yuya'")
    wlx, wly = compute_dimensions()
    sec_num = fenceng(
        sec_interval=sec_interval, 
        layer_array=layer_array,
        subsurface_level=subsurface_level
    )
    itasca.command("model save 'fenceng'")

    # pingheng
    itasca.command("model restore 'fenceng'")

    # send contact properties to FISH
    itasca.fish.set('fric', fric)
    itasca.fish.set('rfric', rfric)
    itasca.fish.set('dpnr', dpnr)
    itasca.fish.set('dpsr', dpsr)
    itasca.fish.set('F0', F0)

    run_dat_file("pingheng-linear.dat")

    # kaiwa
    # Restore the model
    itasca.command("model restore 'pingheng'")
    
    # Reset ball attributes
    itasca.command("ball attribute velocity 0 spin 0 displacement 0")

    # get a dict ball objects of each section near the ball at the top of the model
    top_ball_pos = get_balls_max_pos(1)
    
    ball_objects_dict = {}
    for i in range(1, sec_num + 1):
        ball_objects_dict[str(i)] = get_balls_object_in_area(str(i), top_ball_pos-1, top_ball_pos)

    # delete empty list in ball_objects_dict
    ball_objects_dict = {k: v for k, v in ball_objects_dict.items() if v}
    
    sec_num = len(ball_objects_dict)

    y_disps_list = {}
    
    # Loop through sections
    for i in range(opencut_sec, sec_num-2):
        section_name = str(i)
        name = 'result' + str(i)
        
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
            itasca.command(f"model save '{name}'")
            itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-5")
        elif i < sec_num-3:
            itasca.command(f"model save '{name}'")
            itasca.command(f"model solve cycle {step_interval} or ratio-average 1e-3")
        else:
            itasca.command(f"model save '{name}'")
            itasca.command(f"model solve ratio-average 1e-3")

        # get avg y disp of each section and plot the y disp vs section number
        y_disps = [get_avg_ball_y_disp(ball_objects_dict[str(i)]) for i in range(1, sec_num + 1)]
        y_disps_list[i] = y_disps
        plt.plot(range(1, sec_num + 1), y_disps, label=f'{name}')
        plt.xlabel('Section Number')
        plt.ylabel('Average Y Displacement')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plot_y_displacement_heatmap(window_size=1.0, model_width=wlx, model_height=wly, name=section_name, save_path=image_save_path)

    plt.savefig(f'{image_save_path}/{name}.png')

    # save y_disps_list to csv
    with open(f'{name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row with step numbers
        writer.writerow(['Section'] + list(np.fromiter(y_disps_list.keys(), dtype=float)*sec_interval))
        # Write data rows
        for section in range(1, sec_num + 1):
            row = [section] + [y_disps_list[step][section-1] for step in y_disps_list]
            writer.writerow(row)
    
    # Save final model
    itasca.command("model save 'final'")
