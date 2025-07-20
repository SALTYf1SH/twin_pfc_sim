import itasca
from itasca import ball, wall, ballarray
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.ndimage import convolve

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
    
def delete_balls_outside_area(x_min, x_max, y_min, y_max):
    for ball_obj in ball.list():
        pos = ball_obj.pos()
        if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
            ball_obj.delete()

def compute_dimensions():
    wp_right = wall.find('boxWallRight2')
    wp_left = wall.find('boxWallLeft4')
    wp_up = wall.find('boxWallTop3')
    wp_down = wall.find('boxWallBottom1')
    wlx = wp_right.pos_x() - wp_left.pos_x()
    wly = wp_up.pos_y() - wp_down.pos_y()
    return wlx, wly

def set_balls_group_in_area(x_min, x_max, y_min, y_max, group_name, slot_name=None):
    # Check if ballarray methods are available
    try:
        # Try using ballarray methods
        positions = ballarray.pos()
        in_area = (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) & \
                  (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max)
        
        if slot_name:
            ballarray.set_group(in_area, group_name, slot_name)
        else:
            ballarray.set_group(in_area, group_name)
            
    except (AttributeError, TypeError):
        # Fallback to using ball class methods
        for ball_obj in ball.list():
            pos = ball_obj.pos()
            if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                if slot_name:
                    ball_obj.set_group(group_name, slot_name)
                else:
                    ball_obj.set_group(group_name)

def get_balls_max_pos(dim):
    '''
    Get the maximum position of balls in the given dimension

    Args:
        dim (int): Dimension index (0 for x, 1 for y)
    
    Returns:
        float: Maximum position of balls in the given dimension
    '''
    try:
        # Try using ballarray method
        positions = ballarray.pos()
        return np.max(positions[:, dim])
    except (AttributeError, TypeError):
        # Fallback to using ball class methods
        positions = [ball_obj.pos()[dim] for ball_obj in ball.list()]
        return max(positions)

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

def interpolate_nan_values(matrix, method='cubic'):
    """
    Interpolate NaN values in a matrix using the average of neighboring cells.
    """
    nan_mask = np.isnan(matrix)
    if not np.any(nan_mask):  # If no NaN values, return original matrix
        return matrix
    
    # Define a 3x3 kernel with all ones
    kernel = np.ones((3, 3), dtype=int)
    
    # Step 2: Convolve the mask to count NaNs in the neighborhood
    # Use 'constant' mode with cval=0 to treat out-of-bound as non-NaN
    nan_count = convolve(nan_mask.astype(int), kernel, mode='constant', cval=0)
    
    # Internal cells where all 9 cells (center + 8 neighbors) are NaN
    internal_condition = (nan_mask) & (nan_count == 9)
    
    # Step 3: Handle border cells
    # Define the shape
    rows, cols = matrix.shape
    
    # Initialize border condition mask
    border_condition = np.zeros_like(nan_mask)
    
    # Check top and bottom borders
    for i in [0, rows-1]:
        for j in range(cols):
            if nan_mask[i, j]:
                # Extract the neighborhood, handling boundaries
                neighbors = nan_mask[max(i-1, 0):min(i+2, rows), max(j-1, 0):min(j+2, cols)]
                if np.all(neighbors):
                    # Check from current position to the edge are NaN
                    if (i == 0 and np.all(nan_mask[0:i+1, j])) or (i == rows-1 and np.all(nan_mask[i:rows, j])):
                        border_condition[i, j] = True
    
    # Check left and right borders (excluding corners already checked)
    for j in [0, cols-1]:
        for i in range(1, rows-1):
            if nan_mask[i, j]:
                neighbors = nan_mask[max(i-1, 0):min(i+2, rows), max(j-1, 0):min(j+2, cols)]
                if np.all(neighbors):
                    if (j == 0 and np.all(nan_mask[i, 0:j+1])) or (j == cols-1 and np.all(nan_mask[i, j:cols])):
                        border_condition[i, j] = True
    
    # Combine internal and border conditions
    final_mask = internal_condition | border_condition

    # Create coordinates for valid (non-nan) points
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    x_valid = x[~nan_mask]
    y_valid = y[~nan_mask]
    points = np.column_stack((x_valid, y_valid))
    
    # Get valid values
    values = matrix[~nan_mask]
    
    # Create coordinates for all points
    xi, yi = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    
    # Interpolate
    matrix = griddata(points, values, (xi, yi), method=method)

    # Restore nan values that meet the conditions
    matrix[final_mask] = np.nan

    return matrix

def get_balls_y_displacement_matrix(window_size, model_width, model_height, interpolate, overlap=0.5):
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
    if interpolate:
        # Interpolate nan values according to the average value of 8 neighboring cells
        displacement_matrix = interpolate_nan_values(displacement_matrix, method=interpolate)
    
    return displacement_matrix, x_centers, y_centers

# Example usage and plotting:
def plot_y_displacement_heatmap(window_size, model_width, model_height, name, interpolate='nearest', resu_path=".", overlap=0.5):
    """
    Create and plot the displacement heatmap.
    
    Args:
        window_size (float): Size of the square sliding window
        overlap (float): Overlap ratio between windows (0 to 1)
    """
    
    # Create heatmap data
    disp_matrix, x_centers, y_centers = get_balls_y_displacement_matrix(
        window_size, model_width, model_height, interpolate, overlap
    )
    
    # Create heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        disp_matrix,
        extent=[0, x_centers[-1]-x_centers[0], 0, y_centers[-1]-y_centers[0]],
        origin='lower',
        aspect='equal',
        cmap='coolwarm_r'
    )
    
    im_ratio = disp_matrix.shape[0]/disp_matrix.shape[1]
    plt.colorbar(label='Y Displacement', fraction=0.046*im_ratio, pad=0.02)
    plt.clim(-6,0)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Y Displacement Heatmap: {name}')
    plt.savefig(os.path.join(resu_path, 'img', f'displacement_heatmap_{name}.png'), dpi=400, bbox_inches='tight')
    plt.close()

    # Also save the displacement matrix to a npz file
    # Save under the "mat" folder under the result path
    # Include the excavation position in the file name
    np.savez(os.path.join(resu_path, 'mat', f'y_displacement_matrix_{name}.npz'), disp_matrix=disp_matrix, x_centers=x_centers, y_centers=y_centers)


def fenceng(layer_array):
    """
    Create layer and section groups for fenceng model.

    Args:
        sec_interval (float): Section interval
        layer_array (list): Layer array
        first_section_length (float): Length of the first section
        subsurface_level (float): Subsurface level

    Returns:
        int: Number of sections
    """
        
    # Get starting x,y position
    wlx, wly = compute_dimensions()
    ypos0 = wall.find('boxWallBottom1').pos_y()
    xpos0 = wall.find('boxWallLeft4').pos_x()
    
    # Calculate number of sections based on wall width and interval

    height_array = [0]
    height_array.extend(layer_array)


    # Create layer groups
    for i in range(1, len(height_array)):
        ypos_up = ypos0 + height_array[i]
        ypos_down = ypos0 if i == 1 else ypos0 + height_array[i-1]

        print(f"Layer {i}: ypos_down: {ypos_down}, ypos_up: {ypos_up}")
        
        # Assign balls to layer groups
        set_balls_group_in_area(
            x_min=-1000, 
            x_max=1000,
            y_min=ypos_down,
            y_max=ypos_up,
            group_name=str(i),
            slot_name='layer'
        )
    sec_num = len(height_array)
    return sec_num
    
    

# Call the function
if __name__ == "__main__":
    itasca.command("model restore 'yuya'")
    # Print all methods of ballarray
    wlx, wly = compute_dimensions()
    fenceng(3, [5,], 5, 5)
    plot_y_displacement_heatmap(3, wlx, wly, 'fenceng', overlap=0.5)