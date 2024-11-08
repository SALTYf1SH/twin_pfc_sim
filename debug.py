# Import the itasca module
import itasca
from itasca import ballarray
import numpy as np
import os

# Prevent Python state from resetting when issuing 'model new' or 'model restore'
itasca.command("python-reset-state false")

# Define the model setup within a multi-line string
model_setup = """
model new
model domain extent -1 1 -1 1
model cmat default model linear property kn 1e4 dp_nratio 0.2
ball generate cubic box -0.9 0.9 rad 0.05
ball attr dens 2600
"""

# Execute the model setup commands
itasca.command(model_setup)

# Method 1: Range-based search
def set_balls_group_in_area(x_min, x_max, y_min, y_max, group_name, slot_name=None):
    # Remove existing group if it exists
    if slot_name:
        command_str = f'ball group "{group_name}" slot "{slot_name}" remove'
    else:
        command_str = f'ball group "{group_name}" remove'
    itasca.command(command_str)
    
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

# Method 2: Box-based search
def get_balls_in_box(center_x, center_y, width, height):
    x_min = center_x - width/2
    x_max = center_x + width/2
    y_min = center_y - height/2
    y_max = center_y + height/2
    
    ball_ids = []
    balls = itasca.ball.list()
    
    for ball in balls:
        pos = ball.pos()
        if (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max):  # Fixed comparison
            ball_ids.append(ball.id())
    
    return ball_ids

# Method with radius consideration
def get_balls_in_area_with_radius(x_min, x_max, y_min, y_max):
    ball_ids = []
    balls = itasca.ball.list()
    
    for ball in balls:
        pos = ball.pos()
        rad = ball.radius()
        
        if (x_min - rad <= pos[0] <= x_max + rad) and (y_min - rad <= pos[1] <= y_max + rad):  # Fixed comparison
            ball_ids.append(ball.id())
    
    return ball_ids

# Method with properties
def get_ball_properties_in_area(x_min, x_max, y_min, y_max):
    ball_properties = []
    balls = itasca.ball.list()
    
    for ball in balls:
        pos = ball.pos()
        if (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max):  # Fixed comparison
            properties = {
                'id': ball.id(),
                'position': pos,
                'radius': ball.radius(),
                'velocity': ball.vel()
            }
            ball_properties.append(properties)
    
    return ball_properties

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

# Example usage
# try:
#     # List of .dat files to run
#     dat_files = [
#         "yuya-new.dat",
#         "fenceng-new.dat",
#         "pingheng-new.dat"
#     ]
    
#     # Run each .dat file
#     for dat_file in dat_files:
#         success = run_dat_file(dat_file)
#         if not success:
#             print(f"Failed to execute {dat_file}")
            
# except Exception as e:
#     print(f"An error occurred: {str(e)}")


# Example usage:
try:
    # Get balls in a specific area
    balls_in_area = set_balls_group_in_area(-0.5, 0.5, -0.5, 0.5, 'test', 'test')
    print(f"Balls in area: {balls_in_area}")
    
    # Get balls in a box centered at (0.5, 0.5)
    balls_in_box = get_balls_in_box(0.5, 0.5, 1.0, 1.0)
    print(f"Balls in box: {balls_in_box}")
    
    # Get balls with radius consideration
    balls_with_radius = get_balls_in_area_with_radius(0.0, 1.0, 0.0, 1.0)
    print(f"Balls in area (with radius): {balls_with_radius}")
    
    # Get ball properties
    ball_properties = get_ball_properties_in_area(0.0, 1.0, 0.0, 1.0)
    print(f"Ball properties: {ball_properties}")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
