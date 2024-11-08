from itasca import ball, wall, ballarray

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

def fenceng(height_array, sec_interval):
        
    # Get starting x,y position
    wlx, wly = compute_dimensions()
    ypos0 = wall.find('boxWallBottom1').pos_y()
    xpos0 = wall.find('boxWallLeft4').pos_x()
    
    # Calculate number of sections based on wall width and interval
    sec_num = int(wlx // sec_interval)

    assert sec_num > 0, "Number of sections must be greater than 0"

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
            group_name=str(i),
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
    if xpos_right < (wlx - itasca.fish.get('rdmax')):
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

# Call the function
if __name__ == "__main__":
    itasca.command("model restore 'yuya'")
    _, wly = compute_dimensions()  # Get wall dimensions
    fenceng(height_array=[0, 5, wly-5, wly], sec_interval=3)
