#classification/vizualization.py

import json
import matplotlib.pyplot as plt

def plot_nose_trajectory(file_path, title, color='b'):
    """
    Visualizes the trajectory of the nose in 3D space from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing nose coordinates.
        title (str): Header for the plot.
        color (str): Color of the trajectory line (default is 'b' for blue).
    """
    # Reading the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extracting nose coordinates
    nose_coords = [frame['landmarks']['nose'] for frame in data['data']]
    if not nose_coords:
        print(f"Error: No 'nose' data found in {file_path}")
        return
    
    x = [coord[0] for coord in nose_coords]
    y = [coord[1] for coord in nose_coords]
    z = [coord[2] for coord in nose_coords]
    
    # Extracting start, mid, and end points
    start_point = (x[0], y[0], z[0])
    mid_point = (x[len(x)//2], y[len(y)//2], z[len(z)//2])
    end_point = (x[-1], y[-1], z[-1])
    
    # Building the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the trajectory
    ax.plot(x, y, z, color=color, label='Trajectory')
    
    # Plotting the start, mid, and end points
    ax.scatter(*start_point, color='green', s=100, label='Start')
    ax.scatter(*mid_point, color='blue', s=100, label='Mid')
    ax.scatter(*end_point, color='red', s=100, label='End')
    
    # Setting labels for axes
    ax.set_xlabel('X (horizontal, м)')
    ax.set_ylabel('Y (depth, м)')
    ax.set_zlabel('Z (vertical, м)')
    
    # Adding title and legend
    ax.set_title(title)
    ax.legend()
    
    # Setting the view angle
    ax.view_init(elev=20, azim=30)
    
    # Gridding the plot
    ax.grid(True)
    
    plt.show()