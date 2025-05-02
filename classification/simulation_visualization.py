# simulation_visualization.py
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def simulate_trajectory(file_path, title='Nose Trajectory Simulation', color = 'b'):
    """
    Simulates the trajectory of a point (e.g., nose) in 3D space based on data from a JSON file.

    Args:
        file_path (str): Path to the JSON file with coordinates.
    """
    # Load data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract nose coordinates from each frame
    nose_coords = [frame['landmarks']['nose'] for frame in data['data'] if 'nose' in frame['landmarks']]
    
    if not nose_coords:
        print(f"Error: 'nose' data is missing in {file_path}")
        return
    
    # Split coordinates into X, Y, Z lists
    x = [coord[0] for coord in nose_coords]
    y = [coord[1] for coord in nose_coords]
    z = [coord[2] for coord in nose_coords]
    
    # Create a figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Axis labels with physical meaning
    ax.set_xlabel('X (horizontal, m)')
    ax.set_ylabel('Y (depth, m)')
    ax.set_zlabel('Z (height, m)')
    
    # Graph title
    ax.set_title(title)
    
    # Initialize trajectory line and current position point
    line, = ax.plot([], [], [], color=color, label='Trajectory')
    point, = ax.plot([], [], [], 'o', color=color, markersize=8, label='Current Position')
    
    # Add legend
    ax.legend()
    
    # Set axis limits based on data
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    
    # Initialization function for animation
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    # Animation update function
    def animate(i):
        # Update trajectory line up to the current frame
        line.set_data(x[:i+1], y[:i+1])
        line.set_3d_properties(z[:i+1])
        # Update the current position point
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        return line, point
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(x), init_func=init, blit=False, interval=100)
    
    # Set initial view angle for better 3D perception
    ax.view_init(elev=20, azim=30)
    
    # Enable grid for spatial orientation
    ax.grid(True)
    
    # Display the graph
    plt.show()