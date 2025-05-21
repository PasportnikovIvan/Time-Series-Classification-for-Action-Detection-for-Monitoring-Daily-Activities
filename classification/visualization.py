#classification/vizualization.py

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mediapipe as mp
import numpy as np
import os

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

def simulate_full_body_trajectory(file_path, title='Nose Trajectory Simulation', color = 'b'):
    """
    Simulates the trajectory of the full body skeleton in 3D space based on data from a JSON file.
    Args:
        file_path (str): Path to the JSON file with coordinates.
        title (str): Title for the simulation plot.
        color (str): Color of the skeleton lines and points (default is 'b' for blue).
    """
    # Load data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract all landmarks from each frame
    # Extract frames, timestamps, object coords, and audio
    frames = [frame['landmarks'] for frame in data['data'] if 'landmarks' in frame]
    timestamps = [frame['timestamp'] for frame in data['data']]
    obj_list = [frame.get('obj_coords') for frame in data['data']]
    sound_amp = [frame.get('sound_amp', 0.0) for frame in data['data']]
    if not frames or not all(all(key in frame for key in ['nose']) for frame in frames):
        print(f"Error: Incomplete landmark data in {file_path}")
        return

    # Get landmark names from MediaPipe Pose
    landmark_names = [
        "nose",
        "left eye (inner)",
        "left eye",
        "left eye (outer)",
        "right eye (inner)",
        "right eye",
        "right eye (outer)",
        "left ear",
        "right ear",
        "mouth (left)",
        "mouth (right)",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left pinky",
        "right pinky",
        "left index",
        "right index",
        "left thumb",
        "right thumb",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
        "left heel",
        "right heel",
        "left foot index",
        "right foot index"
    ]
    
    # Define connections between landmarks (from MediaPipe Pose)
    connections = mp.solutions.pose.POSE_CONNECTIONS

    # Extract coordinates for all landmarks across frames
    coords = {name: [[frame[name][j] for frame in frames] for j in range(3)] for name in landmark_names}

    # Prepare object trajectory lists
    obj_x, obj_y, obj_z = [], [], []
    has_object = any(o is not None for o in obj_list)
    for o in obj_list:
        if o is None:
            obj_x.append(np.nan)
            obj_y.append(np.nan)
            obj_z.append(np.nan)
        else:
            ox, oy, oz = o
            obj_x.append(ox)
            obj_y.append(oy)
            obj_z.append(oz)

    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Axis labels
    ax.set_xlabel('X (horizontal, m)')
    ax.set_ylabel('Y (depth, m)')
    ax.set_zlabel('Z (height, m)')
    # Graph title
    ax.set_title(title)

    # Set axis limits based on data
    all_x = [x for name in coords for x in coords[name][0]] + obj_x
    all_y = [y for name in coords for y in coords[name][1]] + obj_y
    all_z = [z for name in coords for z in coords[name][2]] + obj_z
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # Initialize lines for connections and points for landmarks
    lines = [ax.plot([], [], [], color=color, alpha=0.5)[0] for _ in connections]
    points = ax.plot([], [], [], 'o', color=color, markersize=4, label='Landmarks')[0]
    if has_object:
        obj_point = ax.plot([], [], [], 'X', color='k', markersize=8, label='Object')[0]

    # Add legend
    ax.legend()
    ax.grid(True)
    
    # Initialization function for animation
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        points.set_data([], [])
        points.set_3d_properties([])
        if has_object:
            obj_point.set_data([], [])
            obj_point.set_3d_properties([])
        return lines + [points] + ([obj_point] if has_object else [])
    
    # Animation update function
    def animate(i):
        # Update points (all landmarks)
        x_points = [coords[name][0][i] for name in landmark_names]
        y_points = [coords[name][1][i] for name in landmark_names]
        z_points = [coords[name][2][i] for name in landmark_names]
        points.set_data(x_points, y_points)
        points.set_3d_properties(z_points)

        # Update lines (connections)
        for idx, (start_idx, end_idx) in enumerate(connections):
            start_name = landmark_names[start_idx]
            end_name = landmark_names[end_idx]
            x_line = [coords[start_name][0][i], coords[end_name][0][i]]
            y_line = [coords[start_name][1][i], coords[end_name][1][i]]
            z_line = [coords[start_name][2][i], coords[end_name][2][i]]
            lines[idx].set_data(x_line, y_line)
            lines[idx].set_3d_properties(z_line)

        # Update object marker if present
        if has_object:
            obj_point.set_data([obj_x[i]], [obj_y[i]])
            obj_point.set_3d_properties([obj_z[i]])

        return lines + [points] + ([obj_point] if has_object else [])
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), init_func=init, blit=False, interval=100)
    
    # Set initial view angle for better 3D perception
    ax.view_init(elev=20, azim=30)
    
    # Display the graph
    plt.show()

    # --- Audio amplitude plot ---
    plt.figure(figsize=(8, 3))
    plt.plot(timestamps, sound_amp, color='tab:orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Audio amplitude')
    plt.title(f'{title} — Sound Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_nose_velocity(file_paths, action):
    """
    Visualizes the velocity of the nose for multiple sessions of an action.
    Args:
        file_paths (list): List of paths to JSON files for the action.
        action (str): Name of the action (e.g., 'falling', 'lying').
    """
    plt.figure(figsize=(12, 8))

    # Define color map for different sessions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        timestamps = [frame['timestamp'] for frame in data['data'] if 'nose' in frame['landmarks']]
        nose_coords = [frame['landmarks']['nose'] for frame in data['data'] if 'nose' in frame['landmarks']]
        
        if not timestamps or len(timestamps) < 2:
            print(f"Warning: Insufficient data in {file_path} for velocity calculation")
            continue
        
        dt = np.diff(timestamps)
        dx = np.diff([coord[0] for coord in nose_coords])
        dy = np.diff([coord[1] for coord in nose_coords])
        dz = np.diff([coord[2] for coord in nose_coords])
        
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt
        
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)

        # Get session number from filename (assuming format: action_XX_...)
        session_num = os.path.basename(file_path).split('_')[1]
        
        plt.plot(timestamps[1:], velocity, color=colors[idx % len(colors)], alpha=0.8, label=f"{action} session {session_num}" if action not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.title(f'Nose Velocity for {action.capitalize()}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()