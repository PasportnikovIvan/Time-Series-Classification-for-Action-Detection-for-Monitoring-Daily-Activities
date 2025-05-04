#classification/simulation_visualization.py

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mediapipe as mp

def simulate_trajectory(file_path, title='Nose Trajectory Simulation', color = 'b'):
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
    frames = [frame['landmarks'] for frame in data['data'] if 'landmarks' in frame]
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
    
    # Extract coordinates for all landmarks across frames
    coords = {name: [[frame[name][j] for frame in frames] for j in range(3)] for name in landmark_names}

    # Define connections between landmarks (from MediaPipe Pose)
    connections = mp.solutions.pose.POSE_CONNECTIONS

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
    all_x = [x for name in coords for x in coords[name][0]]
    all_y = [y for name in coords for y in coords[name][1]]
    all_z = [z for name in coords for z in coords[name][2]]
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # Initialize lines for connections and points for landmarks
    lines = [ax.plot([], [], [], color=color, alpha=0.5)[0] for _ in connections]
    points = ax.plot([], [], [], 'o', color=color, markersize=4, label='Landmarks')[0]

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
            return lines + [points]
    
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

        return lines + [points]
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), init_func=init, blit=False, interval=100)
    
    # Set initial view angle for better 3D perception
    ax.view_init(elev=20, azim=30)
    
    # Display the graph
    plt.show()