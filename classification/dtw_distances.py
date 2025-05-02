# classification/vizualization.py
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from vizualization import plot_nose_trajectory
import os

def plot_nose_velocity(file_paths, action, color='b'):
    """
    Visualizes the velocity of the nose for multiple sessions of an action.

    Args:
        file_paths (list): List of paths to JSON files for the action.
        action (str): Name of the action (e.g., 'falling', 'lying').
        color (str): Color for plotting the velocity curves (default is blue).
    """
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
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
        
        plt.plot(timestamps[1:], velocity, color=color, alpha=0.6, label=f"{action} session" if action not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Nose Velocity for {action}')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_dtw_distances(reference_file, other_files):
    """
    Computes DTW distances between a reference trajectory and all other trajectories.

    Args:
        reference_file (str): Path to the reference JSON file.
        other_files (list): List of paths to other JSON files to compare with.

    Returns:
        list: List of tuples (file_path, dtw_distance, action_name) sorted by distance.
    """
    # Load reference trajectory
    with open(reference_file, 'r') as f:
        ref_data = json.load(f)
    ref_coords = [frame['landmarks']['nose'] for frame in ref_data['data'] if 'nose' in frame['landmarks']]
    
    if not ref_coords:
        print(f"Error: No 'nose' data in reference file {reference_file}")
        return []
    
    distances = []
    for file_path in other_files:
        # Load comparison trajectory
        with open(file_path, 'r') as f:
            data = json.load(f)
        coords = [frame['landmarks']['nose'] for frame in data['data'] if 'nose' in frame['landmarks']]
        
        if not coords:
            print(f"Warning: No 'nose' data in {file_path}")
            continue
        
        # Compute DTW distance
        distance, _ = fastdtw(ref_coords, coords, dist=euclidean)
        
        # Extract action name from file path (e.g., 'falling' from 'dataset/globalLandmarks/falling/...')
        action_name = file_path.split(os.sep)[2]  # Assuming path like 'dataset/globalLandmarks/action/...'
        distances.append((file_path, distance, action_name))
    
    # Sort by DTW distance
    distances.sort(key=lambda x: x[1])
    return distances