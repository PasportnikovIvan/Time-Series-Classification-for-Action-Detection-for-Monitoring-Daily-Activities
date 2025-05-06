#classification/vizualization.py

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import pdb

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
        action_name = file_path.split(os.sep)[1]  # Assuming path like 'dataset/globalLandmarks/action/...'
        distances.append((file_path, distance, action_name))
    
    # Sort by DTW distance
    distances.sort(key=lambda x: x[1])
    return distances

def classify_with_dtw(train_files, test_files):
    """
    Classifies test files using DTW by comparing with train files.

    Args:
        train_files (list): List of (file_path, action) for training data.
        test_files (list): List of (file_path, action) for test data.

    Returns:
        list: Predicted actions for test files, list of true actions.
    """
    predictions = []
    if isinstance(test_files, tuple):
        test_files = [test_files]  # Convert single tuple to a list of tuples
    true_labels = [action for _, action in test_files]

    for test_path, _ in test_files:
        with open(test_path, 'r') as f_test:
            test_data = json.load(f_test)
        test_coords = [frame['landmarks']['nose'] for frame in test_data['data'] if 'nose' in frame['landmarks']]
        if not test_coords:
            print(f"Warning: No 'nose' data in test file {test_files[0]}")
            predictions.append(None)
            continue

        distances = []    
        for train_path, train_action in train_files:
            with open(train_path, 'r') as f_train:
                train_data = json.load(f_train)
            train_coords = [frame['landmarks']['nose'] for frame in train_data['data'] if 'nose' in frame['landmarks']]
            if not train_coords:
                print(f"Warning: No 'nose' data in train file {train_path}")
                continue
                
            distance, _ = fastdtw(test_coords, train_coords, dist=euclidean)
            distances.append((distance, train_action))
        
        if distances:
            print(distances)
            closest_action = min(distances, key=lambda x: x[0])[1]  # Get action with smallest distance
            print(f"Predicted action for {test_files[0]}: {closest_action}")
            predictions.append(closest_action)
        else:
            predictions.append(None)
    
    return predictions, true_labels