#classification/vizualization.py

import json
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import pdb

def extract_nose_trajectory(file_path):
    """
    Extracts the nose trajectory from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: List of nose coordinates.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    nose_trajectory = []
    for frame in data['data']:
        if 'nose' in frame['landmarks']:
            nose_trajectory.append(frame['landmarks']['nose'])
    return nose_trajectory

def extract_all_landmarks(file_path):
    """
    Extracts all landmarks from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: List of all landmarks for each frame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    landmarks = []
    for frame in data['data']:
        if 'landmarks' in frame:
            frame_landmarks = []
            # Sort landmarks by key to ensure consistent order
            for key in sorted(frame['landmarks'].keys()):
                frame_landmarks.extend(frame['landmarks'][key])  # Add all coordinates
            landmarks.append(frame_landmarks)
    return landmarks

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
    ref_coords = extract_all_landmarks(reference_file)
    if not ref_coords:
        print(f"Error: No landmark data in reference file {reference_file}")
        return []
    
    distances = []
    for file_path in other_files:
        # Load comparison trajectory
        coords = extract_all_landmarks(file_path)
        if not coords:
            print(f"Warning: No landmark data in {file_path}")
            continue
        
        # Compute DTW distance
        distance, _ = fastdtw(ref_coords, coords, dist=euclidean)
        
        # Extract action name from file path (e.g., 'falling' from 'dataset/globalLandmarks/falling/...')
        action_name = file_path.split(os.sep)[1]  # Assuming path like 'dataset/globalLandmarks/action/...'
        distances.append((file_path, distance, action_name))
    
    # Sort by DTW distance
    distances.sort(key=lambda x: x[1])
    return distances

def compute_dtw_distance_matrix(file_list, use_all_landmarks=True):
    """
    Calculates the DTW distance matrix for all pairs of files in the list.

    Args:
        file_list (list): List of paths to JSON files.
        use_all_landmarks (bool): Use all landmarks or only nose trajectory or only nose trajectory.

    Returns:
        np.ndarray: DTW distance matrix.
        list: List of file paths.
    """
    n = len(file_list)
    distance_matrix = np.zeros((n, n))
    
    # Traectories for DTW
    trajectories = []
    for file_path in file_list:
        if use_all_landmarks:
            coords = extract_all_landmarks(file_path)
        else:
            coords = extract_nose_trajectory(file_path)
        if not coords:
            print(f"Пропуск {file_path}: нет валидных данных")
            continue
        trajectories.append(coords)
    
    # Calculate DTW distances
    for i, j in combinations(range(n), 2):
        distance, _ = fastdtw(trajectories[i], trajectories[j], dist=euclidean)
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # Simmetric property of DTW
    
    return distance_matrix, file_list

def plot_distance_matrix(distance_matrix, file_list, cmap='viridis', title="DTW Distance Matrix"):
    """
    Plots the DTW distance matrix as a heatmap.
    Args:
        distance_matrix (np.ndarray): DTW distance matrix.
        file_list (list): List of file paths.
        title (str): Title for the plot.
    """
    labels = [os.path.basename(f).split('_')[0] + '_' + os.path.basename(f).split('_')[1] for f in file_list]
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, xticklabels=labels, yticklabels=labels, cmap=cmap, annot=False)
    plt.title(title)
    plt.savefig('dtw_distance_matrix.png')  # Save the figure
    plt.show()
    # plt.close()

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
        test_coords = extract_all_landmarks(test_path)
        if not test_coords:
            print(f"Warning: No landmark data in test file {test_path}")
            predictions.append(None)
            continue

        distances = []    
        for train_path, train_action in train_files:
            train_coords = extract_all_landmarks(train_path)
            if not train_coords:
                print(f"Warning: No landmark data in train file {train_path}")
                continue
                
            distance, _ = fastdtw(test_coords, train_coords, dist=euclidean)
            distances.append((distance, train_action))
        
        if distances:
            closest_action = min(distances, key=lambda x: x[0])[1]  # Get action with smallest distance
            print(f"Predicted action for {test_files[0]}: {closest_action}")
            predictions.append(closest_action)
        else:
            predictions.append(None)
    
    return predictions, true_labels