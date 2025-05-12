#classification/vizualization.py

import os
import json
import numpy as np
from fastdtw import fastdtw
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import euclidean
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
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

def DTW(a, b):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    This function calculates the DTW distance, which measures similarity between two temporal sequences that may vary in speed or length. It uses the FastDTW algorithm with Euclidean distance as the metric.
    Args:
        a (array-like): The first sequence (list, numpy array, etc.).
        b (array-like): The second sequence (list, numpy array, etc.).
    Returns:
        float: The DTW distance between the two input sequences.
    """
    # Convert inputs to numpy arrays if they aren't already
    A = np.array(a)
    B = np.array(b)

    # Handle 3D arrays by flattening the first dimension
    if len(A.shape) == 3:
        A = A.reshape(A.shape[0], -1)
    if len(B.shape) == 3:
        B = B.reshape(B.shape[0], -1)
        
    distance, _ = fastdtw(A, B, dist=euclidean)
    return distance

def custom_dtw_metric(x, y):
    """
    Custom DTW metric for scikit-learn KNeighborsClassifier.
    
    Args:
        x (array): First sequence.
        y (array): Second sequence.
        
    Returns:
        float: DTW distance.
    """
    return DTW(x, y)

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
        distance = DTW(ref_coords, coords)
        
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
            print(f"Pass {file_path}: no landmark data")
            continue
        trajectories.append(coords)
    
    # Calculate DTW distances
    for i, j in combinations(range(n), 2):
        distance = DTW(trajectories[i], trajectories[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance  # Simmetric property of DTW
    
    return distance_matrix, file_list

def plot_distance_matrix(distance_matrix, file_list, save_png=False, cmap='viridis', title="DTW Distance Matrix"):
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
    if save_png:
        plt.savefig('dtw_distance_matrix.png')  # Save the figure
    plt.show()

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
                
            distance = DTW(test_coords, train_coords)
            distances.append((distance, train_action))
        
        if distances:
            closest_action = min(distances, key=lambda x: x[0])[1]  # Get action with smallest distance
            print(f"Predicted action for {os.path.basename(test_path)}: {closest_action}")
            predictions.append(closest_action)
        else:
            predictions.append(None)
    
    return predictions, true_labels

def classify_with_knn_dtw(train_files, test_files, k=3, n_clusters=3):
    """
    Performs kNN classification with DTW and clustering.
    Args:
        train_files (list): List of (file_path, action) for training data.
        test_files (list): List of (file_path, action) for test data.
        k (int): Number of neighbors to consider (default is 3).
    Returns:
        list: Predicted actions, true actions.
    """
    X_train = [extract_all_landmarks(fp) for fp, _ in train_files]
    y_train = [lbl for _, lbl in train_files]
    if isinstance(test_files, tuple):
        test_files = [test_files]
    X_test  = [extract_all_landmarks(fp) for fp, _ in test_files]
    y_test  = [lbl for _, lbl in test_files]
    
    # Computing DTW distance matrix for train data
    print(f"Preparing data for clustering...")
    distance_matrix, file_list = compute_dtw_distance_matrix([fp for fp, _ in train_files], use_all_landmarks=True)
    plot_distance_matrix(distance_matrix, file_list, save_png=True, title="DTW matrix for clustering")

    # Clustering with k-medoids
    print(f"Clustering {len(X_train)} samples into {n_clusters} clusters...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    kmedoids.fit(distance_matrix)
    cluster_labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    for cluster_id in range(n_clusters):
        cluster_samples = [y_train[i] for i in range(len(y_train)) if cluster_labels[i] == cluster_id]
        print(f"Cluster {cluster_id}: {cluster_samples}")

    predictions = []
    for i, test_seq in enumerate(X_test):
        print(f"\nClassifying test sample {i+1}/{len(X_test)}...")
        # Calculating DTW distances to medoids
        distances_to_medoids = []
        for medoid_idx in medoid_indices:
            medoid_seq = X_train[medoid_idx]
            distance = DTW(test_seq, medoid_seq)
            distances_to_medoids.append(distance)
        
        # Getting closest cluster
        closest_cluster = np.argmin(distances_to_medoids)
        
        # Getting cluster indices in closest cluster
        cluster_indices = [i for i, cl in enumerate(cluster_labels) if cl == closest_cluster]
        
        # Calculating DTW distances to all samples in cluster
        distances_to_cluster = []
        for idx in cluster_indices:
            train_seq = X_train[idx]
            distance = DTW(test_seq, train_seq)
            distances_to_cluster.append((distance, y_train[idx]))
        
        # Sorting distances and getting k nearest
        distances_to_cluster.sort(key=lambda x: x[0])
        cluster_labels_set = set(label for _, label in distances_to_cluster)
        print(f"Closest cluster {closest_cluster}: has {len(cluster_indices)} samples, labels: {cluster_labels_set}")
        k_nearest = distances_to_cluster[:k]
        
        # Getting labels k nearest neighbours
        k_labels = [label for _, label in k_nearest]
        print(f"Nearest labels: {k_labels}")
        
        # Using weights
        epsilon = 1e-5  # Avoid zero_division
        weighted_votes = Counter()
        for dist, label in k_nearest:
            weight = 1 / (dist + epsilon)
            weighted_votes[label] += weight

        # Most weight prediction
        prediction = weighted_votes.most_common(1)[0][0]
        print(f"Predicted: {prediction}, Actual: {y_test[i]}")
        predictions.append(prediction)
    
    return predictions, y_test