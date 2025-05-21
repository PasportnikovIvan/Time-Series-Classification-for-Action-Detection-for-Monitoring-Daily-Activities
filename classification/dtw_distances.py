#classification/dtw_distances.py

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

# --- Utility to load and flatten nose trajectory ---
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

# --- Utility to load and flatten all landmarks (motion only) ---
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
        seq = []
        frame_landmarks = frame.get('landmarks', {})
        # Sort landmarks by key to ensure consistent order
        for key in sorted(frame_landmarks.keys()):
            seq.extend(frame_landmarks[key])           # Add all coordinates each lm[key] is [x,y,z]
        landmarks.append(seq)
    return landmarks

# --- Utility to load motion+object+sound features ---
def extract_features(file_path, include_obj=False, include_sound=False):
    """
    Load JSON and return a list of per-frame feature vectors:
     - flattened landmarks
     - optionally appended obj_coords (3 floats)
     - optionally appended sound_amp (1 float)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    sequences = []
    for frame in data['data']:
        seq = []
        # 1) motion
        lm = frame.get('landmarks', {})
        for key in sorted(lm.keys()):
            seq.extend(lm[key])
        # 2) object coords
        if include_obj:
            obj = frame.get('obj_coords')
            if obj is None:
                seq.extend([0.0, 0.0, 0.0])
            else:
                seq.extend(obj)
        # 3) sound amplitude
        if include_sound:
            seq.append(frame.get('sound_amp', 0.0))
        sequences.append(seq)
    return sequences

# --- DTW wrapper ---
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
    # ensure 2D: (frames, features)
    if A.ndim == 1: A = A.reshape(len(A), -1)
    if B.ndim == 1: B = B.reshape(len(B), -1)
        
    distance, _ = fastdtw(A, B, dist=euclidean)
    return distance

# --- Compute DTW distances between a reference trajectory and all other trajectories ---
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
        
        # Extract action name from file path (e.g., 'falling' from 'dataset/processed/falling/...')
        action_name = file_path.split(os.sep)[1]  # Assuming path like 'dataset/processed/action/...'
        distances.append((file_path, distance, action_name))
    
    # Sort by DTW distance
    distances.sort(key=lambda x: x[1])
    return distances

# --- Compute DTW distance matrix ---
def compute_dtw_distance_matrix(file_list=None, sequences=None, use_all_landmarks=True, include_obj=False, include_sound=False):
    """
    Calculates the DTW distance matrix either from a list of file paths
    or from a pre‐extracted list of sequences.
    Args:
        file_list (list[str], optional):
            List of JSON file paths. If provided, `sequences` is ignored.
        sequences (list[list], optional):
            List of per‐frame feature sequences (e.g. X_train). If provided,
            distances are computed on these directly.
        use_all_landmarks (bool):
            If using file_list, whether to load all landmarks (True) or
            only nose trajectory (False).
        include_obj (bool):
            If True, append object coordinates to the feature vector.
        include_sound (bool):
            If True, append sound amplitude to the feature vector.
    Returns:
        distance_matrix (np.ndarray): n×n symmetric DTW distance matrix.
        labels (list): List of file paths (if file_list given) or indices.
    """
    # 1) Determine trajectories and labels
    if sequences is None:
        # load from file_list
        if file_list is None:
            raise ValueError("Either file_list or sequences must be provided")
        trajectories = []
        labels = []
        for path in file_list:
            if use_all_landmarks:
                seq = (extract_features(path, include_obj, include_sound)
                       if (include_obj or include_sound)
                       else extract_all_landmarks(path))
            else:
                seq = extract_nose_trajectory(path)
            if not seq:
                print(f"Warning: no data in {path}, skipping")
                continue
            trajectories.append(seq)
            labels.append(path)
    else:
        # use provided sequences
        trajectories = sequences
        # if file_list provided, use it; else label by index
        if file_list is not None:
            if len(file_list) != len(sequences):
                raise ValueError("file_list and sequences must be same length")
            labels = file_list
        else:
            labels = [f"seq_{i}" for i in range(len(sequences))]

    n = len(trajectories)
    distance_matrix = np.zeros((n, n), dtype=float)
    
    # Calculate DTW distances
    for i, j in combinations(range(n), 2):
        distance = DTW(trajectories[i], trajectories[j])
        distance_matrix[i, j] = distance_matrix[j, i] = distance # Simmetric property of DTW
    
    return distance_matrix, file_list

# --- Plotting DTW distance matrix ---
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

# --- 1-NN classification via DTW ---
def classify_with_dtw(train_files, test_files, include_obj=False, include_sound=False):
    """
    Classifies test files using DTW by comparing with train files.
    Args:
        train_files (list): List of (file_path, action) for training data.
        test_files (list): List of (file_path, action) for test data.
        include_obj: whether to append obj_coords
        include_sound: whether to append sound_amp
    Returns:
        list: Predicted actions for test files, list of true actions.
    """
    predictions, true_labels = [], []

    for test_path, true_label in test_files:
        true_labels.append(true_label)
        
        # load appropriate features
        if include_obj or include_sound:
            test_seq = extract_features(test_path, include_obj, include_sound)
        else:
            test_seq = extract_all_landmarks(test_path)
        if not test_seq:
            print(f"Warning: No landmark data in test file {test_path}")
            predictions.append(None)
            continue

        # compute distances to all train
        distances = []
        for train_path, train_label in train_files:
            if include_obj or include_sound:
                train_seq = extract_features(train_path, include_obj, include_sound)
            else:
                train_seq = extract_all_landmarks(train_path)
            if not train_seq:
                print(f"Warning: No landmark data in train file {train_path}")
                continue
                
            distance = DTW(test_seq, train_seq)
            distances.append((distance, train_label))
        
        if distances:
            closest_action = min(distances, key=lambda x: x[0])[1]  # Get action with smallest distance
            print(f"Predicted action for {os.path.basename(test_path)}: {closest_action}")
            predictions.append(closest_action)
        else:
            predictions.append(None)
    
    return predictions, true_labels

# --- k-NN-DTW with clustering ---
def classify_with_knn_dtw(train_files, test_files, k=3, n_clusters=3, include_obj=False, include_sound=False, show_plot=True):
    """
    Performs kNN classification with DTW and clustering.
    Args:
        train_files (list): List of (file_path, action) for training data.
        test_files (list): List of (file_path, action) for test data.
        k (int): Number of neighbors to consider (default is 3).
        include_obj: whether to append obj_coords.
        include_sound: whether to append sound_amp.
    Returns:
        list: Predicted actions, true actions.
    """
    # Prepare train data
    X_train, y_train = [], []
    for path, label in train_files:
        seq = (extract_features(path, include_obj, include_sound)
               if (include_obj or include_sound)
               else extract_all_landmarks(path))
        X_train.append(seq)
        y_train.append(label)

    # Prepare test data
    X_test, y_test = [], []
    for path, label in test_files:
        seq = (extract_features(path, include_obj, include_sound)
               if (include_obj or include_sound)
               else extract_all_landmarks(path))
        X_test.append(seq)
        y_test.append(label)
    
    # Computing DTW distance matrix for train data
    print(f"Preparing data for clustering...")
    train_paths = [fp for fp, _ in train_files]
    distance_matrix, file_list = compute_dtw_distance_matrix(file_list = train_paths, sequences = X_train)
    if show_plot:
        plot_distance_matrix(distance_matrix, file_list, save_png=False, title="DTW matrix for clustering")

    # Clustering with k-medoids
    print(f"Clustering {len(X_train)} samples into {n_clusters} clusters...")
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    kmedoids.fit(distance_matrix)
    cluster_labels = kmedoids.labels_
    medoid_indices = kmedoids.medoid_indices_
    for cluster_id in range(n_clusters):
        cluster_samples = [y_train[i] for i in range(len(y_train)) if cluster_labels[i] == cluster_id]
        print(f"Cluster {cluster_id}: {cluster_samples}")

    # Classify each test
    predictions = []
    for i, test_seq in enumerate(X_test):
        print(f"\nClassifying test sample {i+1}/{len(X_test)}...")
        # Calculating DTW distances to medoids
        distances_to_medoids = [DTW(test_seq, X_train[medoid_idx]) for medoid_idx in medoid_indices]
        
        # Getting closest cluster
        closest_cluster = np.argmin(distances_to_medoids)
        
        # Getting cluster indices in closest cluster
        cluster_indices = [i for i, cl in enumerate(cluster_labels) if cl == closest_cluster]
        
        # Calculating DTW distances to all samples in cluster
        distances_to_cluster = [(DTW(test_seq, X_train[idx]), y_train[idx]) for idx in cluster_indices]
        
        # Sorting distances
        distances_to_cluster.sort(key=lambda x: x[0])
        cluster_labels_set = set(label for _, label in distances_to_cluster)
        print(f"Closest cluster {closest_cluster}: has {len(cluster_indices)} samples, labels: {cluster_labels_set}")

        # Getting k nearest neighbours
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