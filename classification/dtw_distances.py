#classification/vizualization.py

from data_utils import preprocess_data
import os
import json
import numpy as np
from fastdtw import fastdtw
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
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

def cluster_data(data, n_clusters=3):
    """
    Cluster the data using KMeans.
    Args:
        data (list): List of sequences.
        n_clusters (int): Number of clusters.
        
    Returns:
        list: Cluster labels for each sequence.
    """
    # Preprocess data for clustering
    features = preprocess_data(data)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    return clusters

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
    
    # Step 1: Cluster the training data
    cluster_labels = cluster_data(X_train, n_clusters)

    # Step 2: Create separate kNN classifiers for each cluster
    cluster_classifiers = {}
    predictions = []

    for cluster_id in range(n_clusters):
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) < k:
            print(f"Warning: Cluster {cluster_id} has {len(cluster_indices)} samples, less than k={k}.")
            continue
            
        # Get the training data for this cluster
        X_cluster = [X_train[i] for i in cluster_indices]
        y_cluster = [y_train[i] for i in cluster_indices]
        pdb.set_trace()
        # Create kNN classifier for this cluster
        classifier = KNeighborsClassifier(n_neighbors=min(k, len(X_cluster)), 
                                          algorithm='brute',
                                          metric=custom_dtw_metric)
        classifier.fit(X_cluster, y_cluster)
        
        # Store the classifier
        cluster_classifiers[cluster_id] = (classifier, X_cluster)
    pdb.set_trace()
    # Step 3: Classify test samples
    for x_test in X_test:
        # Find the closest cluster for this test sample
        min_distance = float('inf')
        best_cluster = -1
        
        for cluster_id, (_, X_cluster) in cluster_classifiers.items():
            # Calculate average distance to samples in this cluster
            distances = [DTW(x_test, x_cluster) for x_cluster in X_cluster]
            avg_distance = np.mean(distances)
            
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_cluster = cluster_id
        
        if best_cluster == -1:
            # If no valid cluster found, use the entire training set
            print("Warning: No valid cluster found for test sample, using entire training set.")
            classifier = KNeighborsClassifier(n_neighbors=min(k, len(X_train)), 
                                             algorithm='brute',
                                             metric=custom_dtw_metric)
            classifier.fit(X_train, y_train)
            pred = classifier.predict([x_test])[0]
        else:
            # Use the classifier for the best cluster
            classifier, _ = cluster_classifiers[best_cluster]
            pred = classifier.predict([x_test])[0]
        
        predictions.append(pred)
    
    return np.array(predictions), np.array(y_test)
    return predictions, y_test