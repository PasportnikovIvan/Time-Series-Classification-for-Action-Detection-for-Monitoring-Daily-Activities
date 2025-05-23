#classification/main.py

from data_utils import get_first_session_files, get_all_session_files, collect_all_files, load_split_files
from visualization import plot_nose_trajectory, simulate_full_body_trajectory, plot_nose_velocity
from dtw_distances import compute_dtw_distances, classify_with_dtw, compute_dtw_distance_matrix, plot_distance_matrix, classify_with_knn_dtw
from cross_validation import cross_validate_knn_dtw
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import pdb

# Paths and parameters
RAWS_DIR         = 'dataset/raws'
PROCESSED_DIR    = 'dataset/processed'
SPLITS_DIR       = 'splits/global_tt'
ALL_FILES_DIR    = 'splits/all_files'
TRAIN_SPLIT      = os.path.join(SPLITS_DIR, 'train_files.txt')
TEST_SPLIT       = os.path.join(SPLITS_DIR, 'test_files.txt')
ACTIONS          = ['standing', 'sitting', 'walking', 'bad_walking', 'falling_floor', 'falling_bed', 'lying_bed', 'lying_floor']
ACTIONS_TESTS    = ['timed-up-and-go', 'sppb']
ACTION_COLORS    = {
    'timed-up-and-go': 'b',
    'SPPB': 'b',
    'falling_floor': 'r',
    'falling_bed': 'orange',
    'bad_walking': 'purple',
    'walking': 'y',
    'lying_floor': 'g',
    'lying_bed': 'r',
    'sitting': 'g',
    'standing': 'm'
}
N_CLUSTERS       = 8
K_NEIGHBORS      = 3
N_FOLDS          = 5

def visualize_trajectories(actions, camera_dir, global_dir, action_colors, session = 1, nose=False, body=False, show_camera=False):
    """
    Visualizes the trajectories for the first sessions of actions.
    Args:
        actions (list): List of actions to visualize.
        camera_dir (str): Path to the cameraLandmarks directory.
        global_dir (str): Path to the globalLandmarks directory.
        action_colors (dict): Dictionary mapping actions to colors.
        session (int): Chosen session for visualization file.
    """
    for action in actions:
        camera_file, global_file = get_first_session_files(action, camera_dir, global_dir, session=session)
        if camera_file and global_file:
            print(f"Visualization for: {action}")
            if nose:
                if show_camera:
                    plot_nose_trajectory(camera_file, f"{action} - Camera coordinates", color=action_colors[action])
                plot_nose_trajectory(global_file, f"{action} - Global coordinates", color=action_colors[action])
            if body:
                if show_camera:
                    simulate_full_body_trajectory(camera_file, f"{action} - Camera Trajectory Simulation", color=action_colors[action])
                simulate_full_body_trajectory(global_file, f"{action} - Global Trajectory Simulation", color=action_colors[action])
        else:
            print(f"Files for {action} not found.")

def visualize_velocities(actions, global_dir):
    """
    Visualizes the velocities for the first sessions of actions.
    Args:
        actions (list): List of actions to visualize.
        global_dir (str): Path to the globalLandmarks directory.

    """
    for action in actions:
        global_file = get_all_session_files(action, global_dir)
        if global_file:
            print(f"Velocity visualization for: {action}")
            plot_nose_velocity(global_file, action)
        else:
            print(f"No valid files for velocity visualization of {action}")

def compute_and_print_dtw_distances(reference_action, camera_dir, global_dir, actions):
    """
    Computes and prints DTW distances for a reference action against all other actions.
    Args:
        reference_action (str): Action to use as reference for DTW distance computation.
        camera_dir (str): Path to the cameraLandmarks directory.
        global_dir (str): Path to the globalLandmarks directory.
        actions (list): List of actions to compute DTW distances against.
    """
    ref_camera_file, ref_global_file = get_first_session_files(reference_action, camera_dir, global_dir)
    if ref_global_file:
        print(f"Computing DTW distances with reference: {ref_global_file}")
        all_files = collect_all_files(actions, global_dir)
        distances = compute_dtw_distances(ref_global_file, all_files)
        
        # Print sorted DTW distances
        print("\nDTW Distances (sorted):")
        for file_path, distance, action_name in distances:
            print(f"Action: {action_name:<25} DTW Distance: {distance:.2f},   File: {file_path}")
    else:
        print(f"No valid reference file found for {reference_action}")

def compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False, include_obj=False, include_sound=False):
    """
    Computes and plots the distance matrix for all files.
    Args:
        all_files (list): List of paths to JSON files.
        use_all_landmarks (bool): Whether to use all landmarks or just the nose.
    """
    distance_matrix, file_list = compute_dtw_distance_matrix(all_files, use_all_landmarks=use_all_landmarks, include_obj=include_obj, include_sound=include_sound)
    plot_distance_matrix(distance_matrix, file_list, save_png, cmap='viridis')

def perform_1NN_classification(train_files, test_files, include_obj=False, include_sound=False):
    """
    Performs 1NN classification using DTW on the provided training and testing files.
    Args:
        train_files (list): List of training file paths.
        test_files (list): List of testing file paths.
    """
    # DTW Classification
    predictions, trues = classify_with_dtw(train_files, test_files, include_obj=include_obj, include_sound=include_sound)

    # Filter out None predictions (if any)
    valid_pairs = [(p, t) for p, t in zip(predictions, trues) if p is not None]
    if valid_pairs:
        pred_valid, true_valid = zip(*valid_pairs)
        # Evaluation
        accuracy = accuracy_score(true_valid, pred_valid)
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(true_valid, pred_valid))
        print("Confusion Matrix:")
        print(confusion_matrix(true_valid, pred_valid))
    else:
        print("No valid predictions made.")

def perform_kNN_dtw_with_clusters(train_files, test_files, k=3, n_clusters=3, include_obj=False, include_sound=False, show_plot=False):
    """
    Performs kNN classification using DTW metric on the provided training and testing files.
    Args:
        train_files (list): List of (file_path, action) for training data.
        test_files (list): List of (file_path, action) for test data.
        k (int): Number of neighbors to consider (default is 3).
    """
    # Cluster → kNN
    predictions, trues = classify_with_knn_dtw(train_files, test_files, k=k, n_clusters=n_clusters, include_obj=include_obj, include_sound=include_sound, show_plot=show_plot)

    # Accuracy
    valid_pairs = [(p, t) for p, t in zip(predictions, trues) if p is not None]
    if valid_pairs:
        pred_valid, true_valid = zip(*valid_pairs)
        # Evaluation
        accuracy = accuracy_score(true_valid, pred_valid)
        print(f"\nClustered kNN-DTW Accuracy (k={k}, clusters={n_clusters}): {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(true_valid, pred_valid))
        print("Confusion Matrix:")
        print(confusion_matrix(true_valid, pred_valid))
    else:
        print("No valid predictions made.")

def perform_cross_validation_knn_dtw(global_dir, actions, n_folds=5, k=3, n_clusters=3, include_obj=False, include_sound=False, show_plot=True):
    """
    Performs k-fold cross-validation for k-NN classification using DTW metric.
    Args:
        global_dir (str): Path to the globalLandmarks directory.
        actions (list): List of actions to consider.
        n_splits (int): Number of folds for cross-validation (default is 5).
        k (int): Number of neighbors to consider (default is 3).
        n_clusters (int): Number of clusters for k-NN classification (default is 3).
    """
    accuracies = cross_validate_knn_dtw(global_dir, actions, n_splits=n_folds, k=k, n_clusters=n_clusters, include_obj=include_obj, include_sound=include_sound, show_plot=show_plot)
    print(f"\n{n_folds}-fold CV mean accuracy: {np.mean(accuracies):.3f}  σ={np.std(accuracies):.3f}")

def main():
    # 1) Load split files
    train = load_split_files(TRAIN_SPLIT, PROCESSED_DIR)
    test  = load_split_files(TEST_SPLIT,  PROCESSED_DIR)
    all_files = collect_all_files(ACTIONS, directory = PROCESSED_DIR)

    # 2) Visualize trajectories and velocities
    visualize_trajectories(ACTIONS, RAWS_DIR, PROCESSED_DIR, ACTION_COLORS, session = 4, nose=False, body=True, show_camera=False)
    visualize_velocities(ACTIONS, PROCESSED_DIR)
    compute_and_print_dtw_distances('sitting', RAWS_DIR, PROCESSED_DIR, ACTIONS)
    compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False)
    compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False, include_obj=True, include_sound=False)
    compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False, include_obj=True, include_sound=True)

    # 3) 1-NN + DTW (motion only)
    perform_1NN_classification(train, test)
    # (motion + object)
    perform_1NN_classification(train, test, include_obj=True, include_sound=False)
    # (motion + object + sound)
    perform_1NN_classification(train, test, include_obj=True, include_sound=True)

    # 4) k-NN + DTW (motion only)
    perform_kNN_dtw_with_clusters(train, test, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, show_plot=True)
    # (motion + object)
    perform_kNN_dtw_with_clusters(train, test, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, include_obj=True, include_sound=False, show_plot=True)
    # (motion + object + sound)
    perform_kNN_dtw_with_clusters(train, test, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, include_obj=True, include_sound=True, show_plot=True)

    # 5) Cross-validation (motion only)
    perform_cross_validation_knn_dtw(PROCESSED_DIR, ACTIONS, n_folds=N_FOLDS, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, show_plot=False)
    # (motion + object)
    perform_cross_validation_knn_dtw(PROCESSED_DIR, ACTIONS, n_folds=N_FOLDS, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, include_obj=True, include_sound=False, show_plot=False)
    # (motion + object + sound)
    perform_cross_validation_knn_dtw(PROCESSED_DIR, ACTIONS, n_folds=N_FOLDS, k=K_NEIGHBORS, n_clusters=N_CLUSTERS, include_obj=True, include_sound=True, show_plot=False)

if __name__ == "__main__":
    main()