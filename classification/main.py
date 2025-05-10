#classification/main.py

from data_utils import get_first_session_files, get_all_session_files, collect_all_files, load_split_files
from visualization import plot_nose_trajectory, simulate_full_body_trajectory, plot_nose_velocity
from dtw_distances import compute_dtw_distances, classify_with_dtw, compute_dtw_distance_matrix, plot_distance_matrix
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import pdb

def visualize_trajectories(actions, camera_dir, global_dir, action_colors, session = 1):
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
            plot_nose_trajectory(camera_file, f"{action} - Camera coordinates", color=action_colors[action])
            plot_nose_trajectory(global_file, f"{action} - Global coordinates", color=action_colors[action])
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

def compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False):
    """
    Computes and plots the distance matrix for all files.
    Args:
        all_files (list): List of paths to JSON files.
        use_all_landmarks (bool): Whether to use all landmarks or just the nose.
    """
    distance_matrix, file_list = compute_dtw_distance_matrix(all_files, use_all_landmarks=use_all_landmarks)
    plot_distance_matrix(distance_matrix, file_list, save_png, cmap='viridis')

def perform_classification(train_files, test_files):
    """
    Performs classification using DTW on the provided training and testing files.
    Args:
        train_files (list): List of training file paths.
        test_files (list): List of testing file paths.
    """
    # DTW Classification
    predictions, true_labels = classify_with_dtw(train_files, test_files)

    # Filter out None predictions (if any)
    valid_pairs = [(p, t) for p, t in zip(predictions, true_labels) if p is not None]
    if valid_pairs:
        pred_valid, true_valid = zip(*valid_pairs)
        # Evaluation
        accuracy = accuracy_score(true_valid, pred_valid)
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(true_valid, pred_valid))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_valid, pred_valid))
    else:
        print("No valid predictions made.")

def main():
    # Paths to directories with data
    camera_directory = 'dataset/cameraLandmarks' # Relative path to dataset from classification directory
    global_directory = 'dataset/globalLandmarks'
    splits_dir = 'splits/global'
    all_files_dir = 'splits/all_files'
    actions = ['timed-up-and-go', 'falling', 'sitting', 'standing'] # List of actions to visualize
    action_colors = {
        'falling': 'r',
        'timed-up-and-go': 'b',
        'sitting': 'g',
        'standing': 'm'
    }
    all_files = collect_all_files(actions, directory = global_directory)
    # Load split files
    train_files = load_split_files(os.path.join(splits_dir, 'train_files.txt'), global_directory)
    test_files = load_split_files(os.path.join(splits_dir, 'test_files.txt'), global_directory)

    visualize_trajectories(actions, camera_directory, global_directory, action_colors, session = 2)

    visualize_velocities(actions, global_directory)

    compute_and_print_dtw_distances('standing', camera_directory, global_directory, actions)

    compute_and_plot_distance_matrix(all_files, use_all_landmarks=True, save_png=False)

    perform_classification(train_files, test_files)

if __name__ == "__main__":
    main()