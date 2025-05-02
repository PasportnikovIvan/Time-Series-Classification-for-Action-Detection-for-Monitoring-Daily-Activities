#classification/main.py
import os
import json
import matplotlib.pyplot as plt
from vizualization import plot_nose_trajectory
from simulation_visualization import simulate_trajectory
from dtw_distances import plot_nose_velocity, compute_dtw_distances

def get_first_session_files(action, camera_dir, global_dir, subject='ivan'):
    """
    Finds best matching files for a given action in cameraLandmarks and globalLandmarks directories.

    Args:
        action (str): Name of the action (e.g., 'falling', 'lying', etc.).
        camera_dir (str): Path to the cameraLandmarks directory.
        global_dir (str): Path to the globalLandmarks directory.
        subject (str): Name of the subject (default is 'ivan').

    Returns:
        tuple: Paths to the camera and global files for the action, or (None, None) if not found.
    """
    # Making paths to directories
    action_camera_dir = os.path.join(camera_dir, action)
    action_global_dir = os.path.join(global_dir, action)
    
    # Getting all files in the directories
    camera_files = sorted([f for f in os.listdir(action_camera_dir) 
                          if f.endswith(f'_cameralandmarksdata_{subject}.json')])
    global_files = sorted([f for f in os.listdir(action_global_dir) 
                          if f.endswith(f'_globallandmarksdata_{subject}.json')])
    
    # Going through all files and checking for 'nose' data
    for camera_file, global_file in zip(camera_files, global_files):
        camera_path = os.path.join(action_camera_dir, camera_file)
        global_path = os.path.join(action_global_dir, global_file)
        
        # Reading JSON files
        with open(camera_path, 'r') as f:
            camera_data = json.load(f)
        with open(global_path, 'r') as f:
            global_data = json.load(f)
        
        # Checking if 'nose' data is present in all frames
        camera_valid = all('nose' in frame['landmarks'] for frame in camera_data['data'])
        global_valid = all('nose' in frame['landmarks'] for frame in global_data['data'])
        
        if camera_valid and global_valid:
            print(f"Found valid file pair for {action}: {camera_file}, {global_file}")
            return camera_path, global_path
        else:
            print(f"Skipping {camera_file} or {global_file}: 'nose' data missing in some frames")
    
    print(f"No valid files found for {action} with complete 'nose' data")
    return None, None

def get_all_session_files(action, global_dir, subject='ivan'):
    """
    Finds all valid files for a given action in globalLandmarks directory.

    Args:
        action (str): Name of the action (e.g., 'falling', 'lying', etc.).
        global_dir (str): Path to the globalLandmarks directory.
        subject (str): Name of the subject (default is 'ivan').

    Returns:
        list: Paths to all valid global files for the action.
    """
    action_global_dir = os.path.join(global_dir, action)
    global_files = sorted([f for f in os.listdir(action_global_dir) 
                          if f.endswith(f'_globallandmarksdata_{subject}.json')])
    
    valid_files = []
    for global_file in global_files:
        global_path = os.path.join(action_global_dir, global_file)
        with open(global_path, 'r') as f:
            global_data = json.load(f)
        
        if all('nose' in frame['landmarks'] for frame in global_data['data']):
            valid_files.append(global_path)
        else:
            print(f"Skipping {global_file}: 'nose' data missing in some frames")
    
    return valid_files

def collect_all_files(actions, global_dir, subject='ivan'):
    """
    Collects all valid files across all actions in globalLandmarks directory.

    Args:
        actions (list): List of action names.
        global_dir (str): Path to the globalLandmarks directory.
        subject (str): Name of the subject (default is 'ivan').

    Returns:
        list: Paths to all valid global files across all actions.
    """
    all_files = []
    for action in actions:
        all_files.extend(get_all_session_files(action, global_dir, subject))
    return all_files

def main():
    # Paths to directories with data
    camera_directory = 'dataset/cameraLandmarks' # Relative path to dataset from classification directory
    global_directory = 'dataset/globalLandmarks'
    actions = ['falling', 'lying', 'sitting', 'standing', 'standing-side-by-side']
    subject = 'ivan' # Replace with the subject name you want to visualize
    action_colors = {
        'falling': 'r',
        'lying': 'b',
        'sitting': 'g',
        'standing': 'm',
        'standing-side-by-side': 'c'
    }

    # Visualize trajectories (first valid session)
    for action in actions:
        camera_file, global_file = get_first_session_files(action, camera_directory, global_directory, subject)
        if camera_file and global_file:
            print(f"Visualization for: {action}")
            # Vizualization for camera coordinates (red color)
            plot_nose_trajectory(camera_file, f"{action} - Camera coordinates", color=action_colors[action])
            # Vizualization for global coordinates (blue color)
            plot_nose_trajectory(global_file, f"{action} - Global coordinates", color=action_colors[action])
            # Simulation visualization for camera coordinates
            simulate_trajectory(camera_file, f"{action} - Camera Trajectory Simulation", color=action_colors[action])
            # Simulation visualization for global coordinates
            simulate_trajectory(global_file, f"{action} - Global Trajectory Simulation", color=action_colors[action])
        else:
            print(f"Files for {action} not found.")
    
    # Visualize velocities (all valid sessions)
    plt.figure(figsize=(12, 8))
    for action in actions:
        global_files = get_all_session_files(action, global_directory, subject)
        if global_files:
            print(f"Velocity visualization for: {action}")
            plot_nose_velocity(global_files, action, color=action_colors[action])
        else:
            print(f"No valid files for velocity visualization of {action}")
    plt.show()

    # Compute DTW distances
    reference_action = 'standing'  # Choose reference action
    ref_camera_file, ref_global_file = get_first_session_files(reference_action, camera_directory, global_directory, subject)
    if ref_global_file:
        print(f"Computing DTW distances with reference: {ref_global_file}")
        all_files = collect_all_files(actions, global_directory, subject)
        distances = compute_dtw_distances(ref_global_file, all_files)
        
        # Print sorted DTW distances
        print("\nDTW Distances (sorted):")
        for file_path, distance, action_name in distances:
            print(f"File: {file_path}, Action: {action_name}, DTW Distance: {distance:.2f}")
    else:
        print(f"No valid reference file found for {reference_action}")

if __name__ == "__main__":
    main()