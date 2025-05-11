#classification/data_utils.py

import os
import json
import numpy as np

def load_split_files(split_file, global_dir):
    """
    Loads file paths from a split file (e.g., train_files.txt) and maps them to full paths.
    Args:
    split_file (str): Path to the split file (e.g., 'splits/global/train_files.txt').
    global_dir (str): Base directory for globalLandmarks.
    Returns:
    list: List of tuples (file_path, action).
    """
    with open(split_file, 'r') as f:
        lines = f.readlines()

    files = []
    for line in lines:
        rel_path = line.strip()
        action = rel_path.split(os.sep)[1]  # e.g., 'dataset/globalLandmarks/falling/...'
        full_path = os.path.join(global_dir, *rel_path.split(os.sep)[1:])
        files.append((full_path, action))
    return files

def get_first_session_files(action, camera_dir, global_dir, session=1):
    """
    Finds best matching files for a given action in cameraLandmarks and globalLandmarks directories.
    Args:
        action (str): Name of the action (e.g., 'falling', 'lying', etc.).
        camera_dir (str): Path to the cameraLandmarks directory.
        global_dir (str): Path to the globalLandmarks directory.
    Returns:
        tuple: Paths to the camera and global files for the action, or (None, None) if not found.
    """
    # Making paths to directories
    action_camera_dir = os.path.join(camera_dir, action)
    action_global_dir = os.path.join(global_dir, action)
    
    # Getting all files in the directories
    camera_files = sorted([f for f in os.listdir(action_camera_dir) 
                          if f.endswith(".json")])
    global_files = sorted([f for f in os.listdir(action_global_dir) 
                          if f.endswith(".json")])
    passed_files = 0
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
        if passed_files < (session - 1):
            passed_files += 1
            continue
        if camera_valid and global_valid:
            print(f"Found valid file pair for {action}: {camera_file}, {global_file}")
            return camera_path, global_path
        else:
            print(f"Skipping {camera_file} or {global_file}: 'nose' data missing in some frames")
    
    print(f"No valid files found for {action} with complete 'nose' data")
    return None, None

def get_all_session_files(action, directory):
    """
    Finds all valid files for a given action in globalLandmarks directory.
    Args:
        action (str): Name of the action (e.g., 'falling', 'lying', etc.).
        directory (str): Path to the globalLandmarks directory.
    Returns:
        list: Paths to all valid global files for the action.
    """
    action_global_dir = os.path.join(directory, action)
    global_files = sorted([f for f in os.listdir(action_global_dir) 
                          if f.endswith(".json")])
    
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

def collect_all_files(actions, directory):
    """
    Collects all valid files across all actions in globalLandmarks directory.
    Args:
        actions (list): List of action names.
        directory (str): Path to the globalLandmarks directory.
    Returns:
        list: Paths to all valid global files across all actions.
    """
    all_files = []
    for action in actions:
        all_files.extend(get_all_session_files(action, directory))
    return all_files