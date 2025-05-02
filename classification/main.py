import os
import json
from vizualization import plot_nose_trajectory

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

def main():
    # Path to directories with data
    camera_directory = 'dataset/cameraLandmarks'  # Relative path to dataset from classification directory
    global_directory = 'dataset/globalLandmarks'
    actions = ['falling', 'lying', 'sitting', 'standing', 'standing-side-by-side']
    subject = 'ivan'  # Replace with the subject name you want to visualize

    # Going through each action and visualizing the data
    for action in actions:
        camera_file, global_file = get_first_session_files(action, camera_directory, global_directory, subject)
        if camera_file and global_file:
            print(f"Vizualization for: {action}")
            # Vizualization for camera coordinates (red color)
            plot_nose_trajectory(camera_file, f"{action} - Camera coordinates", color='r')
            # Vizualization for global coordinates (blue color)
            plot_nose_trajectory(global_file, f"{action} - Global coordinates", color='b')
        else:
            print(f"Files for {action} not found.")

if __name__ == "__main__":
    main()