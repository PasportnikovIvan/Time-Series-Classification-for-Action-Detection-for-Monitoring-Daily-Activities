import os
import json
import numpy as np
import pdb

def convert_camera_to_global(camera_dir, global_dir):
    """
    Fixing the camera landmarks to global landmarks.

    Args:
        camera_dir (str): Path to the directory containing camera landmarks files.
        global_dir (str): Path to the directory where global landmarks files will be saved.
    """
    # Create the global directory if it doesn't exist
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)

    # Go through all files in the camera directory
    for root, dirs, files in os.walk(camera_dir):
        for file in files:
            if file.endswith("_cameralandmarksdata_ivan.json"):
                camera_file_path = os.path.join(root, file)

                # Read the camera landmarks data
                with open(camera_file_path, 'r') as f:
                    data = json.load(f)

                # Extract the rotation matrix and translation vector from the header
                rotation_matrix = np.array(data['header']['matrix']['rotation'])
                translation_vector = np.array(data['header']['matrix']['translation']).reshape(3, 1)

                # Create a new structure for global landmarks
                global_data = {
                    "header": data['header'],  # Copy header from camera data
                    "data": []
                }

                # For each frame in the camera data, convert landmarks to global coordinates
                for frame in data['data']:
                    global_frame = {
                        "timestamp": frame['timestamp'],
                        "landmarks": {}
                    }
                    for key, cam_coords in frame['landmarks'].items():
                        # Covert cam_coords to numpy array and reshape it
                        cam_coords = np.array(cam_coords).reshape(3, 1)
                        # Apply the formula: P_global = R^T * (P_cam - T)
                        global_coords = np.dot(rotation_matrix.T, (cam_coords - translation_vector))
                        # Save the global coordinates in the new structure
                        global_frame['landmarks'][key] = global_coords.flatten().tolist()
                    global_data['data'].append(global_frame)

                # Get the relative path for the global directory
                relative_path = os.path.relpath(root, camera_dir)
                global_action_dir = os.path.join(global_dir, relative_path)
                if not os.path.exists(global_action_dir):
                    os.makedirs(global_action_dir)
                global_file_path = os.path.join(
                    global_action_dir,
                    file.replace("cameralandmarksdata", "globallandmarksdata")
                )

                # Save the global landmarks data to the new directory
                with open(global_file_path, 'w') as f:
                    json.dump(global_data, f, indent=4)
                print(f"Converted and saved: {global_file_path}")

# Usage
camera_directory = 'dataset/cameraLandmarks'
global_directory = 'dataset/globalLandmarks'
convert_camera_to_global(camera_directory, global_directory)