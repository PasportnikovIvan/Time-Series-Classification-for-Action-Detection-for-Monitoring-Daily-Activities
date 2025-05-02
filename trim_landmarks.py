import os
import json

def trim_json_data(file_path, start_index=9, end_index=89):
    """
    Trim the JSON data to keep only a specific range of frames.

    Args:
        file_path (str): Path to the JSON file.
        start_index (int): Index of the first frame to keep (default is 9, 10 timestamp).
        end_index (int): Index of the last frame to keep (default is 89, 90 timestamp).
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if the data has enough frames
    if 'data' in data and len(data['data']) > end_index:
        # Trim the data
        data['data'] = data['data'][start_index:end_index + 1]
    else:
        print(f"Warning: file {file_path} consists of less {end_index + 1} frames. Trim is passed.")
        return
    
    # Save the trimmed data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Trimmed and saved: {file_path}")

def trim_all_files_in_directory(directory, start_index=10, end_index=89):
    """
    Trim all JSON files in a directory to keep only a specific range of frames.

    Args:
        directory (str): Path to the directory containing JSON files.
        start_index (int): Index of the first frame to keep.
        end_index (int): Index of the last frame to keep.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                trim_json_data(file_path, start_index, end_index)

# Path to directories with data
camera_directory = 'dataset/cameraLandmarks'
global_directory = 'dataset/globalLandmarks'

if __name__ == "__main__":
    trim_all_files_in_directory(camera_directory)
    trim_all_files_in_directory(global_directory)