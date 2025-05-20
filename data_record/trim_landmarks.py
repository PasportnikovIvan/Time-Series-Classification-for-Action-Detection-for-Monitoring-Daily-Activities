#data_record/trim_landmarks.py
# # Trim JSON data to keep only a specific range of frames

import os
import json

def trim_json_data(data, start_index=10, end_index=89):
    """
    Trim the JSON data to keep only a specific range of frames.

    Args:
        file_path (str): Path to the JSON file.
        start_index (int): Index of the first frame to keep (default is 9, 10 timestamp).
        end_index (int): Index of the last frame to keep (default is 89, 90 timestamp).
    """
    return data[start_index:end_index + 1]