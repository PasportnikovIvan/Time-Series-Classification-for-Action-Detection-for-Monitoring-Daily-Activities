#data_record/detection_and_repair.py
# # Detection and Repair Module

import numpy as np
from config import *
from scipy.interpolate import interp1d

def detect_misdetections(data):
    misdetection_indices = []
    prev_landmarks = None
    for i, frame in enumerate(data):
        current_landmarks = frame['landmarks']
        if prev_landmarks:
            max_distance = 0
            for key in current_landmarks:
                if key in prev_landmarks:
                    prev_coords = np.array(prev_landmarks[key])
                    curr_coords = np.array(current_landmarks[key])
                    distance = np.linalg.norm(curr_coords - prev_coords)
                    max_distance = max(max_distance, distance)
            if max_distance > DISTANCE_THRESHOLD:
                misdetection_indices.append(i)
                print(f"Misdetection at frame {i}: Max distance {max_distance:.2f}m")
        prev_landmarks = current_landmarks
    return misdetection_indices

def repair_misdetections(data, misdetection_indices):
    repaired_data = data.copy()
    timestamps = np.array([frame['timestamp'] for frame in data])
    
    for idx in misdetection_indices:
        # Find valid frames before and after
        before_idx = idx - 1
        after_idx = idx + 1
        while before_idx in misdetection_indices and before_idx >= 0:
            before_idx -= 1
        while after_idx in misdetection_indices and after_idx < len(data):
            after_idx += 1
        
        if before_idx < 0 or after_idx >= len(data):
            continue  # Skip if no valid frames available
        
        before_frame = data[before_idx]
        after_frame = data[after_idx]
        t_before, t_after = timestamps[before_idx], timestamps[after_idx]
        t_current = timestamps[idx]
        
        # Interpolate landmarks
        interpolated_landmarks = {}
        for key in before_frame['landmarks']:
            if key in after_frame['landmarks']:
                x = [t_before, t_after]
                y = np.array([before_frame['landmarks'][key], after_frame['landmarks'][key]])
                f = interp1d(x, y, axis=0)
                interpolated_landmarks[key] = f(t_current).tolist()
        
        repaired_data[idx]['landmarks'] = interpolated_landmarks
        print(f"Repaired frame {idx} using interpolation")
    
    return repaired_data