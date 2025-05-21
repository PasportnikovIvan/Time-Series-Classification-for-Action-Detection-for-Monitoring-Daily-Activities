import json
import numpy as np
from scipy.interpolate import interp1d

# Constants
ACTION = 'walking'
SESSION = '10'
SUBJECT = 'ivan'
LANDMARKS_DIR = f'dataset/processed'

INPUT_FILE = f'{LANDMARKS_DIR}/{ACTION}/{ACTION}_{SESSION}_globallandmarksdata_{SUBJECT}.json'
OUTPUT_FILE_CLEAN = f'{ACTION}_{SESSION}_cleaned.json'
OUTPUT_FILE_REPAIRED = f'{ACTION}_{SESSION}_repaired.json'

DISTANCE_THRESHOLD = 0.5  # Meters

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def detect_misdetections(data):
    misdetection_indices = []
    prev_landmarks = None
    for i, frame in enumerate(data['data']):
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

def remove_misdetections(data, misdetection_indices):
    clean_data = [frame for i, frame in enumerate(data['data']) if i not in misdetection_indices]
    return {"metadata": data["metadata"], "data": clean_data}

def repair_misdetections(data, misdetection_indices):
    repaired_data = data['data'].copy()
    timestamps = np.array([frame['timestamp'] for frame in data['data']])
    
    for idx in misdetection_indices:
        # Find valid frames before and after
        before_idx = idx - 1
        after_idx = idx + 1
        while before_idx in misdetection_indices and before_idx >= 0:
            before_idx -= 1
        while after_idx in misdetection_indices and after_idx < len(data['data']):
            after_idx += 1
        
        if before_idx < 0 or after_idx >= len(data['data']):
            continue  # Skip if no valid frames available
        
        before_frame = data['data'][before_idx]
        after_frame = data['data'][after_idx]
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
    
    return {"metadata": data["metadata"], "data": repaired_data}

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Load data
    data = load_data(INPUT_FILE)
    
    # Detect misdetections
    misdetection_indices = detect_misdetections(data)

    print(f"Detected {len(misdetection_indices)} misdetections")
    if not misdetection_indices:
        print("No misdetections found.")
        return
    # Remove misdetections
    clean_data = remove_misdetections(data, misdetection_indices)
    save_data(clean_data, OUTPUT_FILE_CLEAN)
    print(f"Saved cleaned data to {OUTPUT_FILE_CLEAN}")
    
    # Repair misdetections
    repaired_data = repair_misdetections(data, misdetection_indices)
    save_data(repaired_data, OUTPUT_FILE_REPAIRED)
    print(f"Saved repaired data to {OUTPUT_FILE_REPAIRED}")

if __name__ == "__main__":
    main()