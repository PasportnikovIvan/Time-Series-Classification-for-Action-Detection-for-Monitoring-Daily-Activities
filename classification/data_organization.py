# /classification/data_organization.py
import json
import numpy as np
from collections import defaultdict

class DataOrganizer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.movement_vectors = defaultdict(list)

    def read_and_extract_data(self, file_path):
        """Read JSON file and extract landmark data."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract movement type from the header
        movement_type = data['header']['action']
        
        # Extract landmarks from each frame
        for frame in data['data']:
            frame_landmarks = []
            for landmark_name, coordinates in frame['landmarks'].items():
                frame_landmarks.extend(coordinates)  # Flatten [x, y, z] into a single list
            self.movement_vectors[movement_type].append(frame_landmarks)

    def extract_all_data(self):
        """Extract data from all files and organize into movement vectors."""
        for file in self.file_paths:
            self.read_and_extract_data(file)

    def pad_sequences(self, start=10, end=90):
        """Adjust sequences to remove first and last 10 frames."""
        for movement, vectors in self.movement_vectors.items():
            adjusted_vectors = []
            for vector in vectors:
                # Ensure the sequence has at least 100 frames
                if len(vector) >= 100:
                    adjusted_vector = vector[start:end]
                    adjusted_vectors.append(adjusted_vector)
                else:
                    # Handle sequences with less than 100 frames
                    print(f"Sequence with less than 100 frames in {movement} data.")
            self.movement_vectors[movement] = np.array(adjusted_vectors)

    def save_vectors(self, output_dir):
        """Save movement vectors to files."""
        for movement, vectors in self.movement_vectors.items():
            np.save(f"{output_dir}/{movement}_vectors.npy", vectors)

    def get_movement_data(self, movement):
        """Get processed data for a specific movement."""
        return self.movement_vectors.get(movement, None)
    
if __name__ == "__main__":
    file_paths = [
        'dataset/cameraLandmarks/falling/falling_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/falling/falling_02_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/falling/falling_03_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/standing/standing_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/standing/standing_02_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/standing/standing_03_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/sitting/sitting_01_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/sitting/sitting_02_cameralandmarksdata_ivan.json',
        'dataset/cameraLandmarks/sitting/sitting_03_cameralandmarksdata_ivan.json'
    ]
    organizer = DataOrganizer(file_paths)
    organizer.extract_all_data()
    organizer.pad_sequences(start=10, end=90)
    organizer.save_vectors('classification/output_vectors')
    
    standing_data = organizer.get_movement_data('standing')
    if standing_data is not None:
        print(f"Standing data shape: {standing_data.shape}")