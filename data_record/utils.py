#data_record/utils.py
# # Utility functions for data collection and processing

import cv2
import numpy as np
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_MATRIX, PRINT_LANDMARKS_TO_CONSOLE

#=============== DEPTH CALCULATION ===============
def calculate_median_depth(x, y, depth_frame, radius=6, min_valid_pixels=3):
    """
    Calculate median depth for 25% of closest points around a given point.
    
    Args:
    depth_frame (np.ndarray): Depth frame
    x (int): X-coordinate of the center point
    y (int): Y-coordinate of the center point
    radius (int): Radius of the window around the center point
    min_valid_pixels (int): Minimum number of valid pixels to calculate median
    
    Returns:
    float: Median depth value
    """
    # Get the window boundaries
    y_min = max(0, y - radius)
    y_max = min(IMAGE_HEIGHT, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(IMAGE_WIDTH, x + radius + 1)

    # Extract a window around the point
    window = depth_frame[y_min:y_max, x_min:x_max]
    
    # Flatten the window and remove zero values
    depths = window[window > 0]
    
    # Return the median of the closest points
    if len(depths) > min_valid_pixels:
        # Sort the depths and take the first 25%
        sorted_depths = np.sort(depths)
        return np.median(sorted_depths[:max(1, len(sorted_depths) // 4)])
    else: # If not enough valid pixels, return 0
        print(f"Not enough valid pixels for depth calculation at ({x}, {y}). Found: {len(depths)}")
        return 0  # Return 0 if no valid depths are found

#=============== PIXELS -> CAMERA COORDS ===============
def pixels_to_camera_coordinates(x, y, depth, camera_matrix=CAMERA_MATRIX):
    """
    Terning pixel coords into camera axes.
    
    Args:
    x (int): X-coord in pixels.
    y (int): Y-coord in pixels.
    depth (float): depth in meters.
    
    Returns:
    tuple: Cam coords (X, Y, Z).
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # making meters
    cam_x = (x - cx) * depth / fx
    cam_y = (y - cy) * depth / fy
    cam_z = depth  # Z-is the same
    
    return cam_x, cam_y, cam_z

#=============== CAMERA -> GLOBAL COORDS ===============
def convert_landmarks_to_global(landmarks, rvec, tvec):
    """
    Convert local pose landmarks to global using ArUco marker.

    Args:
        landmarks (list): List of landmarks with x, y, z coordinates
        rvec (np.ndarray): ArUco marker rotation
        tvec (np.ndarray): ArUco marker position

    Returns:
        list: Global landmarks
    """
    if rvec is None or tvec is None:
        return landmarks
    
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_vector = tvec.reshape(3, 1)
    if PRINT_LANDMARKS_TO_CONSOLE:
        print("R matrix:", rotation_matrix.tolist(), "T vec:", translation_vector.flatten().tolist())

    global_landmarks = {}
    for key, cam_coords in landmarks.items():
        # Covert cam_coords to numpy array and reshape it
        cam_coords = np.array(cam_coords).reshape(3, 1)
        # Should be P_global = R^T * (P_cam - T)
        global_coords = np.dot(rotation_matrix.T, (cam_coords - translation_vector))
        # Save the global coordinates in the new structure
        global_landmarks[key] = global_coords.flatten().tolist()
    return global_landmarks