#data_record/utils.py
# # Utility functions for data collection and processing

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from config import *

#=============== METADATA ===============
def build_metadata(camera_matrix, distortion_coeffs, depth_scale):
    """
    Build metadata for the action.
    """
    metadata = {
        "action": ACTION_NAME,
        "session": ACTION_SESSION,
        "subject": ACTION_SUBJECT,
        "subject_age": SUBJECT_AGE,
        "subject_gender": SUBJECT_GENDER,
        "subject_health_status": SUBJECT_HEALTH_STATUS,
        "location": ACTION_LOCATION,
        "lighting_conditions": ACTION_LIGHTING_CONDITIONS,
        "camera_model": CAMERA_MODEL,
        "resolution": CAMERA_RESOLUTION,
        "frame_rate": CAMERA_FRAME_RATE,
        "recording_date": RECORDING_DATE,
        "notes": NOTES,
        "camera_intrinsics": {
            "camera_matrix": camera_matrix,
            "distortion": distortion_coeffs,
            "depth_scale": depth_scale
        },
    }
    return metadata

#=============== ROTATION MATRIX ===============
def rodrigues_to_matrix(rvec):
    """
    Convert a Rodrigues rotation vector to a 3x3 rotation matrix.
    """
    rot = R.from_rotvec(np.array(rvec))
    return rot.as_matrix()

#=============== DEPTH CALCULATION ===============
def calculate_median_depth(x, y, depth_frame, depth_scale, radius=5, min_valid_pixels=3):
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
        return np.median(sorted_depths[:max(1, len(sorted_depths) // 4)]) * depth_scale
    else: # If not enough valid pixels, return 0
        print(f"Not enough valid pixels for depth calculation at ({x}, {y}). Found: {len(depths)}")
        # return depth_frame[x, y] * depth_scale
        return 0.0 # Return 0 if no valid depths are found

#=============== PIXELS -> CAMERA COORDS ===============
def pixels_to_camera_coordinates(x, y, depth, camera_matrix):
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
#--------------- POINT ----------------
def convert_point_to_global(cam_point, rvec, tvec):
    """
    Convert a single 3D point from camera frame into the global ArUco frame.
    cam_point: (3,) array in camera coords (e.g., tvec of object marker)
    rvec, tvec: pose of GLOBAL marker (from solvePnP)
    """
    R_mat, _ = cv2.Rodrigues(rvec)
    cam_pt = np.array(cam_point).reshape(3,1)
    tvec = np.array(tvec).reshape(3,1)
    # P_global = R^T * (P_cam - t)
    gl = R_mat.T.dot(cam_pt - tvec)
    return gl.flatten().tolist()

#--------------- LANDMARKS ----------------
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