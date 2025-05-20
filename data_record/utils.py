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
        "resolution": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "frame_rate": f"{FRAME_RATE}fps",
        "audio_rate": f"{AUDIO_RATE}Hz",
        "audio_channels": f"{AUDIO_CHANNELS} channels",
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
def calculate_median_depth(x, y, depth_frame, depth_scale):
    """
    Calculate median depth for 25% of closest points around a given point.
    Args:
        x (int): X-coordinate of the center point
        y (int): Y-coordinate of the center point
        depth_frame (np.ndarray): Depth frame
        depth_scale (float): Depth scale factor
    Returns:
        float: Median depth of the closest points
    """
    # Get the window boundaries
    y_min = max(0, y - RADIUS_WINDOW)
    y_max = min(IMAGE_HEIGHT, y + RADIUS_WINDOW + 1)
    x_min = max(0, x - RADIUS_WINDOW)
    x_max = min(IMAGE_WIDTH, x + RADIUS_WINDOW + 1)

    # Extract a window around the point
    window = depth_frame[y_min:y_max, x_min:x_max]

    # Flatten the window and remove zero values
    depths = window[window > 0]
    
    # Return the median of the closest points
    if len(depths) > 0:
        # Sort the depths and take the k closest points
        k = max(1, int(len(depths) * MEDIAN_PERCENT))
        # Use partition with k-1 to get exactly k smallest values
        closest = np.partition(depths, k-1)[:k]
        return np.median(closest) * depth_scale
    else: # If not enough valid pixels, return 0
        print(f"Not enough valid pixels for depth calculation at ({x}, {y}). Found: {depth_frame[x, y] * depth_scale}")
        return depth_frame[x, y] * depth_scale

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