#data_record/data_processing.py
# # Module for data processing data

import cv2
import mediapipe as mp
from config import *
from utils import pixels_to_camera_coordinates, calculate_median_depth

#=============== PROCESSING LANDMARKS FROM POSE ===============
def process_landmarks(results, color_image, depth_image, depth_scale, camera_matrix):
    """
    Processing the img to get landmarks and its coordinates.
    """
    if not results.pose_landmarks:
        return {}
    
    landmarks = {}
    pixel_coords = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        x, y = int(landmark.x * IMAGE_WIDTH), int(landmark.y * IMAGE_HEIGHT) # pixels
        # check if coords are in img borders
        if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
            median_depth = calculate_median_depth(x, y, depth_image, depth_scale) # depth data in meters
            pixel_coords[idx] = (x, y, median_depth)
            cam_coords = pixels_to_camera_coordinates(x, y, median_depth, camera_matrix)
            landmarks[LANDMARKS_COLLECTION[idx]] = cam_coords

            if PRINT_LANDMARKS_TO_CONSOLE:
                print(f"Landmark {idx}: pixel_coords(xy)={x, y}, camera_coords(xyz)={cam_coords}, name={LANDMARKS_COLLECTION[idx]}")
        else:
            print(f"Landmark {idx}: out of bounds")
     
    # Draw landmarks
    if VISUAL_LANDMARKS:
        mp.solutions.drawing_utils.draw_landmarks(
            color_image, 
            results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=3),
            mp.solutions.drawing_utils.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=1),
        )
        # Show 3D coordinates as overlay
        for i in [0, 11, 23]:
            if i in pixel_coords: 
                x, y, depth = pixel_coords[i]
                x_m, y_m, _ = landmarks[LANDMARKS_COLLECTION[i]]
                name = LANDMARKS_COLLECTION.get(i, f"Point {i}") 
                cv2.putText(color_image, f"{name}:({x_m:.1f}m, {y_m:.1f}m) d={depth:.2f}m", 
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return landmarks