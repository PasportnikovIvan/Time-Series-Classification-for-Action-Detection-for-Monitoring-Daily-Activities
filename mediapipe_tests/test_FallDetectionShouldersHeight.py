import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from time import time
import json

def detectPose(color_frame, depth_frame, pose, display=True):
    # Convert images to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the color image to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Process the image with Mediapipe
    results = pose.process(rgb_image)
    
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * IMAGE_WIDTH), int(landmark.y * IMAGE_HEIGHT)
            if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                depth = calculate_median_depth(depth_image, x, y)  # Use the depth data
                landmarks.append((x, y, depth * depth_scale)) #  Convert depth to meters
            
        # Draw landmarks
        mp_drawing.draw_landmarks(
            color_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=2),
        )
        # Show 3D coordinates as overlay
        for i, (x, y, depth) in enumerate(landmarks[-3:]):
            cv2.putText(color_image, f"Point {i}: Depth={depth:.2f}m", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    else:
        return None, None  
    if display:
        cv2.imshow('Pose Landmarks', color_image)
    return color_image, landmarks

def calculate_median_depth(depth_frame, x, y, radius=5):
    """
    Calculate median depth for 25% of closest points around a given point.
    
    Args:
    depth_frame (np.ndarray): Depth frame
    x (int): X-coordinate of the center point
    y (int): Y-coordinate of the center point
    radius (int): Radius of the window around the center point
    
    Returns:
    float: Median depth value
    """
    # Extract a window around the point
    window = depth_frame[max(0, y-radius):min(y+radius+1, IMAGE_HEIGHT), max(0, x-radius):min(x+radius+1, IMAGE_WIDTH)]
    
    # Flatten the window and remove zero values
    depths = window.flatten()[window.flatten() != 0]
    
    # Sort depths and select 25% closest points
    sorted_depths = np.sort(depths)
    num_points = len(sorted_depths)
    quarter_index = int(num_points * 0.25)
    
    # Return the median of the closest points
    if quarter_index > 0:
        return np.median(sorted_depths[:quarter_index])
    return 0  # Return 0 if no valid depths are found

def detectFall(landmarks, previous_avg_shoulder_height):

    left_shoulder_y = landmarks[11][1]
    right_shoulder_y = landmarks[12][1]
    
    # Calculate the average y-coordinate of the shoulder
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    if previous_avg_shoulder_height == 0:
        previous_avg_shoulder_height = avg_shoulder_y
        return False, previous_avg_shoulder_height
    fall_threshold = previous_avg_shoulder_height * 1.5
    print(previous_avg_shoulder_height, avg_shoulder_y, end="\n")
    
    # Check if the average shoulder y-coordinate falls less than the previous average shoulder height
    if avg_shoulder_y > fall_threshold:
        previous_avg_shoulder_height = avg_shoulder_y
        return True, previous_avg_shoulder_height
    else:
        previous_avg_shoulder_height = avg_shoulder_y
        return False, previous_avg_shoulder_height

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# RealSense cam setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# img resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Get depth sensor's depth scale (conversion factor)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

time1 = 0
previous_avg_shoulder_height = 0
fall_detected = False

# Process video frames
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while True:
        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        
        color_image, landmarks = detectPose(color_frame, depth_frame, pose, display=True)
        
        time2 = time()
        
        if (time2 - time1) > 2: 
            if landmarks is not None:
                # print(landmarks)
                fall_detected, previous_avg_shoulder_height = detectFall(landmarks, previous_avg_shoulder_height)
                
                if fall_detected:                 
                    print("Fall detected!")
                     
            time1 = time2
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()