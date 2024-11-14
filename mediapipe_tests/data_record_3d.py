import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import json
from time import time

landmarks_collection = {0: "Nose",
                        1: "left eye (inner)",
                        2: "left eye",
                        3: "left eye (outer)",
                        4: "right eye (inner)",
                        5: "right eye",
                        6: "right eye (outer)",
                        7: "left ear",
                        8: "right ear",
                        9: "mouth (left)",
                        10: "mouth (right)",
                        11: "L_Shoulder",
                        12: "R_Shoulder",
                        13: "left elbow",
                        14: "right elbow",
                        15: "left wrist",
                        16: "right wrist",
                        17: "left pinky",
                        18: "right pinky",
                        19: "left index",
                        20: "right index",
                        21: "left thumb",
                        22: "right thumb",
                        23: "L_Hip",
                        24: "R_Hip",
                        25: "left knee",
                        26: "right knee",
                        27: "L_Ankle",
                        28: "R_Ankle",
                        29: "left heel",
                        30: "right heel",
                        31: "left foot index",
                        32: "right foot index",
                        }

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
    window = depth_frame[max(0, y-radius):min(y+radius+1, IMAGE_HEIGHT),
                         max(0, x-radius):min(x+radius+1, IMAGE_WIDTH)]
    
    # Flatten the window and remove zero values
    depths = window.flatten()[window.flatten() != 0]
    
    # Sort depths and select 25% closest points
    sorted_depths = np.sort(depths)
    num_points = len(sorted_depths)
    quarter_index = int(num_points * 0.25)
    
    # Return the median of the closest points
    if quarter_index > 0:
        return np.median(sorted_depths[:quarter_index + 1])
    return 0  # Return 0 if no valid depths are found

def convert_to_head_relative_coordinates(landmarks, head_landmark):
    """
    Convert all landmarks to be relative to the head position.
    
    Args:
    landmarks (list): List of landmarks with x, y, z coordinates
    head_landmark (list): Head landmark with x, y, z coordinates
    
    Returns:
    list: Landmarks with coordinates relative to the head
    """
    head_relative_landmarks = []
    
    for landmark in landmarks:
        relative_x = landmark[0] - head_landmark[0]
        relative_y = landmark[1] - head_landmark[1]
        relative_z = landmark[2] - head_landmark[2]
        
        head_relative_landmarks.append([relative_x, relative_y, relative_z])
    
    return head_relative_landmarks

def detectFall(landmarks, previous_avg_shoulder_height):
    """
    The logic of fall detection. !TODO better!
    
    Args:
    landmarks (list): List of landmarks with x, y, z coordinates
    previous_avg_shoulder_height (float): Previous the average y-coordinate of the shoulders
    
    Returns:
    Tuple: Bool result of the detection and the average y-coordinate of the shoulders
    """

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
    
# Fall detection variables
previous_avg_shoulder_height = 0
fall_detected = False

# Data collection setup
last_save_time = 0
data_to_save = []

# Process video frames
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while True:
        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        
        # Convert images to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the color image to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Process the image with Mediapipe
        results = pose.process(rgb_image)
        
        # If landmarks are detected
        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                color_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=2),
            )
            
            # Calculate 3D coordinates with depth
            landmarks = []    
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x, y = int(landmark.x * IMAGE_WIDTH), int(landmark.y * IMAGE_HEIGHT)
                # check if coords are in im borders
                if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                    median_depth = calculate_median_depth(depth_image, x, y)  # Use the depth data
                    depth = median_depth * depth_scale # Convert depth to meters
                    landmarks.append((x, y, depth)) 

                    print(f"Landmark {idx}: x={x}, y={y}, depth={depth:.2f}m, name={landmarks_collection[idx]}")
                else:
                    print(f"Landmark {idx}: out of bounds")

            # Convert to head-relative coordinates
            head_landmark = landmarks[0]  # Nose
            head_relative_landmarks = convert_to_head_relative_coordinates(landmarks, head_landmark)
            
            print("=============  Head-relative coordinates:  =============")
            for idx, landmark in enumerate(head_relative_landmarks):
                print(f"Landmark {idx}: {landmark}, {landmarks_collection[idx]}")
            print("=======================  END  ==========================")

            current_time = time()
            # Fall detection and data saving each 2 seconds
            if (current_time - last_save_time) > 2:
                # Prepare data for saving
                required_indices = [0, 11, 12, 23, 24, 27, 28]
                
                # Forming head_data with using landmarks_collection for every required index
                head_data = {landmarks_collection[idx]: landmarks[idx] 
                            for idx in required_indices if idx < len(landmarks)}
                
                # TODO improve logic
                fall_detected, previous_avg_shoulder_height = detectFall(landmarks, previous_avg_shoulder_height)
             
                head_data["fall_detected"] = fall_detected  # Mark fall as detected in data
                
                data_to_save.append(head_data)  # Add the data to the collection
                last_save_time = current_time  # Reset the time counter
            
            # Draw fall detection status on color frame
            cv2.putText(color_image, "No Fall" if not fall_detected else "Fall Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not fall_detected else (0, 0, 255), 2)
            
            # Show 3D coordinates as overlay
            for i, (x, y, depth) in enumerate(landmarks[-3:]):
                cv2.putText(color_image, f"Point {i}: Depth={depth:.2f}m", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
        # Display the image
        cv2.imshow('Fall Detection', color_image)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release resources
pipeline.stop()
cv2.destroyAllWindows()

# Save collected data to file
output_file = "falling_ivan_01.json"
with open(output_file, "w") as f:
    json.dump(data_to_save, f, indent=4)
print(f"Data saved to {output_file}")