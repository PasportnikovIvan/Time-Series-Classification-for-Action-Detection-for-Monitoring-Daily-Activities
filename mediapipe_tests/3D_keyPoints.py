import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs
import numpy as np
import json
import os

landmarks_collection = {0: "nose",
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
                        11: "left shoulder",
                        12: "right shoulder",
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
                        23: "left hip",
                        24: "right hip",
                        25: "left knee",
                        26: "right knee",
                        27: "left ankle",
                        28: "right ankle",
                        29: "left heel",
                        30: "right heel",
                        31: "left foot index",
                        32: "right foot index",
                        }

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# RealSense cam setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# img resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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
    closest_points = sorted_depths[:quarter_index+1]
    
    # Calculate median
    return np.median(closest_points)

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

# getting pose with MediaPipe
frame_count = 0
data_to_save = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # making coords from standert values [0, 1] to pixel values
                x_px = int(landmark.x * IMAGE_WIDTH)
                y_px = int(landmark.y * IMAGE_HEIGHT)

                # check if coords are in im borders
                if 0 <= x_px < IMAGE_WIDTH and 0 <= y_px < IMAGE_HEIGHT:
                    # Get depth frame as numpy array
                    depth_array = np.asanyarray(depth_frame.get_data())

                    # Calculate median depth
                    median_depth = calculate_median_depth(depth_array, x_px, y_px)

                    landmarks.append([x_px, y_px, median_depth])
                    
                    print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, depth={depth_frame.get_distance(x_px, y_px)}, median_depth={median_depth}")
                else:
                    print(f"Landmark {idx} out of bounds with coordinates x={x_px}, y={y_px}")

             # Convert to head-relative coordinates
            head_landmark = landmarks[0]  # Nose
            head_relative_landmarks = convert_to_head_relative_coordinates(landmarks, head_landmark)
            
            print("Head-relative coordinates:")
            for idx, landmark in enumerate(head_relative_landmarks):
                print(f"{landmarks_collection[idx]}: {landmark}")

            # Save head and shoulders data every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                head_data = {
                    "head": head_relative_landmarks[0],
                    "left_shoulder": head_relative_landmarks[11],
                    "right_shoulder": head_relative_landmarks[12]
                }
                data_to_save.append(head_data)
                print(f"Saved frame {frame_count // 30} data")

        cv2.imshow("MediaPipe Pose", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pipeline.stop()
cv2.destroyAllWindows()

# Save collected data to file
output_file = "pose_data.json"
with open(output_file, "w") as f:
    json.dump(data_to_save, f)
print(f"Data saved to {output_file}")