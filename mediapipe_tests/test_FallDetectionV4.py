import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import json

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
previous_positions = []
fall_detected = False
fall_threshold = 0.7  # Adjust this based on testing

# Data collection setup
frame_count = 0
data_to_save = []

# Process video frames
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            key_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP, 
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
                ]

            # Calculate 3D coordinates with depth
            landmarks_data = {}
            for landmark in key_landmarks:
                landmark_data = results.pose_landmarks.landmark[landmark]
                x, y = int(landmark_data.x * IMAGE_WIDTH), int(landmark_data.y * IMAGE_HEIGHT)
                if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                    # depth_image = np.asanyarray(depth_frame.get_data())
                    depth = calculate_median_depth(depth_image, x, y)  # Use the depth data
                   
                    previous_positions.append((x, y, depth * depth_scale)) # Convert depth to meters

            # Save data to JSON every 60 frames
            frame_count += 1
            if frame_count >= 60:
                landmarks_data[landmarks_collection[landmark]] = {
                    "x": previous_positions[-1][0], 
                    "y": previous_positions[-1][1], 
                    "depth": previous_positions[-1][2]
                    }
                data_to_save.append(landmarks_data)
                frame_count = 0

            # Check for fall detection logic
            if len(previous_positions) >= 2:
                _, _, prev_depth = previous_positions[-2]
                _, _, current_depth = previous_positions[-1]
                depth_diff = abs(current_depth - prev_depth)

                # Detect if depth change exceeds threshold
                if depth_diff > fall_threshold:
                    fall_detected = True
                else:
                    fall_detected = False
                    
                # Draw fall detection status on color frame
                cv2.putText(color_image, f"Fall Detected: {fall_detected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not fall_detected else (0, 0, 255), 2)

            # Show 3D coordinates as overlay
            for i, (x, y, depth) in enumerate(previous_positions[-3:]):
                cv2.putText(color_image, f"Point {i}: Depth={depth:.2f}m", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the image
        cv2.imshow('Fall Detection', color_image)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
pipeline.stop()
cv2.destroyAllWindows()

# Save collected data to file
output_file = "fall_detection_data.json"
with open(output_file, "w") as json_file:
    json.dump(data_to_save, json_file, indent=4)