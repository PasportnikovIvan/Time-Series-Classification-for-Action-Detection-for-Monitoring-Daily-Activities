import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import json
from time import time
from typing import List, Tuple, Dict

#======================== CONSTANTS ========================
#------------------------ Window ------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30
VISUAL_LANDMARKS = True
PRINT_LANDMARKS_TO_CONSOLE = False

#------------------------ Aruco ------------------------
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = aruco.DetectorParameters()
MARKER_SIZE = 0.0861  # Marker size in METERS
MARKER_POINTS = np.array(
    [
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
        [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    ],
    dtype=np.float32,
)
# 1280x720
CAMERA_MATRIX = np.array([
    [642.33569336, 0.0,          641.48535156], 
    [0.0,          641.68328857, 370.55108643], 
    [0.0,          0.0,          1.0]
])
# # 640x480
# CAMERA_MATRIX = np.array([
#     [385.40142822, 0.0,          320.89123535], 
#     [0.0,          385.00997925, 246.3306427], 
#     [0.0,          0.0,          1.0]
# ])
DIST_COEFFS = np.array([-0.05550327, 0.06885497, 0.00032144, 0.00124271, -0.0222161])
    
#------------------------ Saving Action ------------------------
# Define constants for action
ACTION_TYPE = "lying"
ACTION_SESSION = "04"
ACTION_SUBJECT = "ivan"
NOTES = "lying to side, without baggy clothes for better model recognition"

FILE_NAME_LANDMARKS = f'{ACTION_TYPE}_{ACTION_SESSION}_cameralandmarksdata_{ACTION_SUBJECT}.json'
FILE_NAME_GLOBAL = f'{ACTION_TYPE}_{ACTION_SESSION}_globallandmarksdata_{ACTION_SUBJECT}.json'
PARAMETER_TIMESTEP = 0.1
ACTION_LENGTH = 100 # actions
#HEADER: action, subject, (tMatrix, rMatrix optionally for camera coordinates), location, session... etc = METADATA
METADATA = {
    "action": ACTION_TYPE,
    "session": ACTION_SESSION,
    "subject": ACTION_SUBJECT,
    "subject_age": "21",
    "subject_gender": "male",
    "subject_health_status": "healthy",
    "location": "bubenec_dorm",
    "lighting_conditions": "bright",
    "camera_model": "Intel RealSense D455",
    "resolution": "640x480",
    "frame_rate": "30fps",
    "recording_date": "2025-01-14",
    "notes": NOTES,
    "camera_intrinsics": {
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "fx": CAMERA_MATRIX[0, 0],
        "fy": CAMERA_MATRIX[1, 1],
        "cx": CAMERA_MATRIX[0, 2],
        "cy": CAMERA_MATRIX[1, 2],
        "distortion": DIST_COEFFS.tolist()
    },
    "matrix": {
        "rotation": [
            [-0.8670117071470875, 0.38487316993921966, 0.3164859281719007], 
            [-0.02207920202025121, 0.6048497900254076, -0.7960334417248871], 
            [-0.4977983612950547, -0.69715807000074, -0.5159141565462236]
        ],
        "translation": [
            -1.21286381,
            0.1149645,
            2.48923353
        ]
    }
}

#------------------------ Landmarks ------------------------
LANDMARKS_COLLECTION = {
    0: "nose",
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
    32: "right foot index"
}

#------------------------ Threshold for Misdetection ------------------------
POSITION_THRESHOLD = 0.9  # Max allowed movement in meters between frames

#=============== CAMERA SETUP AND MEDIAPIPE ===============
def setup_camera_and_pose():
    """
    Setup for RealSense camera and MediaPipe.
    """
    # RealSense cam setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, FRAME_RATE)
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, FRAME_RATE)
    profile = pipeline.start(config)
    
    # Get depth sensor's depth scale (conversion factor)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5, 
        model_complexity=2
    )
    return pipeline, pose, depth_scale

#=============== FRAME COLLECTION AND PROCESSING ===============
def collect_frame(pipeline, pose, start_time):
    """
    Collect a frame from the camera and process it with MediaPipe Pose.
    """
    # Capture frames from RealSense
    frames = pipeline.wait_for_frames()
    time_of_frame = frames.get_timestamp() / 1000.0 - start_time
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None, None, None
    
    # Convert images to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # ArUco markers detection
    rvec, tvec, _ = detect_aruco_markers(color_image)

    results = pose.process(rgb_image)
    return time_of_frame, color_image, depth_image, results, (rvec, tvec)

#=============== ARUCO MARKER DETECTION ===============
def detect_aruco_markers(image):
    """
    Detect ArUco markers and return their coords and orientation. 

    Args:
        image (np.ndarray): imput img

    Returns:
        tuple: Tuple of rotations (rvec), positions (tvec) and id of detected markers.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray_image, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None:
        return None, None, None
    
    for i, corner in enumerate(corners):
        _, rvec, tvec = cv2.solvePnP(MARKER_POINTS, corner, CAMERA_MATRIX, DIST_COEFFS)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.drawFrameAxes(image, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)
        if PRINT_LANDMARKS_TO_CONSOLE:
            print(f"ArUco ID={ids[i][0]}, Position={tvec.flatten()}, Rotation={rvec.flatten()}")
    return rvec, tvec, ids

#=============== DEPTH CALCULATION ===============
def calculate_median_depth(x, y, depth_frame, radius=5):
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
    
    # Return the median of the closest points
    if len(depths) > 0:
        sorted_depths = np.sort(depths)
        return np.median(sorted_depths[:len(sorted_depths) // 4])
    return 0  # Return 0 if no valid depths are found

#=============== PIXELS -> CAMERA COORDS ===============
def pixels_to_camera_coordinates(x, y, depth):
    """
    Terning pixel coords into camera axes.
    
    Args:
    x (int): X-coord in pixels.
    y (int): Y-coord in pixels.
    depth (float): depth in meters.
    
    Returns:
    tuple: Cam coords (X, Y, Z).
    """
    fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
    cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
    
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

#=============== PROCESSING LANDMARKS FROM POSE ===============
def process_landmarks(results, color_image, depth_image, depth_scale, camera_matrix=CAMERA_MATRIX):
    """
    Processing the img to get landmarks and its coordinates.
    """
    if not results.pose_landmarks:
        return {}
    
    pixel_coords = {}
    landmarks = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        x, y = int(landmark.x * IMAGE_WIDTH), int(landmark.y * IMAGE_HEIGHT) # pixels
        # check if coords are in img borders
        if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
            median_depth = calculate_median_depth(x, y, depth_image) * depth_scale # depth data in meters
            pixel_coords[idx] = (x, y, median_depth)
            cam_coords = pixels_to_camera_coordinates(x, y, median_depth)
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

#======================== MAIN ========================
def main():
    """
    Main func for rendering frames and save data.
    """
    pipeline, pose, depth_scale = setup_camera_and_pose()
    
    frame_data = {
        "camera_landmarks": [],
        "global_landmarks": []
    }

    start_time = time()
    last_save_time = 0
    frame_count = 0

    try: # Process video frames
        while True:
            time_of_frame, color_image, depth_image, results, (rvec, tvec) = collect_frame(pipeline, pose, start_time)
            if color_image is None and depth_image is None and rvec is None and tvec is None:
                continue

            # Calculate 3D coordinates with depth
            landmarks = process_landmarks(results, color_image, depth_image, depth_scale)
            if PRINT_LANDMARKS_TO_CONSOLE and landmarks:
                print(f"Extracted Landmarks: {landmarks}")
            
            global_landmarks = convert_landmarks_to_global(landmarks, rvec, tvec)
            if PRINT_LANDMARKS_TO_CONSOLE and global_landmarks:
                print(f"Global Landmarks: {global_landmarks}")
            
            # data saving at defined intervals
            current_time = time_of_frame
            if (current_time - last_save_time) >= PARAMETER_TIMESTEP:
                # saving frame data
                frame_data["camera_landmarks"].append({
                    "timestamp": time_of_frame,
                    "landmarks": landmarks
                })
                frame_data["global_landmarks"].append({
                    "timestamp": time_of_frame,
                    "landmarks": global_landmarks
                })
                
                frame_count += 1
                last_save_time = current_time
                print("Frame", frame_count, "with time", last_save_time)
                
                # Check if we've reached the desired ACTION_LENGTH
                if frame_count >= ACTION_LENGTH:
                    with open(FILE_NAME_LANDMARKS, 'w') as cam_file:
                        json.dump({"header": METADATA, "data": frame_data["camera_landmarks"]}, cam_file, indent=4)
                    with open(FILE_NAME_GLOBAL, 'w') as global_file:
                        json.dump({"header": METADATA, "data": frame_data["global_landmarks"]}, global_file, indent=4)
                    print(f"Saved {ACTION_LENGTH} actions")
                    break
                
            # Display the image
            cv2.imshow('Real World Landmark Coordinates', color_image)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()