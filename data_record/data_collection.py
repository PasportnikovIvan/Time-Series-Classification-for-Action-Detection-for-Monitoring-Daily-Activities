#data_record/data_collection.py
# # Module for data collection and preprocessing using Mediapipe and pyrealsense

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from config import *

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
        model_complexity=1
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