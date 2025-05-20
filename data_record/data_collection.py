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

    # get color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    camera_matrix = np.array([
        [intr.fx,    0,    intr.ppx],
        [   0,    intr.fy, intr.ppy],
        [   0,       0,        1   ]
    ])
    distortion_coeffs = np.array(intr.coeffs)  # 5-element list

    # Setup MediaPipe Pose
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5, 
        model_complexity=1
    )
    return pipeline, pose, depth_scale, camera_matrix, distortion_coeffs

#=============== FRAME COLLECTION AND PROCESSING ===============
def collect_frame(pipeline, pose, start_time, camera_matrix, distortion_coeffs):
    """
    Collect a frame from the camera and process it with MediaPipe Pose.
    """
    # Capture frames from RealSense
    frames = pipeline.wait_for_frames()
    time_of_frame = frames.get_timestamp() / 1000.0 - start_time
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    
    # Convert images to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # ArUco markers detection
    rvec, tvec, _ = detect_aruco_markers(color_image, camera_matrix, distortion_coeffs)

    results = pose.process(rgb_image)
    result = (time_of_frame, color_image, depth_image, results, (rvec, tvec))
    return result

#=============== ARUCO MARKER DETECTION ===============
def detect_aruco_markers(image, camera_matrix, distortion_coeffs):
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

    for corner in corners:
        _, rvec, tvec = cv2.solvePnP(MARKER_POINTS, corner, camera_matrix, distortion_coeffs)
        if VISUAL_LANDMARKS:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(image, camera_matrix, distortion_coeffs, rvec, tvec, 0.1)
        if PRINT_LANDMARKS_TO_CONSOLE:
            print(f"ArUco ID={ids[0][0]}, Position={tvec.flatten()}, Rotation={rvec.flatten()}")
    return rvec, tvec, ids