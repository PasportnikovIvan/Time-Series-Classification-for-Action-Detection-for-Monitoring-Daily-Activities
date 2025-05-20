#data_record/data_collection.py
# # Module for data collection and preprocessing using Mediapipe and pyrealsense

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import sounddevice as sd
import queue
from config import *

#=============== CAMERA, AUDIO, AND MEDIAPIPE SETUP ===============
def setup_camera_and_pose():
    """
    Setup RealSense camera, MediaPipe Pose, and audio input stream.
    Returns:
        pipeline, pose, depth_scale,
        camera_matrix, distortion_coeffs, audio_queue
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

    # Camera intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    camera_matrix = np.array([
        [intr.fx,    0,    intr.ppx],
        [   0,    intr.fy, intr.ppy],
        [   0,       0,        1   ]
    ], dtype=float)
    distortion_coeffs = np.array(intr.coeffs, dtype=float)  # 5-element list

    # Setup MediaPipe Pose
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5, 
        model_complexity=1
    )

    # Audio capture setup
    audio_queue = queue.Queue(maxsize=10)

    def _audio_callback(indata, frames, time_info, status):
        # Put audio buffer into queue
        audio_queue.put(indata.copy())

    sd.default.samplerate = AUDIO_RATE
    sd.default.channels = AUDIO_CHANNELS
    audio_stream = sd.InputStream(callback=_audio_callback)
    audio_stream.start()

    return pipeline, pose, depth_scale, camera_matrix, distortion_coeffs, audio_queue

#=============== FRAME COLLECTION AND PROCESSING ===============
def collect_frame(pipeline, pose, start_time, camera_matrix, distortion_coeffs, audio_queue):
    """
    Collect a frame from the camera, MediaPipe Pose, ArUco, and get audio amplitude.
    Returns tuple:
       (timestamp, color_image, depth_image,
        mp_results, markers_poses, audio_amp)
    """
    # -- Capture frames from RealSense --
    frames = pipeline.wait_for_frames()
    time_of_frame = frames.get_timestamp() / 1000.0 - start_time
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    
    # -- Convert to OpenCV arrays --
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # -- Detect ArUco markers --
    markers_poses = detect_aruco_markers(color_image, camera_matrix, distortion_coeffs)

    # -- MediaPipe Pose --
    mp_results = pose.process(rgb_image)

    # -- Audio amplitude (RMS) --
    try:
        audio_buffer = audio_queue.get()
        # Mix down to mono if stereo
        mono = audio_buffer.mean(axis=1) if audio_buffer.ndim > 1 else audio_buffer
        audio_amp = float(np.sqrt(np.mean(mono**2)))
    except queue.Empty:
        audio_amp = 0.0
        print("[WARN] Audio queue is empty")

    return (time_of_frame, color_image, depth_image, mp_results, markers_poses, audio_amp)

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
        return None
    if VISUAL_LANDMARKS:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    poses = {}
    for corner, mid in zip(corners, ids.flatten()):
        if mid in (GLOBAL_MARKER_ID, OBJECT_MARKER_ID):
            # corner shape (4,1,2) -> reshape to (4,2); solvePnP needs 2D points
            image_points = corner.reshape(-1, 2)
            _, rvec, tvec = cv2.solvePnP(MARKER_POINTS, image_points, camera_matrix, distortion_coeffs)
            poses[int(mid)] = (rvec, tvec)

            if VISUAL_LANDMARKS:
                cv2.drawFrameAxes(image, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE / 2)
            if PRINT_LANDMARKS_TO_CONSOLE:
                print(f"ArUco ID={mid}, Position={tvec.flatten()}, Rotation={rvec.flatten()}")
    return poses