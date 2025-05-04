#data_record/config.py
# # Configuration file for data collection and processing

import cv2.aruco as aruco
import numpy as np

#------------------------ Window ------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30
VISUAL_LANDMARKS = True
PRINT_LANDMARKS_TO_CONSOLE = False

#------------------------ Aruco ------------------------
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = aruco.DetectorParameters()
MARKER_SIZE = 0.154  # Marker size in METERS
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
ACTION_TYPE = "sppb"
ACTION_SESSION = "04"
ACTION_SUBJECT = "ivan"
NOTES = "SPPB Prorocol repeated chair stand test. Subject was asked to stand up from a chair and sit down 5 times as fast as possible."

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
    "recording_date": "2025-03-01",
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
            [0.9475199679422059, -0.24697668000900955, 0.20299859576491916],
            [0.029801453915425796, -0.5639730453764839, -0.8252552801608126],
            [0.3183045455147138, 0.7879955098986935, -0.5270154577279207]
        ],
        "translation": [
            -1.892179991149924,
            0.2693138361963459,
            2.8324064062560885
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
POSITION_THRESHOLD = 2.5  # Max allowed movement in meters between frames