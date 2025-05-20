#data_record/config.py
# # Configuration file for data collection and processing

import cv2.aruco as aruco
import numpy as np

#------------------------ METADATA ------------------------
# Define constants for action
ACTION_NAME = "standing"
ACTION_SESSION = "11"
ACTION_SUBJECT = "ivan"
SUBJECT_AGE = "21"
SUBJECT_GENDER = "male"
SUBJECT_HEALTH_STATUS = "healthy"
ACTION_LOCATION = "bubenec_dorm"
ACTION_LIGHTING_CONDITIONS = "bright"
CAMERA_MODEL = "Intel RealSense D455"
RECORDING_DATE = "2025-05-01"
NOTES = "Standing"

#------------------------ Saving Action ------------------------
ACTION_LENGTH = 100 # actions
PARAMETER_TIMESTEP = 1.0 / 10.0  # seconds between saved samples (0.1 s)

GLOBAL_DIRECTORY = '../dataset' # Relative path to dataset from classification directory
FILE_NAME_GLOBAL = f'{GLOBAL_DIRECTORY}/{ACTION_NAME}/{ACTION_NAME}_{ACTION_SESSION}_globallandmarksdata_{ACTION_SUBJECT}.json'

#------------------------ Window ------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30
VISUAL_LANDMARKS = True
PRINT_LANDMARKS_TO_CONSOLE = False

#------------------------ AUDIO ------------------------
AUDIO_RATE = 44100
AUDIO_CHANNELS = 1

#------------------------ DEPTH FILTERING ------------------------
RADIUS_WINDOW = 5     # window size for depth patch
MEDIAN_PERCENT = 0.25 # fraction of closest depths

#------------------------ Aruco ------------------------
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = aruco.DetectorParameters()
GLOBAL_MARKER_ID = 300 # defines your world coordinate frame
OBJECT_MARKER_ID = 100 # attached to the object (e.g., bed or chair)
MARKER_SIZE = 0.1592  # Marker size in METERS
MARKER_POINTS = np.array(
    [
        [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
        [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
        [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    ],
    dtype=np.float32,
)

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