import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs
import numpy as np

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

# getting pose with MediaPipe
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

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # making coords from standert values [0, 1] to pixel values
                x_px = int(landmark.x * IMAGE_WIDTH)
                y_px = int(landmark.y * IMAGE_HEIGHT)

                # check if coords are in im borders
                if 0 <= x_px < IMAGE_WIDTH and 0 <= y_px < IMAGE_HEIGHT:
                    # getting the depth value
                    depth = depth_frame.get_distance(x_px, y_px)
                    print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, depth={depth}")
                else:
                    print(f"Landmark {idx} out of bounds with coordinates x={x_px}, y={y_px}")

        cv2.imshow("MediaPipe Pose", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pipeline.stop()
cv2.destroyAllWindows()