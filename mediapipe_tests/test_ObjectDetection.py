import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=5,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.99,
                                # Chair, Shoe, Cup, Camera
                                model_name='Chair') as objectron: 
        while True:
            # Wait for a new frame from the RealSense camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert RealSense color frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Process the color image with MediaPipe
            color_image.flags.writeable = False
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = objectron.process(rgb_image)

            # Draw the box landmarks on the image if objects are detected
            color_image.flags.writeable = True
            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        color_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(color_image, detected_object.rotation,
                                         detected_object.translation)

            # Display the output image
            cv2.imshow('MediaPipe Objectron with RealSense', cv2.flip(color_image, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Clean up resources
    pipeline.stop()
    cv2.destroyAllWindows()