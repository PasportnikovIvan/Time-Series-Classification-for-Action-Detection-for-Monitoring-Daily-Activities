#data_record/main.py

from config import *
from time import time
import cv2
from data_collection import setup_camera_and_pose, collect_frame
from data_processing import process_landmarks, check_landmark_consistency
from data_storage import save_data
from utils import convert_landmarks_to_global


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
    prev_landmarks = None

    try: # Process video frames
        while True:
            time_of_frame, color_image, depth_image, results, (rvec, tvec) = collect_frame(pipeline, pose, start_time)
            if color_image is None:
                continue

            # Calculate 3D coordinates with depth
            landmarks = process_landmarks(results, color_image, depth_image, depth_scale)
            if PRINT_LANDMARKS_TO_CONSOLE and landmarks:
                print(f"Extracted Landmarks: {landmarks}")

            if not check_landmark_consistency(prev_landmarks, landmarks):
                print("Skipping frame due to misdetection")
                continue
            prev_landmarks = landmarks.copy()

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
                    save_data(
                        {"header": METADATA, "data": frame_data["camera_landmarks"]},
                        FILE_NAME_LANDMARKS
                    )
                    save_data(
                        {"header": METADATA, "data": frame_data["global_landmarks"]},
                        FILE_NAME_GLOBAL
                    )
                    print(f"Saved {ACTION_LENGTH} actions")
                    break

            # Display the image
            cv2.imshow('Real World Landmark Coordinates', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()