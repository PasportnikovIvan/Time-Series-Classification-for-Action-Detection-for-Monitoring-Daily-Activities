#data_record/main.py

from config import *
from time import time
import cv2
from data_collection import setup_camera_and_pose, collect_frame
from data_processing import process_landmarks
from data_storage import save_data
from utils import convert_landmarks_to_global


def main():
    """
    Main func for rendering frames and save data.
    """
    # Setup camera and pose detection
    pipeline, pose, depth_scale = setup_camera_and_pose()

    # Initialize variables
    raw_landmarks = []
    start_time = time()
    last_save_time = 0
    frame_count = 0
    hard_stop = False

    try:
        # Process video frames
        while frame_count < ACTION_LENGTH:
            time_of_frame, color_image, depth_image, results, (rvec, tvec) = collect_frame(pipeline, pose, start_time)
            if color_image is None and depth_image is None and rvec is None and tvec is None:
                continue
            
            # Calculate 3D coordinates with depth
            landmarks = process_landmarks(results, color_image, depth_image, depth_scale)
            if PRINT_LANDMARKS_TO_CONSOLE and landmarks:
                print(f"Extracted Landmarks: {landmarks}")

            current_time = time_of_frame
            if (current_time - last_save_time) >= PARAMETER_TIMESTEP:
                # Saving frame data
                raw_landmarks.append((time_of_frame, landmarks, rvec, tvec))
                frame_count += 1
                last_save_time = current_time
                print(f"Collected frame {frame_count}/{ACTION_LENGTH} at time {current_time:.2f}s")

            # Display the image
            cv2.imshow('Real World Landmark Coordinates', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                hard_stop = True
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if hard_stop:
        print("Hard stop detected. Data will not be saved.")
        return

    frame_data = {
        "camera_landmarks": [],
        "global_landmarks": []
    }

    for i, (time_of_frame, landmarks, rvec, tvec) in enumerate(raw_landmarks):
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        global_landmarks = convert_landmarks_to_global(landmarks, rvec, tvec)
        if PRINT_LANDMARKS_TO_CONSOLE and global_landmarks:
            print(f"Global Landmarks: {global_landmarks}")

        # saving frame data
        frame_data["camera_landmarks"].append({
            "timestamp": time_of_frame,
            "landmarks": landmarks
        })
        frame_data["global_landmarks"].append({
            "timestamp": time_of_frame,
            "landmarks": global_landmarks
        })
        print(f"Processed frame {i + 1}/{ACTION_LENGTH} with timestamp {time_of_frame:.2f}s")

    # Save the data to JSON files
    save_data({"header": METADATA, "data": frame_data["camera_landmarks"]}, FILE_NAME_LANDMARKS)
    save_data({"header": METADATA, "data": frame_data["global_landmarks"]}, FILE_NAME_GLOBAL)
    print(f"Saved {ACTION_LENGTH} actions")

if __name__ == "__main__":
    main()