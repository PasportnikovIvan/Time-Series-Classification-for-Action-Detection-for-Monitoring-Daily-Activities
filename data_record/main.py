#data_record/main.py
# # Main module for data collection and processing

from config import *
from time import time
import cv2
from data_collection import setup_camera_and_pose, collect_frame
from data_processing import process_landmarks
from utils import build_metadata, rodrigues_to_matrix, convert_point_to_global, convert_landmarks_to_global
from data_storage import save_data
from detection_and_repair import detect_misdetections, repair_misdetections
from trim_landmarks import trim_json_data


def main():
    """
    Main func for rendering frames and save data.
    """
    # Setup camera; pose and audio detection
    pipeline, pose, depth_scale, camera_matrix, distortion_coeffs, audio_queue, audio_stream = setup_camera_and_pose()
    print(">>> audio_queue:", audio_queue, "alive streams:", audio_stream)

    # Build metadata once
    metadata = build_metadata(camera_matrix.tolist(), distortion_coeffs.tolist(), depth_scale)
    
    # Initialize variables
    raw_buffer = []
    start_time = time()
    last_save_time = 0
    frame_count = 0
    hard_stop = False

    try:
        # Process video frames
        while frame_count < ACTION_LENGTH:
            # Capture and process one frame
            result = collect_frame(
                pipeline, pose, start_time,
                camera_matrix, distortion_coeffs, audio_queue
            )
            if result is None:
                print("[WARN] collect_frame returned None")
                continue

            (time_of_frame, color_image, depth_image,
             mp_results, markers_poses, audio_amp) = result
            if mp_results is None or markers_poses is None:
                print("[WARN] No pose or markers detected")
                continue

            # global-marker pose
            rvec_gl, tvec_gl = markers_poses.get(GLOBAL_MARKER_ID, (None, None))
            # object-marker pose
            rvec_obj, tvec_obj = markers_poses.get(OBJECT_MARKER_ID, (None, None))
            
            # Calculate 3D coordinates with depth
            landmarks_cam = process_landmarks(
                mp_results, color_image, depth_image,
                depth_scale, camera_matrix
            )
            if PRINT_LANDMARKS_TO_CONSOLE and landmarks_cam:
                print(f"Extracted Landmarks: {landmarks_cam}")

            # Throttle to PARAMETER_TIMESTEP
            if (time_of_frame - last_save_time) >= PARAMETER_TIMESTEP:
                print(f">>> Audio amplitude: {audio_amp:.4f}")
                # Saving frame data
                raw_buffer.append((
                    time_of_frame,
                    landmarks_cam,
                    rvec_gl, tvec_gl,
                    rvec_obj, tvec_obj,
                    audio_amp
                ))
                frame_count += 1
                last_save_time = time_of_frame
                print(f"Collected frame {frame_count}/{ACTION_LENGTH} at time {time_of_frame:.2f}s")

            # Display the image
            cv2.imshow('Landmarks', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                hard_stop = True
                break

    except Exception as e:
        import traceback
        print("[ERROR] Exception in frame loop:", e)
        traceback.print_exc()
        return

    finally:
        audio_stream.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

    if hard_stop:
        print("Hard stop detected. Data will not be saved.")
        return

    data_entries = []

    for i, (
        time_of_frame, landmarks_cam, 
        rvec_gl, tvec_gl, 
        rvec_obj, tvec_obj,
        sound_amp
    ) in enumerate(raw_buffer):
        # Compute rotation matrix list or None
        rmat_list = (
            rodrigues_to_matrix(rvec_gl.flatten().tolist()).tolist()
            if rvec_gl is not None else
            None
        )
        # Translation vector or None
        list_tvec = tvec_gl.flatten().tolist() if tvec_gl is not None else None

        # Object coords in global frame, if all poses available
        # !IMPORTANT! WE NOT USE ROTATION MATRIX, BECAUSE IT IS NOT NECESSARY FOR OBJECT POSITION
        if (rvec_gl is not None) and (tvec_gl is not None) and (tvec_obj is not None):
            obj_coords = convert_point_to_global(tvec_obj, rvec_gl, tvec_gl)
        else:
            obj_coords = None
    
        # Compute global landmarks
        global_landmarks = convert_landmarks_to_global(landmarks_cam, rvec_gl, tvec_gl)
        if PRINT_LANDMARKS_TO_CONSOLE and global_landmarks:
            print(f"Global Landmarks: {global_landmarks}")

        # saving frame data
        data_entries.append({
            "timestamp":       time_of_frame,
            "rotation_matrix": rmat_list,
            "translation_vec": list_tvec,
            "landmarks":       global_landmarks,
            "obj_coords":      obj_coords,
            "sound_amp":       sound_amp
        })
        print(f"Processed frame {i + 1}/{ACTION_LENGTH} with timestamp {time_of_frame:.2f}s")

    # --- Assemble final JSON and save ---
    final_output = {
        "metadata": metadata,
        "data": data_entries
    }
    save_data(final_output, FILE_NAME_RAW)
    print(f'Saved {len(data_entries)} raw frames to {FILE_NAME_RAW}')

    # --- Detect misdetections and repair ---
    misdetection_indices = detect_misdetections(final_output["data"])
    if misdetection_indices:
        repaired_data = repair_misdetections(final_output["data"], misdetection_indices)
        final_output["data"] = repaired_data
        print(f"Repaired {len(misdetection_indices)} misdetections in-memory")
    else:
        print("No misdetections found.")

    # --- Trimming for length constraint (80 frames) ---
    trimmed = trim_json_data(final_output["data"], START_INDEX, END_INDEX)
    final_output["data"] = trimmed
    print(f"Trimmed frames to indices {START_INDEX}-{END_INDEX}, total now {len(final_output['data'])}")

    # --- Save JSON to processed folder ---
    save_data(final_output, FILE_NAME_PROCESSED)
    print(f'Saved {len(final_output["data"])} raw frames to {FILE_NAME_PROCESSED}')

if __name__ == "__main__":
    main()