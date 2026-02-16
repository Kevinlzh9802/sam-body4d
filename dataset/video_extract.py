# extract frame at mm:ss with given time
import cv2
import os
import numpy as np

def extract_frame(video_path: str, time: str) -> np.ndarray:
    """
    Extract a frame from a video at a given time.
    """
    cap = cv2.VideoCapture(video_path)
    time_seconds = time.split(':')
    time_seconds = int(time_seconds[0]) * 60 + int(time_seconds[1])
    cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    ret, frame = cap.read()
    cap.release()
    return frame

def extract_folder(video_folder: str, output_folder: str):
    for subfolder_dir in os.listdir(video_folder):
        subfolder_path = os.path.join(video_folder, subfolder_dir)
        for video_file in os.listdir(subfolder_path):
            if video_file.endswith(".MP4") or video_file.endswith(".mp4"):
                video_path = os.path.join(subfolder_path, video_file)
                time = "00:00:00"
                frame = extract_frame(video_path, time)
                image_name = video_file.split(".")[0] + ".jpg"
                cv2.imwrite(os.path.join(output_folder, image_name), frame)
                print(f"Extracted frame from {video_file} at {time}")


if __name__ == "__main__":
    video_path = "/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras/video/cam04/GH020010_rot.MP4"
    time = "00:03:45"
    frame = extract_frame(video_path, time)
    cv2.imwrite("./experiments/extrinsics/calibration/cam04_345s.jpg", frame)
    # video_folder = "./experiments/video_segs_20s"
    # output_folder = "./experiments/video_segs_20s_frames"
    # extract_folder(video_folder, output_folder)