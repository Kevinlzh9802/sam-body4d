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

if __name__ == "__main__":
    video_path = "/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras/video/cam04/GH020010.MP4"
    time = "00:03:45"
    frame = extract_frame(video_path, time)
    cv2.imwrite("./experiments/extrinsics/calibration/cam04_345s.jpg", frame)