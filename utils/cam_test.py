import cv2
import os
from typing import Optional
import numpy as np
import json

def read_and_save_nth_frame(
    video_path: str,
    frame_number: int,
    output_path: str,
    verbose: bool = True
) -> bool:
    """
    Read and save the n-th frame of a video.
    
    Args:
        video_path (str): Path to the input video file
        frame_number (int): The frame number to extract (0-indexed)
        output_path (str): Path where the frame image will be saved
        verbose (bool): Whether to print status messages (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        if verbose:
            print(f"Error: Video file not found at {video_path}")
        return False
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        if verbose:
            print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate frame number
    if frame_number < 0:
        if verbose:
            print(f"Error: Frame number must be non-negative, got {frame_number}")
        cap.release()
        return False
    
    if frame_number >= total_frames:
        if verbose:
            print(f"Error: Frame number {frame_number} exceeds total frames {total_frames}")
        cap.release()
        return False
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        if verbose:
            print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    
    if not success:
        if verbose:
            print(f"Error: Could not save frame to {output_path}")
        cap.release()
        return False
    
    if verbose:
        print(f"Successfully saved frame {frame_number} to {output_path}")
        print(f"Frame shape: {frame.shape}")
    
    # Release the video capture object
    cap.release()
    
    return True


def read_camera_intrinsics_new(intrinsic_file: str):
    with open(intrinsic_file, "r") as f:
        intrinsic_data = json.load(f)
        params = intrinsic_data['Calibration']['cameras'][0]['model']['ptr_wrapper']['data']['parameters']

        f = params['f']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        ks = [params[f'k{i}']['val'] for i in range(1, 5)]
        dist_coeffs = np.array(ks)

    return K, dist_coeffs

def undistort_img(img_path: str, intrinsic_path: str, scale: float = 0.5) -> np.ndarray:
    K, dist = read_camera_intrinsics_new(intrinsic_path)
    img = cv2.imread(img_path)
    undistorted_img = cv2.undistort(img, K, dist)
    return undistorted_img

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python cam_test.py <video_path> <frame_number> <output_path>")
        print("Example: python cam_test.py video.mp4 100 frame_100.jpg")
    else:
        video_path = sys.argv[1]
        frame_number = int(sys.argv[2])
        output_path = sys.argv[3]
        
        read_and_save_nth_frame(video_path, frame_number, output_path)
        
    # img_path = "D:\\exp\\cam4_01.jpg"
    # for intrinsic_num in [2, 4, 6, 8]:
    #     intrinsic_path = f"D:\\exp\\intrinsics\\parameters-camera-0{intrinsic_num}.json"
    #     undistorted_img = undistort_img(img_path, intrinsic_path)
    #     cv2.imwrite(f"D:\\exp\\cam06_undistorted_by_cam{intrinsic_num}.jpg", undistorted_img)