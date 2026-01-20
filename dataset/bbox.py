"""
Script to read keypoints, undistort them, and visualize on video.

This script provides modular functions to:
1. Read keypoints from a pickle file
2. Undistort keypoints using camera parameters
3. Visualize unprojected points on video frames
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Add parent directory to path to import utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.painter import point_painter
from undistortion import adjust_K


def undistort_keypoints_from_json(
    data: List[Dict[str, Any]],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    scale: float = 1.0,
    alpha: float = 0.0,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Undistort keypoints using camera parameters.
    
    Args:
        data: List of dictionaries containing keypoints for each frame
        K: Camera intrinsic matrix (3, 3)
        dist_coeffs: Distortion coefficients (typically 4, 5, or 8 values)
        scale: Scale factor for camera matrix (default: 1.0)
        alpha: Free scaling parameter (0=crop to valid pixels, 1=keep all)
        image_size: Optional (width, height) for optimal new camera matrix
        
    Returns:
        Tuple of (undistorted_keypoints, camera_info_dict)
        camera_info_dict contains: K, dist_coeffs, new_K
    """
    
    # Compute optimal new camera matrix if image size is provided
    new_K = None
    if image_size is not None:
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, image_size, alpha=alpha, newImgSize=image_size
        )
    else:
        new_K = K
    
    undistorted_data = []
    # Undistort keypoints
    for frame in data:
        kps = frame['kps']  # Shape: (N, 10, 3) where 3 = [x, y, person_id]
        N, num_kps, _ = kps.shape
        
        # Extract x, y coordinates and person IDs
        xy_coords = kps[:, :, :2]  # Shape: (N, 10, 2)
        person_ids = kps[:, :, 2:3]  # Shape: (N, 10, 1)
        
        # Flatten for cv2.undistortPoints: (N*10, 1, 2)
        xy_flat = xy_coords.reshape(-1, 1, 2).astype(np.float64)
        
        # Undistort points
        undistorted_xy = cv2.undistortPoints(
            xy_flat,
            K.astype(np.float64),
            dist_coeffs.astype(np.float64),
            P=new_K.astype(np.float64)
        )
        
        # Reshape back to (N, 10, 2) and concatenate with person IDs
        undistorted_xy = undistorted_xy.reshape(N, num_kps, 2).astype(np.float32)
        undistorted_kps = np.concatenate([undistorted_xy, person_ids], axis=2)  # Shape: (N, 10, 3)
        
        undistorted_data.append({
            'bboxes': frame['bboxes'],
            'kps': undistorted_kps,
        })
    
    # Return undistorted keypoints and camera info
    camera_info = {
        'K': K,
        'dist_coeffs': dist_coeffs,
        'new_K': new_K,
    }
    
    return undistorted_data, camera_info

def get_frame(video_path: str, frame_num: int) -> np.ndarray:
    """
    Get a frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    point_alpha: float = 0.9,
    point_radius: int = 8,
    contour_width: int = 3,
    draw_person_ids: bool = True,
) -> np.ndarray:
    """
    Draw keypoints on a single video frame with different colors per person.
    
    Args:
        frame: Video frame as numpy array (H, W, 3)
        keypoints: Keypoints array of shape (N, 10, 3) where:
                   N = number of people
                   10 = number of keypoints per person
                   3 = [x, y, person_id]
        point_alpha: Alpha blending for points (0-1)
        point_radius: Radius of each point in pixels
        contour_width: Width of contour around points
        draw_person_ids: Whether to draw person IDs as text
        
    Returns:
        Frame with keypoints drawn
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Handle empty keypoints
    if keypoints.size == 0:
        return frame
    
    # Iterate over each person
    num_people = keypoints.shape[0]
    
    for person_idx in range(num_people):
        person_kps = keypoints[person_idx]  # Shape: (10, 3)
        
        # Get person_id from the first keypoint (all should have the same ID)
        person_id = int(person_kps[0, 2])
        
        # Use person_id to determine color (different color per person)
        # Offset by 2 to skip black and white in the colormap
        point_color = (person_id + 2) % 80
        contour_color = (person_id + 3) % 80
        
        # Collect valid keypoints for this person
        valid_kps = []
        valid_kp_indices = []
        
        for kp_idx, kp in enumerate(person_kps):
            x, y = kp[0], kp[1]
            
            # Check if point is within frame bounds
            if 0 <= x < w and 0 <= y < h:
                valid_kps.append([int(x), int(y)])
                valid_kp_indices.append(kp_idx)
        
        if not valid_kps:
            continue
        
        # Draw points using the painter utility
        valid_kps_arr = np.array(valid_kps)
        frame = point_painter(
            frame,
            valid_kps_arr,
            point_color=point_color,
            point_alpha=point_alpha,
            point_radius=point_radius,
            contour_color=contour_color,
            contour_width=contour_width,
        )
        
        # Draw person ID near the first valid keypoint
        if draw_person_ids and valid_kps:
            x, y = valid_kps[0]
            cv2.putText(
                frame,
                f"P{person_id}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # Draw black outline for better visibility
            cv2.putText(
                frame,
                f"P{person_id}",
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    
    return frame


if __name__ == '__main__':
    # main()
    pkl_path = "./experiments/bboxes_kps_refined/428.pkl"
    intrinsic_json_path = "./experiments/intrinsics/intrinsic_2.json"
    video_path = "./experiments/cam04_cut_1.mp4"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    with open(intrinsic_json_path, 'r') as f:
        intrinsics = json.load(f)
    K = np.asarray(intrinsics['intrinsic'], dtype=np.float64)
    K = adjust_K(K, 0.5)
    dist_coeffs = np.asarray(intrinsics['distortion_coefficients'], dtype=np.float64)

    undistorted_data, _ = undistort_keypoints_from_json(data, K, dist_coeffs)
    frame = get_frame(video_path, 0)
    use_fisheye = False
    if use_fisheye:
        undistorted_frame = cv2.undistort(frame, K, dist_coeffs, Knew=K)
    else:
        undistorted_frame = cv2.undistort(frame, K, dist_coeffs)
    frame_kp = draw_keypoints_on_frame(undistorted_frame, undistorted_data[0]['kps'])
    plt.imshow(frame_kp)
    plt.show()
