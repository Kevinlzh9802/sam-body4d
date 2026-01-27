import cv2
import numpy as np
import os
from data_utils import read_camera_intrinsics_new

def adjust_K(K, scale):
    K_resized = np.array([[K[0,0]*scale, 0,           K[0,2]*scale],
             [0,           K[1,1]*scale, K[1,2]*scale],
             [0,           0,         1]])
    return K_resized

def undistort_kps(kps: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort keypoints using camera parameters.
    """
    undistorted_kps = cv2.undistortPoints(kps, K, dist_coeffs)
    return undistorted_kps

def undistort_img(img: np.ndarray, K: np.ndarray, dist: np.ndarray, scale: float = 1.0, use_fisheye: bool = False) -> np.ndarray:
    """
    Undistort a single image using camera intrinsics.
    
    Args:
        img: Input image as numpy array
        intrinsic_path: Path to camera intrinsics JSON file
        scale: Scale factor applied to the image (default: 1.0 for original size)
               If the image is rescaled from original calibration size,
               adjust K accordingly (e.g., 0.5 for half size)
        use_fisheye: Whether to use fisheye distortion model (default: False)
    
    Returns:
        Undistorted image as numpy array
    """
    
    # Adjust intrinsics for rescaled images
    # When image is scaled, focal length and principal point must be scaled too
    if scale != 1.0:
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale  # fx
        K_scaled[1, 1] *= scale  # fy
        K_scaled[0, 2] *= scale  # cx
        K_scaled[1, 2] *= scale  # cy
    else:
        K_scaled = K
    
    # Distortion coefficients remain the same regardless of scale
    dim = (1920, 1080)
    R = np.eye(3)
    if use_fisheye:
        # Fisheye model requires cv2.fisheye module and specific coefficient shape
        dist_fish = np.reshape(dist[:4], (4, 1))
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_scaled, dist_fish, dim, R, balance=1.0, new_size=dim, fov_scale=scale
        )
        undistorted_img = cv2.fisheye.undistortImage(img, K_scaled, dist_fish, Knew=Knew)
        
    else:
        # Standard model uses cv2.undistort with newCameraMatrix
        undistorted_img = cv2.undistort(img, K_scaled, dist, newCameraMatrix=K_scaled)
    
    return undistorted_img


def undistort_and_save_video(
    video_path: str,
    intrinsic_path: str,
    output_path: str,
    scale: float = 1.0,
    codec: str = 'mp4v',
    use_fisheye: bool = False,
) -> bool:
    """
    Undistort a video and save it to a new file.
    
    Args:
        video_path: Path to input video file
        intrinsic_path: Path to camera intrinsics JSON file
        output_path: Path where the undistorted video will be saved
        scale: Scale factor for adjusting intrinsics if video is rescaled (default: 1.0)
        codec: FourCC codec code (default: 'mp4v')
        use_fisheye: Whether to use fisheye distortion model (default: False)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    print(f"Input video: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    K, dist = read_camera_intrinsics_new(intrinsic_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        undistorted_frame = undistort_img(frame, K, dist, scale, use_fisheye)
        out.write(undistorted_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    return True

def undistort_video_folder(
    input_folder: str,
    output_folder: str,
    intrinsic_path: str,
    scale: float = 1.0,
    use_fisheye: bool = False,
) -> bool:
    """
    Undistort a video folder and save it to a new folder.
    """
    for video_path in os.listdir(input_folder):
        undistort_and_save_video(os.path.join(input_folder, video_path), intrinsic_path, os.path.join(output_folder, video_path), scale=scale, use_fisheye=use_fisheye)
    return True

if __name__ == "__main__":
    # Example usage of undistort_and_save_video
    # video_path = ".\\experiments\\vid2-seg8-scaled-denoised.mp4"
    # intrinsic_path = "D:\\exp\\intrinsics\\parameters-camera-04.json"
    # output_path = ".\\experiments\\vid2-seg8-scaled-denoised-undistorted.mp4"
    input_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/segments_test"
    output_folder = "/home/zonghuan/tudelft/projects/datasets/modification/conflab/segments_undistorted"
    intrinsic_path = "./experiments/intrinsics/parameters-camera-04.json"
    scale = 1.0
    use_fisheye = True
    # Undistort and save video
    # undistort_and_save_video(video_path, intrinsic_path, output_path, scale=0.5, use_fisheye=True)
    undistort_video_folder(input_folder, output_folder, intrinsic_path, scale=scale, use_fisheye=use_fisheye)
    # Alternative: use generator function for frame-by-frame processing
    # for frame in undistort_video(video_path, intrinsic_path):
    #     # Process each undistorted frame
    #     pass