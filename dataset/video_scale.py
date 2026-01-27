import cv2
import os

def video_scale(video_path: str, scale: float) -> bool:
    """
    Scale a video by a factor in height and width.
    The result is saved with a '_scaled' suffix.
    """
    cap = cv2.VideoCapture(video_path)

    # Get original properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Generate output path
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_scaled{ext}"
    
    # Use mp4v as a standard codec for MP4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        cap.release()
        return False

    print(f"Scaling video: {width}x{height} -> {new_width}x{new_height} @ {fps} fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame using linear interpolation
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        out.write(resized_frame)
    cap.release()
    out.release()
        
    print(f"Successfully scaled video to: {output_path}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scale a video by a factor")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video path')
    parser.add_argument('-s', '--scale', type=float, required=True, help='Scale factor (e.g., 0.5 for half size)')
    
    args = parser.parse_args()
    video_scale(args.input, args.scale)
