import subprocess
import json
import os

import cv2
import numpy as np

def get_video_info(video_path, use_count_frames: bool = False):
    """
    Get video information using ffprobe.
    
    Args:
        video_path: Path to video file
        use_count_frames: If True, use -count_frames for accurate frame count (slower).
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
    ]
    if use_count_frames:
        cmd.extend(['-count_frames', '-select_streams', 'v:0'])
    cmd.append(video_path)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    
    # Find video stream
    video_stream = None
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            video_stream = stream
            break
    
    if not video_stream:
        raise ValueError("No video stream found")
    
    fps_parts = video_stream['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1])
    duration = float(info['format']['duration'])
    # Use nb_read_frames (from -count_frames) or nb_frames if available
    nb_read = video_stream.get('nb_read_frames')
    nb_frames = video_stream.get('nb_frames')
    if nb_read is not None and nb_read != 'N/A':
        num_frames = int(nb_read)
    elif nb_frames is not None and nb_frames != 'N/A':
        num_frames = int(nb_frames)
    else:
        num_frames = int(round(duration * fps))
    
    return {
        'fps': fps,
        'duration': duration,
        'num_frames': num_frames,
        'width': int(video_stream['width']),
        'height': int(video_stream['height'])
    }

def parse_time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds."""
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Time must be in HH:MM:SS format, got: {time_str}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    return hours * 3600 + minutes * 60 + seconds


def cut_video_segment(video_path: str, start_time: str, end_time: str, output_path: str, 
                      copy_codec: bool = True, verbose: bool = True, log_original: bool = True):
    """
    Cut a video segment from start_time to end_time using ffmpeg.
    
    Args:
        video_path: Path to input video
        start_time: Start time in HH:MM:SS or HH:MM:SS.mmm format
        end_time: End time in HH:MM:SS or HH:MM:SS.mmm format
        output_path: Path to output video
        copy_codec: If True, copy streams without re-encoding (fast, lossless).
                   If False, re-encode (slower, but more compatible)
        verbose: Print ffmpeg output
        log_original: If True, log original video frame count (set False when called in batch to avoid repetition)
    
    Returns:
        True if successful
    
    Example:
        cut_video_segment("input.mp4", "00:01:30", "00:02:45", "output.mp4")
    """
    import os
    
    # Validate input file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Parse times to seconds
    try:
        start_seconds = parse_time_to_seconds(start_time)
        end_seconds = parse_time_to_seconds(end_time)
    except ValueError as e:
        raise ValueError(f"Invalid time format: {e}")
    
    # Validate time range
    if start_seconds < 0:
        raise ValueError(f"Start time cannot be negative: {start_time}")
    if end_seconds <= start_seconds:
        raise ValueError(f"End time ({end_time}) must be after start time ({start_time})")
    
    # Get video info to validate against duration
    try:
        info = get_video_info(video_path)
        duration = info['duration']
        if verbose and log_original:
            print(f"Original video: {info['num_frames']} frames")
        
        if start_seconds >= duration:
            raise ValueError(f"Start time {start_time} ({start_seconds}s) is beyond video duration ({duration}s)")
        if end_seconds > duration:
            print(f"Warning: End time {end_time} ({end_seconds}s) is beyond video duration ({duration}s). "
                  f"Will cut until end of video.")
            end_seconds = duration
    except Exception as e:
        print(f"Warning: Could not get video info: {e}. Proceeding without validation.")
    
    # Calculate duration
    segment_duration = end_seconds - start_seconds
    
    # Build ffmpeg command
    # Using -ss before -i for faster seeking (input seeking)
    # -t for duration instead of -to for better accuracy
    cmd = ['ffmpeg', '-y']  # -y to overwrite output file
    
    # Seek to start time (input seeking - faster)
    cmd.extend(['-ss', str(start_seconds)])
    
    # Input file
    cmd.extend(['-i', video_path])
    
    # Duration of segment
    cmd.extend(['-t', str(segment_duration)])
    
    # Codec options
    if copy_codec:
        # Copy streams without re-encoding (fast, lossless)
        cmd.extend(['-c', 'copy'])
    else:
        # Re-encode (slower but more compatible)
        cmd.extend(['-c:v', 'libx264', '-c:a', 'aac'])
    
    # Avoid non-monotonous DTS issues when using -c copy
    if copy_codec:
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
    
    # Output file
    cmd.append(output_path)
    
    # Print command (only in verbose mode)
    if verbose:
        print(f"Cutting: {video_path} -> {output_path} ({segment_duration:.1f}s)")
    
    # Run ffmpeg (always capture output to avoid ffmpeg version/config/stream spam)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if verbose:
            try:
                out_info = get_video_info(output_path)
                print(f"  Cut video: {out_info['num_frames']} frames")
            except Exception as e:
                print(f"  Cut video: (could not get frame count: {e})")
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with return code {e.returncode}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg")

def cut_video_frames(
    video_path: str,
    start_frame: int,
    num_frames: int,
    output_path: str,
    verbose: bool = True,
) -> bool:
    """
    Extract exactly num_frames starting from start_frame using ffmpeg select filter.
    Re-encodes for frame-accurate extraction (stream copy cannot guarantee exact frames).
    
    Args:
        video_path: Path to input video
        start_frame: Zero-based start frame index
        num_frames: Number of frames to extract
        output_path: Path to output video
        verbose: Print progress
    
    Returns:
        True if successful
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if start_frame < 0 or num_frames <= 0:
        raise ValueError("start_frame must be >= 0 and num_frames must be > 0")
    
    end_frame = start_frame + num_frames - 1  # inclusive in select filter
    # Commas inside between() must be escaped so ffmpeg doesn't treat them as filter separators
    vf = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS"
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', vf,
        '-vsync', 'cfr',
        '-c:v', 'libx264',
        '-an',  # drop audio — select filter only applies to video
        output_path
    ]
    if verbose:
        print(f"Cutting frames {start_frame}-{end_frame} ({num_frames} frames) -> {output_path}")
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with return code {e.returncode}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    if verbose:
        try:
            out_info = get_video_info(output_path)
            print(f"  Cut video: {out_info['num_frames']} frames")
        except Exception as e:
            print(f"  Cut video: (could not verify frame count: {e})")
    return True


def extract_first_frame(video_path: str, output_path: str, verbose: bool = True) -> np.ndarray:
    """
    Extract the first frame from a video and save it as an image.
    Uses OpenCV (same approach as dataset/video_extract.py).
    
    Args:
        video_path: Path to input video
        output_path: Path to output image (e.g. .jpg or .png)
        verbose: Print progress
    
    Returns:
        The frame as a numpy array (BGR), or None if read failed
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    if verbose:
        print(f"  Extracted first frame: {output_path}")
    return frame


def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def split_video_into_segments(
    video_path: str,
    output_dir: str,
    frames_per_segment: int = 1200,
    verbose: bool = True,
) -> list:
    """
    Split a video into segments by exact frame count. Each frame appears exactly once.
    Uses ffprobe -count_frames for accurate frame count and ffmpeg select filter for
    frame-accurate extraction. Also extracts the first frame of each segment as a .jpg
    (e.g. video_name_seg001_frame0.jpg).
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save segments (and first-frame images)
        frames_per_segment: Frames per segment (default: 1200). Last segment gets remainder.
        verbose: Print progress
    
    Returns:
        List of output video file paths
    """
    if verbose:
        print(f"\nSplitting: {video_path}")
    info = get_video_info(video_path, use_count_frames=True)
    total_frames = info['num_frames']
    if verbose:
        print(f"Original video: {total_frames} frames (count_frames)")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    os.makedirs(output_dir, exist_ok=True)
    
    num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment
    
    output_files = []
    for i in range(num_segments):
        start_frame = i * frames_per_segment
        if i == num_segments - 1:
            num_frames = total_frames - start_frame
        else:
            num_frames = frames_per_segment
        if num_frames <= 0:
            break
        output_filename = f"{video_name}_seg{i+1:03d}{video_ext}"
        output_path = os.path.join(output_dir, output_filename)
        if verbose:
            print(f"[INFO] Segment {i+1}/{num_segments}: creating {output_filename} (frames {start_frame}-{start_frame + num_frames - 1}, {num_frames} frames)")
        try:
            cut_video_frames(
                video_path=video_path,
                start_frame=start_frame,
                num_frames=num_frames,
                output_path=output_path,
                verbose=verbose,
            )
            output_files.append(output_path)
            if verbose:
                print(f"[INFO]   Segment created: {output_path}")
            # Extract first frame of this segment
            frame_filename = f"{video_name}_seg{i+1:03d}_frame0.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            if verbose:
                print(f"[INFO]   Extracting first frame for segment {i+1}: {frame_filename}")
            try:
                extract_first_frame(
                    video_path=output_path,
                    output_path=frame_path,
                    verbose=verbose,
                )
                if verbose:
                    print(f"[INFO]   First frame saved: {frame_path}")
            except Exception as e:
                print(f"[WARN]   Error extracting first frame for segment {i+1}: {e}")
        except Exception as e:
            print(f"[ERROR]   Error cutting segment {i+1}: {e}")
    
    if verbose:
        print(f"[INFO] Created {len(output_files)} segments")
    
    return output_files


def process_folder_videos(
    input_folder: str,
    output_base_dir: str = "./experiments/video_segs",
    frames_per_segment: int = 1200,
    video_extensions: list = None,
    verbose: bool = True,
) -> dict:
    """
    Process all videos in a folder, splitting each into fixed-frame segments.
    Each frame appears exactly once. Uses ffprobe -count_frames and re-encodes.
    
    Args:
        input_folder: Path to folder containing videos
        output_base_dir: Base directory for output (default: ./experiments/video_segs)
        frames_per_segment: Frames per segment (default: 1200)
        video_extensions: List of video extensions to process (default: common formats)
        verbose: Print progress
    
    Returns:
        Dictionary mapping input video paths to lists of output segment paths
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.MP4', '.AVI', '.MOV']
    
    # Find all video files
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    video_files = []
    for f in sorted(os.listdir(input_folder)):
        if any(f.endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(input_folder, f))
    
    if not video_files:
        print(f"No video files found in: {input_folder}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Processing {len(video_files)} video(s) from: {input_folder}")
    print(f"Output directory: {output_base_dir}")
    print(f"Frames per segment: {frames_per_segment}")
    print(f"{'='*60}")
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    results = {}
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_name}")
        
        # Create subfolder for this video
        video_output_dir = os.path.join(output_base_dir, video_name)
        
        try:
            segments = split_video_into_segments(
                video_path=video_path,
                output_dir=video_output_dir,
                frames_per_segment=frames_per_segment,
                verbose=verbose,
            )
            results[video_path] = segments
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            results[video_path] = []
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_segments = sum(len(segs) for segs in results.values())
    print(f"Total videos processed: {len(results)}")
    print(f"Total segments created: {total_segments}")
    for video_path, segments in results.items():
        video_name = os.path.basename(video_path)
        print(f"  {video_name}: {len(segments)} segments")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cut video segments using ffmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cut from 1:30 to 2:45
  python video_cut.py -i input.mp4 -s 00:01:30 -e 00:02:45 -o output.mp4
  
  # Re-encode instead of copy (slower but more compatible)
  python video_cut.py -i input.mp4 -s 00:01:30 -e 00:02:45 -o output.mp4 --no-copy
  
  # With milliseconds
  python video_cut.py -i input.mp4 -s 00:01:30.500 -e 00:02:45.750 -o output.mp4

  # Process all videos in a folder (1200 frames per segment, max 6 segments)
  python video_cut.py --folder /path/to/videos
  
  # Process folder with custom frames per segment
  python video_cut.py --folder /path/to/videos --frames-per-segment 600 --output-dir ./my_segments
        """
    )
    
    # Subcommands via mutually exclusive groups
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-i', '--input', type=str, 
                           help='Input video file path (for single cut mode)')
    mode_group.add_argument('--folder', type=str,
                           help='Input folder containing videos (for batch segmentation mode)')
    
    # Single cut mode arguments
    parser.add_argument('-s', '--start', type=str,
                        help='Start time (HH:MM:SS or HH:MM:SS.mmm) - required for single cut')
    parser.add_argument('-e', '--end', type=str,
                        help='End time (HH:MM:SS or HH:MM:SS.mmm) - required for single cut')
    parser.add_argument('-o', '--output', type=str,
                        help='Output video file path - required for single cut')
    
    # Folder mode arguments
    parser.add_argument('--frames-per-segment', type=int, default=1200,
                        help='Frames per segment; last segment gets remainder (default: 1200)')
    parser.add_argument('--output-dir', type=str, default='./experiments/video_segs',
                        help='Output directory for segments (default: ./experiments/video_segs)')
    
    # Common arguments
    parser.add_argument('--no-copy', action='store_true',
                        help='Re-encode instead of copying streams (slower)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')
    
    args = parser.parse_args()
    
    try:
        if args.folder:
            # Folder mode: process all videos and split into segments
            process_folder_videos(
                input_folder=args.folder,
                output_base_dir=args.output_dir,
                frames_per_segment=args.frames_per_segment,
                verbose=not args.quiet,
            )
        else:
            # Single cut mode
            if not args.start or not args.end or not args.output:
                parser.error("Single cut mode requires -s/--start, -e/--end, and -o/--output")
            
            cut_video_segment(
                video_path=args.input,
                start_time=args.start,
                end_time=args.end,
                output_path=args.output,
                copy_codec=not args.no_copy,
                verbose=not args.quiet
            )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)