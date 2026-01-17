import subprocess
import json
import os

def get_video_info(video_path):
    """Get video information using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        video_path
    ]
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
    
    return {
        'fps': fps,
        'duration': duration,
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
                      copy_codec: bool = True, verbose: bool = True):
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
    
    # Print command
    if verbose:
        print(f"Cutting video segment:")
        print(f"  Input: {video_path}")
        print(f"  Start: {start_time} ({start_seconds}s)")
        print(f"  End: {end_time} ({end_seconds}s)")
        print(f"  Duration: {segment_duration}s")
        print(f"  Output: {output_path}")
        print(f"  Command: {' '.join(cmd)}")
    
    # Run ffmpeg
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            text=True,
            check=True
        )
        
        if verbose:
            print(f"✓ Successfully created: {output_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed with return code {e.returncode}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg")

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
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Input video file path')
    parser.add_argument('-s', '--start', type=str, required=True,
                        help='Start time (HH:MM:SS or HH:MM:SS.mmm)')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time (HH:MM:SS or HH:MM:SS.mmm)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output video file path')
    parser.add_argument('--no-copy', action='store_true',
                        help='Re-encode instead of copying streams (slower)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')
    
    args = parser.parse_args()
    
    try:
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