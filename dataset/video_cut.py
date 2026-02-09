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

def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def split_video_into_segments(
    video_path: str,
    output_dir: str,
    segment_duration: float = 20.0,
    max_segments: int = 6,
    copy_codec: bool = True,
    verbose: bool = True,
) -> list:
    """
    Split a video into fixed-duration segments.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save segments
        segment_duration: Duration of each segment in seconds (default: 20)
        max_segments: Maximum number of segments. The last segment will contain
                      all remaining frames if this limit is reached (default: 6)
        copy_codec: If True, copy streams without re-encoding
        verbose: Print progress
    
    Returns:
        List of output file paths
    """
    # Get video info
    info = get_video_info(video_path)
    total_duration = info['duration']
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(video_path)[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of segments (capped at max_segments)
    # If we would exceed max_segments, the last segment gets all remaining frames
    num_segments = int(total_duration // segment_duration)
    remainder = total_duration % segment_duration
    if remainder > 0:
        num_segments += 1
    
    # Cap at max_segments
    num_segments = min(num_segments, max_segments)
    
    if verbose:
        print(f"\nSplitting video: {video_path}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Segment duration: {segment_duration}s (last segment may be longer)")
        print(f"  Number of segments: {num_segments} (max: {max_segments})")
        print(f"  Output directory: {output_dir}")
    
    output_files = []
    
    for i in range(num_segments):
        start_seconds = i * segment_duration
        
        # For the last segment, include all remaining frames
        if i == num_segments - 1:
            end_seconds = total_duration
        else:
            end_seconds = (i + 1) * segment_duration
        
        start_time = seconds_to_time_str(start_seconds)
        end_time = seconds_to_time_str(end_seconds)
        
        # Output filename: original_name_seg001.ext
        output_filename = f"{video_name}_seg{i+1:03d}{video_ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        if verbose and i == num_segments - 1:
            actual_duration = end_seconds - start_seconds
            print(f"  Last segment duration: {actual_duration:.2f}s")
        
        try:
            cut_video_segment(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                output_path=output_path,
                copy_codec=copy_codec,
                verbose=verbose,
            )
            output_files.append(output_path)
        except Exception as e:
            print(f"  Error cutting segment {i+1}: {e}")
    
    if verbose:
        print(f"  Created {len(output_files)} segments")
    
    return output_files


def process_folder_videos(
    input_folder: str,
    output_base_dir: str = "./experiments/video_segs",
    segment_duration: float = 20.0,
    max_segments: int = 6,
    video_extensions: list = None,
    copy_codec: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Process all videos in a folder, splitting each into fixed-duration segments.
    
    Args:
        input_folder: Path to folder containing videos
        output_base_dir: Base directory for output (default: ./experiments/video_segs)
        segment_duration: Duration of each segment in seconds (default: 20)
        max_segments: Maximum number of segments per video. The last segment will
                      contain all remaining frames if this limit is reached (default: 6)
        video_extensions: List of video extensions to process (default: common formats)
        copy_codec: If True, copy streams without re-encoding
        verbose: Print progress
    
    Returns:
        Dictionary mapping input video paths to lists of output segment paths
    
    Output structure:
        <output_base_dir>/
            <video1_name>/
                video1_name_seg001.mp4
                video1_name_seg002.mp4
                ...
            <video2_name>/
                video2_name_seg001.mp4
                ...
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
    print(f"Segment duration: {segment_duration}s (max {max_segments} segments)")
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
                segment_duration=segment_duration,
                max_segments=max_segments,
                copy_codec=copy_codec,
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

  # Process all videos in a folder (split into 20s segments, max 6 segments)
  python video_cut.py --folder /path/to/videos
  
  # Process folder with custom segment duration, max segments, and output directory
  python video_cut.py --folder /path/to/videos --segment-duration 30 --max-segments 10 --output-dir ./my_segments
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
    parser.add_argument('--segment-duration', type=float, default=20.0,
                        help='Duration of each segment in seconds (default: 20)')
    parser.add_argument('--max-segments', type=int, default=6,
                        help='Maximum segments per video; last segment gets remaining frames (default: 6)')
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
                segment_duration=args.segment_duration,
                max_segments=args.max_segments,
                copy_codec=not args.no_copy,
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