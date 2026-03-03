#!/usr/bin/env python3
"""
Batch SAM-3 masklet extraction over all video segments in a folder.

Expected input folder layout:
  <input_folder>/            e.g. /mnt/data/.../428/
    428_seg001.mp4
    428_seg002.mp4
    ...

  <annotation_folder>/       (path given by --annotation-folder)
    428_seg001.json
    428_seg002.json
    ...

Each annotation JSON has the same base name as the segment (e.g. 428_seg001.json for
428_seg001.mp4). In each file, the "shapes" field is a list; each item is one person:
  - "label": real person ID (string or number, converted to int)
  - "points": [[x1, y1], [x2, y2]] — two corners of the bbox rectangle (any order).
The script derives (x, y, w, h) from the two points and builds consecutive IDs for tracking.

Usage:
  python run_sam3_masklets_batch.py \\
      --input-folder /mnt/data/sam4d_body/inputs/videos/428 \\
      --annotation-folder /mnt/data/sam4d_body/inputs/annotations/428 \\
      --output /mnt/data/sam4d_body/outputs/exp_XXX/masklets \\
      --config configs/body4d.yaml
"""

import argparse
import gc
import glob
import json
import os
import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from utils import mask_painter, images_to_mp4, DAVIS_PALETTE
from utils.gpu_profiler import cuda_mem_snapshot, cuda_reset_peak_memory_stats, write_json
from utils.mask_bbox import extract_bboxes_from_masks
from utils.model_factory import build_sam3_from_config
from utils.plot_mask_ids import plot_mask_id_visualization
from utils.video_utils import read_video_metadata
from utils.zip_utils import zip_and_remove_dir


# ---------------------------------------------------------------------------
# Annotation / bbox helpers
# ---------------------------------------------------------------------------

def _discover_segments(input_folder: str) -> List[str]:
    """Find all *_seg*.mp4 files in *input_folder*, sorted by segment number."""
    pattern = os.path.join(input_folder, "*_seg*.mp4")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No segment videos matching *_seg*.mp4 found in: {input_folder}"
        )
    return paths


def _segment_key_from_path(seg_path: str) -> str:
    return os.path.splitext(os.path.basename(seg_path))[0]


def _points_to_bbox_xywh(points: List[List[float]]) -> Tuple[float, float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])
    return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)


def load_annotation_json(
    annotation_folder: str,
    segment_key: str,
) -> List[Dict[str, Any]]:
    """Load annotation JSON for a segment.  Returns list of bbox dicts."""
    pattern = os.path.join(annotation_folder, f"{segment_key}*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Annotation file not found for pattern: {pattern}")
    path = matches[0]
    if len(matches) > 1:
        print(
            f"[WARN] Multiple annotation files matched for '{segment_key}'. "
            f"Using first sorted match: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"{path}: 'shapes' must be a list")

    bbox_entries: List[Dict[str, Any]] = []
    for item in shapes:
        label = item.get("label")
        if label is None:
            continue
        actual_pid = int(label) if not isinstance(label, int) else label
        x, y, w, h = _points_to_bbox_xywh(item.get("points", []))
        bbox_entries.append({"real_id": actual_pid, "x": x, "y": y, "w": w, "h": h})
    return bbox_entries


def bbox_xywh_to_rel(bbox_xywh: List[float], width: int, height: int) -> np.ndarray:
    """Convert [x, y, w, h] in pixels to relative [x1, y1, x2, y2] for SAM."""
    x, y, w, h = bbox_xywh
    return np.array(
        [x / width, y / height, (x + w) / width, (y + h) / height],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Propagation / mask saving
# ---------------------------------------------------------------------------

def save_masklets(
    predictor,
    inference_state,
    output_dir: str,
    fps: float,
    out_obj_ids: List[int],
    max_frame_num_to_track: int,
    frame_number_offset: int = 0,
):
    """Run SAM-3 propagation and save masks/images with global frame numbering."""
    print("[INFO] Running SAM-3 propagation and saving masks...")
    video_segments = {}
    for (
        frame_idx, obj_ids, _low_res_masks, video_res_masks, _obj_scores, _iou_scores,
    ) in predictor.propagate_in_video(
        inference_state, start_frame_idx=0,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=False, propagate_preflight=True,
    ):
        video_segments[int(frame_idx)] = {
            oid: (video_res_masks[i] > 0.0).cpu().float().numpy()
            for i, oid in enumerate(obj_ids)
        }

    out_h = inference_state["video_height"]
    out_w = inference_state["video_width"]

    image_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    img_to_video = []
    num_frames_saved = 0
    for local_idx in tqdm(range(len(video_segments)), desc="Saving masks"):
        global_idx = local_idx + frame_number_offset

        img = inference_state["images"][local_idx].detach().float().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = (
            F.interpolate(img.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False)
            .squeeze(0).permute(1, 2, 0)
        )
        img = (img.float().numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img).convert("RGB")

        msk = np.zeros_like(img[:, :, 0], dtype=np.uint16)
        img_vis = img.copy()
        for out_obj_id, out_mask in video_segments[local_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            img_vis = mask_painter(img_vis, mask, mask_color=4 + int(out_obj_id))
            msk[mask == 255] = int(out_obj_id)

        img_to_video.append(img_vis)
        msk_pil = Image.fromarray(msk.astype(np.uint8)).convert("P")
        msk_pil.putpalette(DAVIS_PALETTE)
        img_pil.save(os.path.join(image_dir, f"{global_idx:08d}.jpg"))
        msk_pil.save(os.path.join(masks_dir, f"{global_idx:08d}.png"))
        num_frames_saved += 1

    return img_to_video, num_frames_saved


def save_empty_masks(
    output_dir: str,
    frame_number_offset: int,
    num_frames: int,
    width: int,
    height: int,
) -> int:
    """Write empty black mask PNGs for a skipped segment."""
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    empty_mask = np.zeros((height, width), dtype=np.uint8)
    for local_idx in range(num_frames):
        global_idx = frame_number_offset + local_idx
        msk_pil = Image.fromarray(empty_mask).convert("P")
        msk_pil.putpalette(DAVIS_PALETTE)
        msk_pil.save(os.path.join(masks_dir, f"{global_idx:08d}.png"))

    return num_frames


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def process_single_segment(
    predictor,
    seg_path: str,
    seg_bbox_entries: List[Dict[str, Any]],
    output_dir: str,
    fps: float,
    max_frames: int,
    frame_offset: int,
    seg_w: int,
    seg_h: int,
    offload_to_cpu: bool,
    cam_res_scale: float,
) -> Tuple[List[np.ndarray], int, Dict[int, int]]:
    """Process one video segment: init state, add prompts, propagate, save.

    Returns ``(vis_frames, num_saved, consecutive_to_actual_mapping)``.
    """
    inference_state = predictor.init_state(
        video_path=seg_path,
        offload_video_to_cpu=offload_to_cpu,
        offload_state_to_cpu=offload_to_cpu,
    )
    predictor.clear_all_points_in_video(inference_state)

    seg_c2a: Dict[int, int] = {}
    out_obj_ids: List[int] = []
    for i, bbox_entry in enumerate(seg_bbox_entries):
        actual_pid = int(bbox_entry.get("real_id", 0))
        consecutive_id = i + 1
        seg_c2a[consecutive_id] = actual_pid

        x = float(bbox_entry.get("x", 0))
        y = float(bbox_entry.get("y", 0))
        w = float(bbox_entry.get("w", 0))
        h = float(bbox_entry.get("h", 0))
        if cam_res_scale != 1.0:
            x *= cam_res_scale
            y *= cam_res_scale
            w *= cam_res_scale
            h *= cam_res_scale
        rel_box = bbox_xywh_to_rel([x, y, w, h], seg_w, seg_h)
        print(
            f"  Consecutive ID {consecutive_id} (actual PID {actual_pid}) "
            f"at local frame 0 (global {frame_offset})"
        )
        _, out_obj_ids, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0, obj_id=int(consecutive_id), box=rel_box,
        )

    out_obj_ids = sorted(set(int(x) for x in out_obj_ids))
    print(f"[INFO]   Tracking {len(out_obj_ids)} object(s): {out_obj_ids}")
    print(f"[INFO]   Segment mapping: {seg_c2a}")

    vis_frames, num_saved = save_masklets(
        predictor=predictor,
        inference_state=inference_state,
        output_dir=output_dir,
        fps=fps,
        out_obj_ids=out_obj_ids,
        max_frame_num_to_track=max_frames,
        frame_number_offset=frame_offset,
    )

    del inference_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vis_frames, num_saved, seg_c2a


# ---------------------------------------------------------------------------
# Post-processing (save all outputs, zip)
# ---------------------------------------------------------------------------

def save_batch_outputs(
    output_dir: str,
    segment_id_mappings: List[Dict[str, Any]],
    all_actual_ids: set,
    all_vis_frames: List[np.ndarray],
    fps: float,
    cumulative_frame_offset: int,
    *,
    input_folder: str,
    annotation_folder: str,
    folder_name: str,
    cfg_path: str,
    segment_paths: List[str],
    width: int,
    height: int,
) -> None:
    """Save combined video, bboxes, ID mappings, metadata, mask-ID viz, then zip."""
    # Combined visualization video
    if all_vis_frames:
        combined_video_path = os.path.join(output_dir, "video_mask.mp4")
        images_to_mp4(all_vis_frames, combined_video_path, fps=fps)
        print(f"\n[INFO] Combined mask video saved to: {combined_video_path}")

    # Bounding boxes from masks
    print("[INFO] Extracting bounding boxes from masks...")
    mask_bbox_data = extract_bboxes_from_masks(
        os.path.join(output_dir, "masks"),
        segment_id_mappings=segment_id_mappings,
    )
    write_json(os.path.join(output_dir, "mask_bbox.json"), mask_bbox_data)

    # Per-segment ID mapping
    id_mapping = {
        "segments": segment_id_mappings,
        "all_actual_ids": sorted(all_actual_ids),
    }
    id_mapping_path = os.path.join(output_dir, "id_mapping.json")
    write_json(id_mapping_path, id_mapping)
    print(f"[INFO] Saved per-segment ID mapping to: {id_mapping_path}")

    # Metadata
    meta = {
        "input_folder": os.path.abspath(input_folder),
        "annotation_folder": os.path.abspath(annotation_folder),
        "folder_name": folder_name,
        "config_path": os.path.abspath(cfg_path),
        "output_dir": os.path.abspath(output_dir),
        "fps": fps,
        "total_frames": cumulative_frame_offset,
        "width": width,
        "height": height,
        "num_segments": len(segment_paths),
        "segment_paths": [os.path.abspath(p) for p in segment_paths],
        "out_obj_ids": sorted(all_actual_ids),
        "image_dir": os.path.join(os.path.abspath(output_dir), "images"),
        "masks_dir": os.path.join(os.path.abspath(output_dir), "masks"),
        "id_mapping_path": id_mapping_path,
        "segment_id_mappings": segment_id_mappings,
    }
    write_json(os.path.join(output_dir, "masklets_meta.json"), meta)

    # Mask ID visualization (before zipping)
    image_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    if os.path.isdir(masks_dir):
        try:
            plot_mask_id_visualization(
                masks_dir=masks_dir, images_dir=image_dir,
                segment_id_mappings=segment_id_mappings,
                output_dir=output_dir, frame_interval=200,
            )
        except Exception as e:
            print(f"[WARN] Mask ID visualization failed: {e}")

    # Zip images/ and masks/
    for d in (image_dir, masks_dir):
        if os.path.isdir(d):
            try:
                zip_and_remove_dir(d)
            except Exception as e:
                print(f"[WARN] Failed to zip {d}: {e}")

    # GPU memory snapshot
    mem = cuda_mem_snapshot()
    write_json(os.path.join(output_dir, "gpu_mem_stage1_batch.json"), mem)
    print(f"\n[INFO] Peak GPU memory (batch stage1): {mem}")
    print(f"[INFO] Processed {len(segment_paths)} segments, {cumulative_frame_offset} total frames.")
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch SAM-3 masklet extraction over video segments",
    )
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing video segments (e.g. .../428/)")
    parser.add_argument("--annotation-folder", type=str, required=True,
                        help="Folder containing per-segment annotation JSONs")
    parser.add_argument("--config", type=str, default="configs/body4d.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--max-frames-per-segment", type=int, default=1800,
                        help="Maximum number of frames to track per segment")
    parser.add_argument("--offload-to-cpu", action="store_true",
                        help="Offload video frames and state to CPU to reduce GPU memory (slower)")
    parser.add_argument(
        "--cam-res-scale",
        type=float,
        default=0.5,
        help="Scale factor applied to annotation [x,y,w,h] before converting to relative coordinates",
    )
    args = parser.parse_args()

    # Resolve config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = OmegaConf.load(cfg_path)
    if args.cam_res_scale <= 0:
        raise ValueError(f"--cam-res-scale must be > 0, got {args.cam_res_scale}")
    print(f"[INFO] Annotation coordinate scale (cam_res_scale): {args.cam_res_scale}")

    # Discover segments
    input_folder = args.input_folder
    folder_name = os.path.basename(os.path.normpath(input_folder))
    print(f"[INFO] Folder name: {folder_name}")

    segment_paths = _discover_segments(input_folder)
    print(f"[INFO] Found {len(segment_paths)} segment(s):")
    for sp in segment_paths:
        print(f"  {sp}")

    annotation_folder = args.annotation_folder
    if not os.path.isdir(annotation_folder):
        raise FileNotFoundError(f"Annotation folder not found: {annotation_folder}")

    # Output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(cfg.runtime["output_dir"], f"masklets_batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Video dimensions from first segment
    fps, _, width, height = read_video_metadata(segment_paths[0])
    print(f"[INFO] Video FPS: {fps}, WxH: {width}x{height}")

    # Build SAM-3 model (reused across segments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("[INFO] Initializing SAM-3 model...")
    _, predictor = build_sam3_from_config(cfg)
    cuda_reset_peak_memory_stats()

    # Process each segment
    segment_id_mappings: List[Dict[str, Any]] = []
    all_actual_ids: set = set()
    cumulative_frame_offset = 0
    all_vis_frames: List[np.ndarray] = []

    for seg_idx, seg_path in enumerate(segment_paths):
        segment_key = _segment_key_from_path(seg_path)
        _, seg_total_frames, seg_w, seg_h = read_video_metadata(seg_path)
        print(f"\n{'='*60}")
        print(f"[INFO] Segment {seg_idx}: {os.path.basename(seg_path)}")
        print(f"[INFO]   frames: {seg_total_frames}, cumulative offset: {cumulative_frame_offset}")

        try:
            seg_bbox_entries = load_annotation_json(annotation_folder, segment_key)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Skipping SAM propagation; writing empty masks for this segment.")
            num_empty = save_empty_masks(
                output_dir=output_dir,
                frame_number_offset=cumulative_frame_offset,
                num_frames=seg_total_frames,
                width=seg_w,
                height=seg_h,
            )
            segment_id_mappings.append({
                "segment_key": segment_key,
                "frame_start": cumulative_frame_offset,
                "frame_end": cumulative_frame_offset + num_empty - 1,
                "consecutive_to_actual": {},
            })
            cumulative_frame_offset += seg_total_frames
            continue
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to load annotation for '{segment_key}': {e}. Skipping segment.")
            cumulative_frame_offset += seg_total_frames
            continue
        if not seg_bbox_entries:
            print(f"[WARN] No shapes for segment key '{segment_key}'. Skipping segment.")
            cumulative_frame_offset += seg_total_frames
            continue

        vis_frames, num_saved, seg_c2a = process_single_segment(
            predictor=predictor,
            seg_path=seg_path,
            seg_bbox_entries=seg_bbox_entries,
            output_dir=output_dir,
            fps=fps,
            max_frames=int(args.max_frames_per_segment),
            frame_offset=cumulative_frame_offset,
            seg_w=seg_w, seg_h=seg_h,
            offload_to_cpu=args.offload_to_cpu,
            cam_res_scale=float(args.cam_res_scale),
        )
        all_vis_frames.extend(vis_frames)
        all_actual_ids.update(seg_c2a.values())
        print(f"[INFO]   Saved {num_saved} frames for segment {seg_idx}.")

        segment_id_mappings.append({
            "segment_key": segment_key,
            "frame_start": cumulative_frame_offset,
            "frame_end": cumulative_frame_offset + num_saved - 1,
            "consecutive_to_actual": {str(k): v for k, v in seg_c2a.items()},
        })
        cumulative_frame_offset += seg_total_frames

    # Save everything
    save_batch_outputs(
        output_dir=output_dir,
        segment_id_mappings=segment_id_mappings,
        all_actual_ids=all_actual_ids,
        all_vis_frames=all_vis_frames,
        fps=fps,
        cumulative_frame_offset=cumulative_frame_offset,
        input_folder=input_folder,
        annotation_folder=annotation_folder,
        folder_name=folder_name,
        cfg_path=cfg_path,
        segment_paths=segment_paths,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    main()
