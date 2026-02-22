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
from utils.zip_utils import zip_and_remove_dir


# ---------------------------------------------------------------------------
# Helpers (shared with run_sam3_masklets.py)
# ---------------------------------------------------------------------------

def read_video_metadata(path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, width, height


def build_sam3_from_config(cfg):
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3["ckpt_path"])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return sam3_model, predictor


def save_masklets(
    predictor,
    inference_state,
    output_dir: str,
    fps: float,
    out_obj_ids: List[int],
    max_frame_num_to_track: int,
    frame_number_offset: int = 0,
):
    """
    Run SAM-3 propagation and save masks/images.

    *frame_number_offset* is added to every output filename so that frame
    numbers are globally unique across segments.
    """
    print("[INFO] Running SAM-3 propagation and saving masks...")
    video_segments = {}
    for (
        frame_idx,
        obj_ids,
        _low_res_masks,
        video_res_masks,
        _obj_scores,
        _iou_scores,
    ) in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=False,
        propagate_preflight=True,
    ):
        video_segments[int(frame_idx)] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().float().numpy()
            for i, out_obj_id in enumerate(obj_ids)
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
            F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
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


def _discover_segments(input_folder: str) -> List[str]:
    """
    Find all *_seg*.mp4 files in *input_folder*, sorted by segment number.
    """
    pattern = os.path.join(input_folder, "*_seg*.mp4")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No segment videos matching *_seg*.mp4 found in: {input_folder}"
        )
    return paths


def _extract_folder_name(input_folder: str) -> str:
    """
    Return the folder name (e.g. "428") from the input path.
    """
    return os.path.basename(os.path.normpath(input_folder))


def _segment_key_from_path(seg_path: str) -> str:
    """
    Return segment key from segment path (e.g. .../428_seg001.mp4 -> 428_seg001).
    """
    return os.path.splitext(os.path.basename(seg_path))[0]


def _points_to_bbox_xywh(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Convert two corner points [[x1,y1], [x2,y2]] to (x, y, w, h).
    Order of corners is arbitrary; min/max are used to get the rectangle.
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def load_annotation_json(
    annotation_folder: str,
    segment_key: str,
) -> List[Dict[str, Any]]:
    """
    Load annotation JSON for a segment from annotation_folder.
    File name must be <segment_key>.json (e.g. 428_seg001.json).

    Reads "shapes" (list). Each item: "label" -> real_id (int), "points" -> [[x1,y1],[x2,y2]].
    Returns list of bbox dicts with keys: real_id, x, y, w, h (derived from the two points).
    """
    path = os.path.join(annotation_folder, f"{segment_key}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

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
        points = item.get("points", [])
        x, y, w, h = _points_to_bbox_xywh(points)
        bbox_entries.append({
            "real_id": actual_pid,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })
    return bbox_entries


def bbox_xywh_to_rel(bbox_xywh: List[float], width: int, height: int) -> np.ndarray:
    """Convert [x, y, w, h] in pixels to relative [x1, y1, x2, y2] for SAM."""
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    rel = np.array(
        [x1 / width, y1 / height, x2 / width, y2 / height],
        dtype=np.float32,
    )
    return rel


def plot_mask_id_visualization(
    masks_dir: str,
    images_dir: str,
    segment_id_mappings: List[Dict[str, Any]],
    output_dir: str,
    frame_interval: int = 200,
) -> str:
    """
    Plot masks with tracking ID and real ID labels every *frame_interval* frames.
    Saves annotated frames to ``<output_dir>/mask_ids/``.
    """
    mask_id_dir = os.path.join(output_dir, "mask_ids")
    os.makedirs(mask_id_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not mask_paths:
        print("[WARN] No mask PNGs found for ID visualization.")
        return mask_id_dir

    _COLORS_BGR = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67),
        (49, 130, 245), (180, 30, 145), (244, 212, 66), (230, 50, 240),
        (69, 239, 191), (212, 190, 250), (144, 153, 70), (255, 190, 220),
    ]

    def _find_mapping(frame_idx: int):
        for seg in segment_id_mappings:
            fs = int(seg["frame_start"])
            fe = int(seg["frame_end"])
            if fs <= frame_idx <= fe:
                c2a = seg.get("consecutive_to_actual", {})
                c2a_int = {int(k): int(v) for k, v in c2a.items()}
                return c2a_int, seg.get("segment_key", "?")
        return {}, "?"

    num_plotted = 0
    for mask_path in mask_paths:
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        frame_idx = int(basename)

        if frame_idx % frame_interval != 0:
            continue

        mask_arr = np.array(Image.open(mask_path))
        obj_ids = np.unique(mask_arr)
        obj_ids = obj_ids[obj_ids != 0]

        if len(obj_ids) == 0:
            continue

        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = os.path.join(images_dir, basename + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break

        if img is None:
            h, w = mask_arr.shape[:2]
            img = np.zeros((h, w, 3), dtype=np.uint8)

        canvas = img.copy()
        c2a, seg_key = _find_mapping(frame_idx)

        for i, oid in enumerate(sorted(obj_ids)):
            color = _COLORS_BGR[i % len(_COLORS_BGR)]
            binary = (mask_arr == oid).astype(np.uint8) * 255

            overlay = np.zeros_like(canvas)
            overlay[binary > 0] = color
            cv2.addWeighted(canvas, 1.0, overlay, 0.4, 0, dst=canvas)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)

            M = cv2.moments(binary)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                coords = cv2.findNonZero(binary)
                if coords is not None:
                    bx, by, bw, bh = cv2.boundingRect(coords)
                    cx, cy = bx + bw // 2, by + bh // 2
                else:
                    continue

            tracking_id = int(oid)
            real_id = c2a.get(tracking_id, tracking_id)
            label = f"T:{tracking_id} R:{real_id}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (cx - 2, cy - th - 6), (cx + tw + 2, cy + 4), (0, 0, 0), -1)
            cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        header = f"Frame {frame_idx} | Seg: {seg_key}"
        cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(mask_id_dir, f"{basename}.jpg")
        cv2.imwrite(out_path, canvas)
        num_plotted += 1

    print(f"[INFO] Saved {num_plotted} mask ID visualization frames to: {mask_id_dir}")
    return mask_id_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch SAM-3 masklet extraction over video segments",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Folder containing video segments (e.g. .../428/)",
    )
    parser.add_argument(
        "--annotation-folder",
        type=str,
        required=True,
        help="Folder containing per-segment annotation JSONs (e.g. .../428/ has 428_seg001.json, ...)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/body4d.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <cfg.runtime.output_dir>/masklets_batch_<timestamp>)",
    )
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=1800,
        help="Maximum number of frames to track per segment",
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Offload video frames and state to CPU to reduce GPU memory (slower)",
    )
    args = parser.parse_args()

    # Resolve config
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = OmegaConf.load(cfg_path)

    # Discover segments
    input_folder = args.input_folder
    folder_name = _extract_folder_name(input_folder)
    print(f"[INFO] Folder name: {folder_name}")

    segment_paths = _discover_segments(input_folder)
    print(f"[INFO] Found {len(segment_paths)} segment(s):")
    for sp in segment_paths:
        print(f"  {sp}")

    annotation_folder = args.annotation_folder
    if not os.path.isdir(annotation_folder):
        raise FileNotFoundError(f"Annotation folder not found: {annotation_folder}")
    print(f"[INFO] Annotation folder: {annotation_folder}")

    # Per-segment ID mappings (each segment gets its own consecutive -> actual mapping)
    segment_id_mappings: List[Dict[str, Any]] = []
    all_actual_ids: set = set()

    # Output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_dir = os.path.join(
            cfg.runtime["output_dir"], f"masklets_batch_{timestamp}"
        )
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Read first segment to get video dimensions (assumed same for all segments)
    fps, _, width, height = read_video_metadata(segment_paths[0])
    print(f"[INFO] Video FPS: {fps}, WxH: {width}x{height}")

    # Build SAM-3 model (once, reused across segments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("[INFO] Initializing SAM-3 model...")
    _, predictor = build_sam3_from_config(cfg)

    cuda_reset_peak_memory_stats()

    # Process segments sequentially
    cumulative_frame_offset = 0
    all_vis_frames: List[np.ndarray] = []

    for seg_idx, seg_path in enumerate(segment_paths):
        seg_name = os.path.basename(seg_path)
        segment_key = _segment_key_from_path(seg_path)
        _, seg_total_frames, seg_w, seg_h = read_video_metadata(seg_path)
        print(f"\n{'='*60}")
        print(f"[INFO] Segment {seg_idx}: {seg_name}")
        print(f"[INFO]   frames: {seg_total_frames}, cumulative offset: {cumulative_frame_offset}")

        # Load bboxes from annotation JSON for this segment (same name as segment: <segment_key>.json)
        try:
            seg_bbox_entries = load_annotation_json(annotation_folder, segment_key)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Skipping segment.")
            cumulative_frame_offset += seg_total_frames
            continue
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to load annotation for '{segment_key}': {e}. Skipping segment.")
            cumulative_frame_offset += seg_total_frames
            continue
        if not seg_bbox_entries:
            print(
                f"[WARN] No shapes (or no valid bboxes) for segment key '{segment_key}'. Skipping segment."
            )
            cumulative_frame_offset += seg_total_frames
            continue

        # Init inference state for this segment
        print("[INFO]   Initializing SAM-3 inference state...")
        inference_state = predictor.init_state(
            video_path=seg_path,
            offload_video_to_cpu=args.offload_to_cpu,
            offload_state_to_cpu=args.offload_to_cpu,
        )
        predictor.clear_all_points_in_video(inference_state)

        # Per-segment mapping (fresh for each segment — IDs start from 1)
        seg_c2a: Dict[int, int] = {}

        # Add box prompts (at local frame 0) using bboxes from JSON (real_id, x, y, w, h)
        out_obj_ids: List[int] = []
        for i, bbox_entry in enumerate(seg_bbox_entries):
            actual_pid = int(bbox_entry.get("real_id", 0))
            consecutive_id = i + 1
            seg_c2a[consecutive_id] = actual_pid
            all_actual_ids.add(actual_pid)

            x = float(bbox_entry.get("x", 0))
            y = float(bbox_entry.get("y", 0))
            w = float(bbox_entry.get("w", 0))
            h = float(bbox_entry.get("h", 0))
            rel_box = bbox_xywh_to_rel([x, y, w, h], seg_w, seg_h)
            print(
                f"  Consecutive ID {consecutive_id} (actual PID {actual_pid}) "
                f"at local frame 0 (global {cumulative_frame_offset})"
            )
            _, out_obj_ids, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(consecutive_id),
                box=rel_box,
            )

        out_obj_ids = sorted(list(set(int(x) for x in out_obj_ids)))
        print(f"[INFO]   Tracking {len(out_obj_ids)} object(s): {out_obj_ids}")
        print(f"[INFO]   Segment mapping: {seg_c2a}")

        # Run propagation and save (with global frame numbering)
        vis_frames, num_saved = save_masklets(
            predictor=predictor,
            inference_state=inference_state,
            output_dir=output_dir,
            fps=fps,
            out_obj_ids=out_obj_ids,
            max_frame_num_to_track=int(args.max_frames_per_segment),
            frame_number_offset=cumulative_frame_offset,
        )
        all_vis_frames.extend(vis_frames)

        print(f"[INFO]   Saved {num_saved} frames for segment {seg_idx}.")

        # Record per-segment mapping with frame range
        segment_id_mappings.append({
            "segment_key": segment_key,
            "frame_start": cumulative_frame_offset,
            "frame_end": cumulative_frame_offset + num_saved - 1,
            "consecutive_to_actual": {str(k): v for k, v in seg_c2a.items()},
        })

        # Free GPU memory before loading next segment
        del inference_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cumulative_frame_offset += seg_total_frames

    # Save combined visualization video
    if all_vis_frames:
        combined_video_path = os.path.join(output_dir, "video_mask.mp4")
        images_to_mp4(all_vis_frames, combined_video_path, fps=fps)
        print(f"\n[INFO] Combined mask video saved to: {combined_video_path}")

    # Extract bounding boxes from all saved masks
    print("[INFO] Extracting bounding boxes from masks...")
    mask_bbox_data = extract_bboxes_from_masks(
        os.path.join(output_dir, "masks"),
        segment_id_mappings=segment_id_mappings,
    )
    mask_bbox_path = os.path.join(output_dir, "mask_bbox.json")
    write_json(mask_bbox_path, mask_bbox_data)
    print(f"[INFO] Saved mask bounding boxes to: {mask_bbox_path}")

    # Save per-segment ID mapping
    id_mapping = {
        "segments": segment_id_mappings,
        "all_actual_ids": sorted(all_actual_ids),
    }
    id_mapping_path = os.path.join(output_dir, "id_mapping.json")
    write_json(id_mapping_path, id_mapping)
    print(f"[INFO] Saved per-segment ID mapping to: {id_mapping_path}")

    # Save metadata
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

    # Plot mask ID visualization every 200 frames (before zipping)
    image_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    if os.path.isdir(masks_dir):
        try:
            plot_mask_id_visualization(
                masks_dir=masks_dir,
                images_dir=image_dir,
                segment_id_mappings=segment_id_mappings,
                output_dir=output_dir,
                frame_interval=200,
            )
        except Exception as e:
            print(f"[WARN] Mask ID visualization failed: {e}")

    # Zip images/ and masks/ to reduce inode count; later stages will unzip on demand
    for d in (image_dir, masks_dir):
        if os.path.isdir(d):
            try:
                zip_and_remove_dir(d)
            except Exception as e:
                print(f"[WARN] Failed to zip {d}: {e}")

    # GPU memory
    mem = cuda_mem_snapshot()
    write_json(os.path.join(output_dir, "gpu_mem_stage1_batch.json"), mem)
    print(f"\n[INFO] Peak GPU memory (batch stage1): {mem}")
    print(f"[INFO] Processed {len(segment_paths)} segments, {cumulative_frame_offset} total frames.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
