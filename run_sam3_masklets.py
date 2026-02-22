#!/usr/bin/env python3
"""
Stage 1/2: Run SAM-3 video tracking from user prompts (box/points) and SAVE masks.

Outputs in <output_dir>/:
  - images/00000000.jpg ... (RGB frames saved as JPG)
  - masks/00000000.png  ... (P-mode palette PNG; pixel value = obj_id, 0=background)
  - video_mask.mp4      ... visualization overlay video
  - masklets_meta.json  ... metadata for stage 2
  - gpu_mem_stage1.json ... peak GPU memory stats for this script
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from utils import mask_painter, images_to_mp4, DAVIS_PALETTE
from utils.image_utils import load_bbox_kp
from utils.gpu_profiler import cuda_mem_snapshot, cuda_reset_peak_memory_stats, write_json
from utils.mask_bbox import extract_bboxes_from_masks
from utils.model_factory import build_sam3_from_config
from utils.video_utils import read_video_metadata


def _parse_boxes(box_strs: List[str]) -> List[Tuple[int, int, np.ndarray]]:
    """
    Each box: "obj_id,frame_idx,x_min,y_min,x_max,y_max" in ABSOLUTE pixels.
    """
    parsed = []
    for s in box_strs:
        parts = s.split(",")
        if len(parts) != 6:
            raise ValueError(
                f"Invalid --boxes entry: {s}. Expected: obj_id,frame_idx,xmin,ymin,xmax,ymax"
            )
        obj_id = int(parts[0])
        frame_idx = int(parts[1])
        coords = np.array([float(p) for p in parts[2:]], dtype=np.float32)
        parsed.append((obj_id, frame_idx, coords))
    return parsed


def _parse_points(point_strs: List[str]) -> Dict[Tuple[int, int], Dict[str, List]]:
    """
    Each point: "obj_id,frame_idx,x,y,label" in ABSOLUTE pixels.
    Returns dict keyed by (obj_id, frame_idx): {"points":[[x,y],...], "labels":[...]}
    """
    points_by_obj_frame: Dict[Tuple[int, int], Dict[str, List]] = {}
    for s in point_strs:
        parts = s.split(",")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid --points entry: {s}. Expected: obj_id,frame_idx,x,y,label"
            )
        obj_id = int(parts[0])
        frame_idx = int(parts[1])
        x, y = float(parts[2]), float(parts[3])
        label = int(parts[4])
        key = (obj_id, frame_idx)
        if key not in points_by_obj_frame:
            points_by_obj_frame[key] = {"points": [], "labels": []}
        points_by_obj_frame[key]["points"].append([x, y])
        points_by_obj_frame[key]["labels"].append(label)
    return points_by_obj_frame


def save_masklets(
    video_path: str,
    predictor,
    inference_state,
    output_dir: str,
    fps: float,
    out_obj_ids: List[int],
    max_frame_num_to_track: int,
):
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
    for out_frame_idx in tqdm(range(0, len(video_segments)), desc="Saving masks"):
        img = inference_state["images"][out_frame_idx].detach().float().cpu()
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
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            img_vis = mask_painter(img_vis, mask, mask_color=4 + int(out_obj_id))
            msk[mask == 255] = int(out_obj_id)

        img_to_video.append(img_vis)
        msk_pil = Image.fromarray(msk.astype(np.uint8)).convert("P")
        msk_pil.putpalette(DAVIS_PALETTE)
        img_pil.save(os.path.join(image_dir, f"{out_frame_idx:08d}.jpg"))
        msk_pil.save(os.path.join(masks_dir, f"{out_frame_idx:08d}.png"))

    out_video_path = os.path.join(output_dir, "video_mask.mp4")
    images_to_mp4(img_to_video, out_video_path, fps=fps)
    print(f"[INFO] Mask video saved to: {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: SAM-3 masklet saving")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--config", type=str, default="configs/body4d.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <cfg.runtime.output_dir>/masklets_<timestamp>)",
    )
    parser.add_argument(
        "--boxes",
        type=str,
        nargs="+",
        default=None,
        help="Boxes (ABS px): 'obj_id,frame_idx,xmin,ymin,xmax,ymax' ...",
    )
    parser.add_argument(
        "--points",
        type=str,
        nargs="+",
        default=None,
        help="Points (ABS px): 'obj_id,frame_idx,x,y,label' ... label 1=pos 0=neg",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1800,
        help="Maximum number of frames to track/save",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = OmegaConf.load(cfg_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_dir = os.path.join(cfg.runtime["output_dir"], f"masklets_{timestamp}")
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    fps, total_frames, width, height = read_video_metadata(args.video)
    print(f"[INFO] Video FPS: {fps}, frames: {total_frames}, WxH: {width}x{height}")

    print("[INFO] Initializing SAM-3 model...")
    _, predictor = build_sam3_from_config(cfg)

    cuda_reset_peak_memory_stats()

    print("[INFO] Initializing SAM-3 inference state...")
    inference_state = predictor.init_state(video_path=args.video)
    predictor.clear_all_points_in_video(inference_state)

    # Build ID mapping for SAM3D-Body compatibility.
    # SAM3D-Body expects consecutive IDs (1, 2, 3, ...) internally.
    # We use consecutive IDs for SAM3 tracking and map them back to actual PIDs later.
    consecutive_to_actual: Dict[int, int] = {}
    actual_to_consecutive: Dict[int, int] = {}
    out_obj_ids: List[int] = []
    
    if args.boxes is not None:
        print("[INFO] Adding box prompts...")
        parsed_boxes = _parse_boxes(args.boxes)
        # Collect all unique actual IDs and build mapping
        unique_actual_ids = sorted(set(obj_id for obj_id, _, _ in parsed_boxes))
        for i, actual_id in enumerate(unique_actual_ids):
            consecutive_id = i + 1
            consecutive_to_actual[consecutive_id] = actual_id
            actual_to_consecutive[actual_id] = consecutive_id
        
        for actual_id, frame_idx, box_abs in parsed_boxes:
            consecutive_id = actual_to_consecutive[actual_id]
            rel_box = box_abs / np.array([width, height, width, height], dtype=np.float32)
            print(f"  Consecutive ID {consecutive_id} (actual ID {actual_id}) at frame {frame_idx}")
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(consecutive_id),  # Use consecutive ID for SAM3
                box=rel_box,
            )

    if args.points is not None:
        print("[INFO] Adding point prompts...")
        points_by_obj_frame = _parse_points(args.points)
        # Collect all unique actual IDs and extend mapping
        unique_actual_ids_pts = sorted(set(obj_id for (obj_id, _) in points_by_obj_frame.keys()))
        # Merge with existing mapping from boxes
        all_actual_ids = sorted(set(list(actual_to_consecutive.keys()) + unique_actual_ids_pts))
        # Rebuild mapping to ensure consecutive IDs
        consecutive_to_actual = {}
        actual_to_consecutive = {}
        for i, actual_id in enumerate(all_actual_ids):
            consecutive_id = i + 1
            consecutive_to_actual[consecutive_id] = actual_id
            actual_to_consecutive[actual_id] = consecutive_id
        
        for (actual_id, frame_idx), data in points_by_obj_frame.items():
            consecutive_id = actual_to_consecutive[actual_id]
            pts = np.array(data["points"], dtype=np.float32)
            pts[:, 0] /= float(width)
            pts[:, 1] /= float(height)
            points_tensor = torch.tensor(pts, dtype=torch.float32)
            labels_tensor = torch.tensor(data["labels"], dtype=torch.int32)
            print(f"  Consecutive ID {consecutive_id} (actual ID {actual_id}) at frame {frame_idx}")
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(consecutive_id),  # Use consecutive ID for SAM3
                points=points_tensor,
                labels=labels_tensor,
            )

    # If no prompts were provided, fall back to the current repo's hardcoded bbox source
    # (mirrors `infer_video.py` behavior).
    if args.boxes is None and args.points is None:
        print("[WARN] No --boxes/--points provided; using hardcoded bboxes_kps_refined prompts (frame 0).")
        bboxes_kps_data = load_bbox_kp("/mnt/data/sam4d_body/inputs/bboxes_kps_refined", "428")
        if bboxes_kps_data is None:
            raise RuntimeError("Failed to load hardcoded bbox/kp pickle for prompts.")
        selected_boxes = list(range(len(bboxes_kps_data[0]["bboxes"])))
        pid_list = bboxes_kps_data[0]["pids"]
        
        # Build mapping from consecutive IDs to actual PIDs
        for i, pid in enumerate(pid_list):
            consecutive_id = i + 1  # 1-based consecutive IDs
            actual_pid = int(pid)
            consecutive_to_actual[consecutive_id] = actual_pid
            actual_to_consecutive[actual_pid] = consecutive_id
        
        for bbox_idx in selected_boxes:
            consecutive_id = bbox_idx + 1  # Use consecutive IDs for SAM3
            actual_pid = int(pid_list[bbox_idx])
            bbox = np.array(bboxes_kps_data[0]["bboxes"][bbox_idx], dtype=np.float32)
            rel_box = bbox / np.array([width, height, width, height], dtype=np.float32)
            print(f"  Consecutive ID {consecutive_id} (actual PID {actual_pid}) at frame 0")
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=consecutive_id,  # Use consecutive ID for SAM3
                box=rel_box,
            )

    out_obj_ids = sorted(list(set([int(x) for x in out_obj_ids])))
    print(f"[INFO] Tracking {len(out_obj_ids)} object(s) with consecutive IDs: {out_obj_ids}")
    print(f"[INFO] ID mapping (consecutive -> actual): {consecutive_to_actual}")

    save_masklets(
        video_path=args.video,
        predictor=predictor,
        inference_state=inference_state,
        output_dir=output_dir,
        fps=fps,
        out_obj_ids=out_obj_ids,
        max_frame_num_to_track=int(args.max_frames),
    )

    # Extract per-person bounding boxes from saved masks
    print("[INFO] Extracting bounding boxes from masks...")
    mask_bbox_data = extract_bboxes_from_masks(
        os.path.join(output_dir, "masks"),
        consecutive_to_actual=consecutive_to_actual,
    )
    mask_bbox_path = os.path.join(output_dir, "mask_bbox.json")
    write_json(mask_bbox_path, mask_bbox_data)
    print(f"[INFO] Saved mask bounding boxes to: {mask_bbox_path}")

    # Save ID mapping to separate file for stage 2 to use
    id_mapping = {
        "consecutive_to_actual": {str(k): v for k, v in consecutive_to_actual.items()},
        "actual_to_consecutive": {str(k): v for k, v in actual_to_consecutive.items()},
    }
    id_mapping_path = os.path.join(output_dir, "id_mapping.json")
    write_json(id_mapping_path, id_mapping)
    print(f"[INFO] Saved ID mapping to: {id_mapping_path}")
    
    meta = {
        "video_path": os.path.abspath(args.video),
        "config_path": os.path.abspath(cfg_path),
        "output_dir": os.path.abspath(output_dir),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "out_obj_ids": out_obj_ids,
        "image_dir": os.path.join(os.path.abspath(output_dir), "images"),
        "masks_dir": os.path.join(os.path.abspath(output_dir), "masks"),
        "id_mapping_path": id_mapping_path,
        "consecutive_to_actual": consecutive_to_actual,
    }
    write_json(os.path.join(output_dir, "masklets_meta.json"), meta)

    mem = cuda_mem_snapshot()
    write_json(os.path.join(output_dir, "gpu_mem_stage1.json"), mem)
    print(f"[INFO] Peak GPU memory (stage1): {mem}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()


