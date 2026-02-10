#!/usr/bin/env python3
"""
Batch SAM-3 masklet extraction over all video segments in a folder.

Expected input folder layout (folder name is a 3-digit number):
  <input_folder>/            e.g. /mnt/data/.../428/
    428_seg000.mp4
    428_seg001.mp4
    428_seg002.mp4
    ...

The folder name (e.g. "428") is used as the pickle file name for loading
bboxes/keypoints from <bbox_pkl_dir>/428.pkl.

The pickle is indexed by **absolute frame number** across the entire long
video.  For each segment we look up the bboxes at the cumulative frame
offset (i.e. frame 0 for seg000, frame N0 for seg001, frame N0+N1 for
seg002, ...).

Usage:
  python run_sam3_masklets_batch.py \\
      --input-folder /mnt/data/sam4d_body/inputs/videos/428 \\
      --bbox-pkl-dir /mnt/data/sam4d_body/inputs/bboxes_kps_refined \\
      --output /mnt/data/sam4d_body/outputs/exp_XXX/masklets \\
      --config configs/body4d.yaml
"""

import argparse
import gc
import glob
import os
import re
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
        help="Folder containing video segments (e.g. .../428/ with 428_seg000.mp4 ...)",
    )
    parser.add_argument(
        "--bbox-pkl-dir",
        type=str,
        required=True,
        help="Directory containing bbox/kp pickle files (e.g. .../bboxes_kps_refined/)",
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
    print(f"[INFO] Folder name (pkl key): {folder_name}")

    segment_paths = _discover_segments(input_folder)
    print(f"[INFO] Found {len(segment_paths)} segment(s):")
    for sp in segment_paths:
        print(f"  {sp}")

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

    # Load bbox pickle (entire long video)
    print(f"[INFO] Loading bbox/kp pickle from: {args.bbox_pkl_dir}/{folder_name}.pkl")
    bboxes_kps_data = load_bbox_kp(args.bbox_pkl_dir, folder_name)
    if bboxes_kps_data is None:
        raise RuntimeError(
            f"Failed to load bbox/kp pickle: {args.bbox_pkl_dir}/{folder_name}.pkl"
        )
    print(f"[INFO] Pickle contains {len(bboxes_kps_data)} frame entries.")

    # Read first segment to get video dimensions (assumed same for all segments)
    fps, _, width, height = read_video_metadata(segment_paths[0])
    print(f"[INFO] Video FPS: {fps}, WxH: {width}x{height}")

    # Build SAM-3 model (once, reused across segments)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("[INFO] Initializing SAM-3 model...")
    _, predictor = build_sam3_from_config(cfg)

    cuda_reset_peak_memory_stats()

    # Build ID mapping from the first frame of the pickle
    # (person IDs / bboxes are assumed consistent across frames)
    pid_list = bboxes_kps_data[0]["pids"]
    num_persons = len(pid_list)
    consecutive_to_actual: Dict[int, int] = {}
    actual_to_consecutive: Dict[int, int] = {}
    for i, pid in enumerate(pid_list):
        consecutive_id = i + 1
        actual_pid = int(pid)
        consecutive_to_actual[consecutive_id] = actual_pid
        actual_to_consecutive[actual_pid] = consecutive_id
    print(f"[INFO] {num_persons} person(s), ID mapping: {consecutive_to_actual}")

    # Process segments sequentially
    cumulative_frame_offset = 0
    all_vis_frames: List[np.ndarray] = []

    for seg_idx, seg_path in enumerate(segment_paths):
        seg_name = os.path.basename(seg_path)
        _, seg_total_frames, seg_w, seg_h = read_video_metadata(seg_path)
        print(f"\n{'='*60}")
        print(f"[INFO] Segment {seg_idx}: {seg_name}")
        print(f"[INFO]   frames: {seg_total_frames}, cumulative offset: {cumulative_frame_offset}")
        print(f"[INFO]   using bbox frame index: {cumulative_frame_offset}")

        # Look up bboxes at the absolute frame offset
        pkl_frame_idx = cumulative_frame_offset
        if pkl_frame_idx >= len(bboxes_kps_data):
            print(
                f"[WARN] Pickle has no entry for frame {pkl_frame_idx} "
                f"(only {len(bboxes_kps_data)} entries). Skipping segment."
            )
            cumulative_frame_offset += seg_total_frames
            continue

        frame_rec = bboxes_kps_data[pkl_frame_idx]
        seg_bboxes = np.asarray(frame_rec["bboxes"], dtype=np.float32)
        seg_pids = frame_rec["pids"]

        # Init inference state for this segment
        print("[INFO]   Initializing SAM-3 inference state...")
        inference_state = predictor.init_state(
            video_path=seg_path,
            offload_video_to_cpu=args.offload_to_cpu,
            offload_state_to_cpu=args.offload_to_cpu,
        )
        predictor.clear_all_points_in_video(inference_state)

        # Add box prompts (at local frame 0) using bboxes from pkl
        out_obj_ids: List[int] = []
        for bbox_idx in range(len(seg_bboxes)):
            actual_pid = int(seg_pids[bbox_idx])
            consecutive_id = actual_to_consecutive.get(actual_pid)
            if consecutive_id is None:
                # New person appeared in later frame — extend mapping
                consecutive_id = max(consecutive_to_actual.keys(), default=0) + 1
                consecutive_to_actual[consecutive_id] = actual_pid
                actual_to_consecutive[actual_pid] = consecutive_id

            bbox = seg_bboxes[bbox_idx]
            rel_box = bbox / np.array([seg_w, seg_h, seg_w, seg_h], dtype=np.float32)
            print(
                f"  Consecutive ID {consecutive_id} (actual PID {actual_pid}) "
                f"at local frame 0 (global {pkl_frame_idx})"
            )
            _, out_obj_ids, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(consecutive_id),
                box=rel_box,
            )

        out_obj_ids = sorted(list(set(int(x) for x in out_obj_ids)))
        print(f"[INFO]   Tracking {len(out_obj_ids)} object(s): {out_obj_ids}")

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
        consecutive_to_actual=consecutive_to_actual,
    )
    mask_bbox_path = os.path.join(output_dir, "mask_bbox.json")
    write_json(mask_bbox_path, mask_bbox_data)
    print(f"[INFO] Saved mask bounding boxes to: {mask_bbox_path}")

    # Save ID mapping
    id_mapping = {
        "consecutive_to_actual": {str(k): v for k, v in consecutive_to_actual.items()},
        "actual_to_consecutive": {str(k): v for k, v in actual_to_consecutive.items()},
    }
    id_mapping_path = os.path.join(output_dir, "id_mapping.json")
    write_json(id_mapping_path, id_mapping)
    print(f"[INFO] Saved ID mapping to: {id_mapping_path}")

    # Save metadata
    meta = {
        "input_folder": os.path.abspath(input_folder),
        "folder_name": folder_name,
        "config_path": os.path.abspath(cfg_path),
        "output_dir": os.path.abspath(output_dir),
        "fps": fps,
        "total_frames": cumulative_frame_offset,
        "width": width,
        "height": height,
        "num_segments": len(segment_paths),
        "segment_paths": [os.path.abspath(p) for p in segment_paths],
        "out_obj_ids": sorted(list(consecutive_to_actual.keys())),
        "image_dir": os.path.join(os.path.abspath(output_dir), "images"),
        "masks_dir": os.path.join(os.path.abspath(output_dir), "masks"),
        "id_mapping_path": id_mapping_path,
        "consecutive_to_actual": consecutive_to_actual,
    }
    write_json(os.path.join(output_dir, "masklets_meta.json"), meta)

    # GPU memory
    mem = cuda_mem_snapshot()
    write_json(os.path.join(output_dir, "gpu_mem_stage1_batch.json"), mem)
    print(f"\n[INFO] Peak GPU memory (batch stage1): {mem}")
    print(f"[INFO] Processed {len(segment_paths)} segments, {cumulative_frame_offset} total frames.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
