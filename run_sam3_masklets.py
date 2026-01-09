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
import json
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


def _cuda_mem_snapshot() -> Dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "device": str(torch.cuda.get_device_name(0)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        _obj_ids,
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
            for i, out_obj_id in enumerate(out_obj_ids)
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("[INFO] Initializing SAM-3 inference state...")
    inference_state = predictor.init_state(video_path=args.video)
    predictor.clear_all_points_in_video(inference_state)

    out_obj_ids: List[int] = []
    if args.boxes is not None:
        print("[INFO] Adding box prompts...")
        for obj_id, frame_idx, box_abs in _parse_boxes(args.boxes):
            rel_box = box_abs / np.array([width, height, width, height], dtype=np.float32)
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                box=rel_box,
            )

    if args.points is not None:
        print("[INFO] Adding point prompts...")
        points_by_obj_frame = _parse_points(args.points)
        for (obj_id, frame_idx), data in points_by_obj_frame.items():
            pts = np.array(data["points"], dtype=np.float32)
            pts[:, 0] /= float(width)
            pts[:, 1] /= float(height)
            points_tensor = torch.tensor(pts, dtype=torch.float32)
            labels_tensor = torch.tensor(data["labels"], dtype=torch.int32)
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
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
        for bbox_idx in selected_boxes:
            obj_id = int(pid_list[bbox_idx])
            bbox = np.array(bboxes_kps_data[0]["bboxes"][bbox_idx], dtype=np.float32)
            rel_box = bbox / np.array([width, height, width, height], dtype=np.float32)
            _, out_obj_ids, _low_res_masks, _video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=rel_box,
            )

    out_obj_ids = sorted(list(set([int(x) for x in out_obj_ids])))
    print(f"[INFO] Tracking {len(out_obj_ids)} object(s): {out_obj_ids}")

    save_masklets(
        video_path=args.video,
        predictor=predictor,
        inference_state=inference_state,
        output_dir=output_dir,
        fps=fps,
        out_obj_ids=out_obj_ids,
        max_frame_num_to_track=int(args.max_frames),
    )

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
    }
    _write_json(os.path.join(output_dir, "masklets_meta.json"), meta)

    mem = _cuda_mem_snapshot()
    _write_json(os.path.join(output_dir, "gpu_mem_stage1.json"), mem)
    print(f"[INFO] Peak GPU memory (stage1): {mem}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()


