#!/usr/bin/env python3
"""
Stage 2 (decoupled): masks/images -> raw MHR params (no temporal smoothing).

Inputs:
  - Stage1 folder with images/ and masks/ (palette masks where pixel value is obj_id)
  - Config YAML for SAM-3D-Body

Outputs:
  - raw_mhr.pt (torch serialized): contains pose_output["mhr"] tensors (flattened B=T*N)
    plus metadata needed for Stage3 smoothing.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_DIR, "models", "sam_3d_body"))

from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import save_raw_mhr
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.gpu_profiler import cuda_mem_snapshot, cuda_reset_peak_memory_stats, write_json
from utils.id_mapping import load_segment_id_mappings, to_actual_pid, find_segment_for_frame
from utils.merge_raw_mhr import (
    merge_raw_mhr_parts, all_parts_ready, discover_part_files, NUM_EXPECTED_PARTS,
)
from utils.model_factory import build_sam3d_body_from_config
from utils.zip_utils import unzip_if_needed, zip_and_remove_dir


# ---------------------------------------------------------------------------
# Camera intrinsics resolution
# ---------------------------------------------------------------------------

def resolve_camera_intrinsics(
    intrinsic_path: Optional[str],
    input_dir: str,
    camera_scale: float,
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    """Resolve camera intrinsics path (explicit or derived) and return ``(path, cam_int_tensor)``."""
    if intrinsic_path and os.path.isfile(intrinsic_path):
        camera_intrinsics_path = intrinsic_path
    else:
        if intrinsic_path:
            raise FileNotFoundError(f"Intrinsic path not found: {intrinsic_path}")
        p = Path(os.path.normpath(input_dir))
        parts = list(p.parts)
        out_idx = None
        for i, part in enumerate(parts):
            if part in {"outputs", "output"}:
                out_idx = i
                break
        if out_idx is None:
            raise FileNotFoundError(
                "Cannot derive camera intrinsics path (no 'outputs' or 'output' in path). Use --intrinsic-path."
            )
        dataset_root = Path(*parts[:out_idx])
        camera_intrinsics_path = str(
            dataset_root / "inputs" / "camera_params_new" / "parameters-camera-04.json"
        )

    if not os.path.isfile(camera_intrinsics_path):
        raise FileNotFoundError(f"Camera intrinsics file not found: {camera_intrinsics_path}")

    K, _ = read_camera_intrinsics_new(camera_intrinsics_path)
    K = adjust_K(K, scale=float(camera_scale))
    cam_int = torch.from_numpy(K).to(device).unsqueeze(0)
    return camera_intrinsics_path, cam_int


# ---------------------------------------------------------------------------
# Corrupted-frame detection & interpolation
# ---------------------------------------------------------------------------

def scan_corrupted_frames(
    images_list: List[str],
    masks_list: List[str],
    n: int,
) -> Tuple[set, List[int], List[str], List[str], int]:
    """Scan for 0-byte files.  Returns ``(corrupted_set, valid_indices, valid_images, valid_masks, n_valid)``."""
    corrupted_set: set = set()
    for i in range(n):
        if os.path.getsize(images_list[i]) == 0 or os.path.getsize(masks_list[i]) == 0:
            corrupted_set.add(i)
            print(f"[WARN] Corrupted frame {i}: "
                  f"image={os.path.getsize(images_list[i])}B ({os.path.basename(images_list[i])}), "
                  f"mask={os.path.getsize(masks_list[i])}B ({os.path.basename(masks_list[i])}) -> will interpolate")
    if corrupted_set:
        print(f"[INFO] {len(corrupted_set)} corrupted frames out of {n} total")

    valid_indices = [i for i in range(n) if i not in corrupted_set]
    valid_images = [images_list[i] for i in valid_indices]
    valid_masks = [masks_list[i] for i in valid_indices]
    n_valid = len(valid_images)
    if n_valid == 0:
        raise FileNotFoundError("All images/masks are corrupted — nothing to process.")
    return corrupted_set, valid_indices, valid_images, valid_masks, n_valid


def interpolate_corrupted_frames(
    valid_frames: List[Dict[str, Any]],
    valid_indices: List[int],
    images_list: List[str],
    n: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Forward-fill corrupted frame slots.  Returns ``(all_frames, num_interpolated)``."""
    all_frames: List[Optional[Dict[str, Any]]] = [None] * n
    for vi, frame_result in enumerate(valid_frames):
        all_frames[valid_indices[vi]] = frame_result

    last_valid: Optional[Dict[str, Any]] = None
    num_interpolated = 0
    for i in range(n):
        if all_frames[i] is not None:
            last_valid = all_frames[i]
        else:
            frame_name = os.path.basename(images_list[i])[:-4]
            if last_valid is not None:
                interpolated = copy.deepcopy(last_valid)
                interpolated["frame"] = frame_name
                interpolated["interpolated"] = True
                all_frames[i] = interpolated
            else:
                all_frames[i] = {"frame": frame_name, "people": [], "obj_ids": [], "interpolated": True}
            num_interpolated += 1

    frames = [f for f in all_frames if f is not None]
    if num_interpolated > 0:
        print(f"[INFO] Interpolated {num_interpolated} corrupted frames from nearest valid neighbor")
    return frames, num_interpolated


# ---------------------------------------------------------------------------
# Per-segment export & mesh-prediction summary
# ---------------------------------------------------------------------------

def save_mesh_prediction_summary(
    frames: List[Dict[str, Any]],
    segment_id_mappings: List[Dict[str, Any]],
    output_path: str,
    frame_interval: int = 200,
) -> None:
    """Save a JSON with per-frame mesh prediction info every *frame_interval* frames."""
    summary_rows = []
    for frame_data in frames:
        frame_name = frame_data["frame"]
        frame_idx = int(frame_name)
        if frame_idx % frame_interval != 0:
            continue
        people = frame_data.get("people", [])
        seg = find_segment_for_frame(frame_idx, segment_id_mappings) if segment_id_mappings else None
        
        people_in_mask = frame_data.get("people_in_mask", [])
        people_sent_to_model = frame_data.get("people_sent_to_model", [])
        excluded = frame_data.get("excluded", [])
        
        summary_rows.append({
            "frame_name": frame_name,
            "frame_idx": frame_idx,
            "segment_key": seg.get("segment_key", "") if seg else "",
            "num_people_in_mask": len(people_in_mask),
            "num_people_sent_to_model": len(people_sent_to_model),
            "num_mesh_predictions": len(people),
            "num_excluded": len(excluded),
            "interpolated": frame_data.get("interpolated", False),
            "people_in_mask": people_in_mask,
            "people_with_predictions": [
                {"tracking_id": p.get("tracking_id", -1), "real_id": p.get("obj_id", -1)}
                for p in people
            ],
            "excluded": excluded,
        })

    if summary_rows:
        write_json(output_path, {"frame_summaries": summary_rows})
        print(f"[INFO] Saved mesh prediction summary ({len(summary_rows)} frames @ {frame_interval}-frame interval) to: {output_path}")


def save_per_segment_raw_mhr(
    frames: List[Dict[str, Any]],
    segment_id_mappings: List[Dict[str, Any]],
    input_dir: str,
    cfg_path: str,
    camera_intrinsics_path: str,
    camera_scale: float,
) -> None:
    """Save one raw_mhr .pt file per segment for debugging."""
    seg_dir = os.path.join(input_dir, "raw_mhr_segments")
    os.makedirs(seg_dir, exist_ok=True)
    for seg_idx, seg in enumerate(segment_id_mappings):
        fs, fe = int(seg["frame_start"]), int(seg["frame_end"])
        seg_frames = [f for f in frames if fs <= int(f["frame"]) <= fe]
        seg_actual_ids = sorted(set(seg["consecutive_to_actual"].values()))
        seg_payload = {
            "frames": seg_frames,
            "meta": {
                "input_dir": input_dir,
                "config_path": cfg_path,
                "camera_intrinsics": camera_intrinsics_path,
                "camera_scale": camera_scale,
                "note": f"Per-segment raw params (segment {seg_idx})",
                "segment_id_mappings": [seg],
                "segment_index": seg_idx,
                "frame_range": [fs, fe],
                "actual_ids_in_segment": seg_actual_ids,
            },
        }
        seg_key = seg.get("segment_key", "")
        seg_label = f"{seg_idx}_{seg_key}" if seg_key else str(seg_idx)
        seg_path = os.path.join(seg_dir, f"raw_mhr_seg_{seg_label}.pt")
        save_raw_mhr(seg_path, seg_payload)
        print(f"[INFO] Saved segment {seg_idx} / {seg_key} ({len(seg_frames)} frames, "
              f"frames [{fs},{fe}], IDs {seg_actual_ids}) -> {seg_path}")


# ---------------------------------------------------------------------------
# Post-merge diagnostics (called after merge completes)
# ---------------------------------------------------------------------------

def _run_post_merge(input_dir: str, all_frames: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Generate summary / per-segment debug files after a merge."""
    segment_id_mappings = meta.get("segment_id_mappings") or load_segment_id_mappings(input_dir)
    save_mesh_prediction_summary(
        all_frames, segment_id_mappings,
        os.path.join(input_dir, "mesh_prediction_summary.json"),
    )
    if segment_id_mappings:
        save_per_segment_raw_mhr(
            all_frames, segment_id_mappings, input_dir,
            meta.get("config_path", ""),
            meta.get("camera_intrinsics", ""),
            float(meta.get("camera_scale", 0.5)),
        )


def _rezip_input_dirs(input_dir: str) -> None:
    """Ensure extracted images/ and masks/ are collapsed back into zip archives."""
    for dirname in ("images", "masks"):
        dir_path = os.path.join(input_dir, dirname)
        if os.path.isdir(dir_path):
            try:
                zip_and_remove_dir(dir_path)
            except Exception as e:
                print(f"[WARN] Failed to zip {dir_path}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: dump raw MHR params from masks/images.  "
                    "Supports splitting into parts (--frame-start/--frame-end) "
                    "and merging (--merge).",
    )
    parser.add_argument("--input", required=True, help="Stage1 output dir (contains images/, masks/)")
    parser.add_argument("--intrinsic-path", default=None,
                        help="Path to camera intrinsics JSON. If unset, derived from input path.")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per inference call")
    parser.add_argument("--out", default=None, help="Output .pt path (default: <input>/raw_mhr.pt)")
    parser.add_argument("--camera-scale", type=float, default=0.5)

    # Partial-run arguments
    parser.add_argument("--frame-start", type=int, default=None,
                        help="First frame index to process (inclusive). 0-based index into the sorted file list.")
    parser.add_argument("--frame-end", type=int, default=None,
                        help="Last frame index to process (exclusive). E.g. --frame-start 0 --frame-end 2400 processes frames [0,2400).")
    parser.add_argument("--part-label", default=None,
                        help="Label for this part (default: auto from frame range). "
                             "Output goes to <input>/raw_mhr_parts/raw_mhr_part_<label>.pt")
    # Merge mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge all raw_mhr_part_*.pt in <input>/raw_mhr_parts/ into raw_mhr.pt and exit.")

    args = parser.parse_args()

    # ---- Merge mode: no GPU needed ----
    if args.merge:
        _, all_frames, meta = merge_raw_mhr_parts(args.input, out_path=args.out)
        _run_post_merge(args.input, all_frames, meta)
        _rezip_input_dirs(args.input)
        return

    is_partial = args.frame_start is not None or args.frame_end is not None

    t0 = time.time()
    input_dir = args.input
    image_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")

    # Auto-extract from zip archives
    for d in (image_dir, masks_dir):
        zip_p = d + ".zip"
        if not os.path.isdir(d) and os.path.isfile(zip_p):
            unzip_if_needed(zip_p, d)
    if not os.path.isdir(image_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing images/ or masks/ in: {input_dir}")

    # Load ID mappings from Stage 1
    segment_id_mappings = load_segment_id_mappings(input_dir)
    if segment_id_mappings:
        print(f"[INFO] Loaded per-segment ID mappings ({len(segment_id_mappings)} segments)")
        for seg in segment_id_mappings:
            print(f"[INFO]   frames [{seg['frame_start']}, {seg['frame_end']}]: {seg['consecutive_to_actual']}")
    else:
        print(f"[WARN] No id_mapping.json found in {input_dir}; IDs will not be converted.")

    cfg_path = args.config or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    estimator = build_sam3d_body_from_config(cfg, device=device)
    cuda_reset_peak_memory_stats()

    os.environ["SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING"] = "1"

    camera_intrinsics_path, cam_int = resolve_camera_intrinsics(
        args.intrinsic_path, input_dir, args.camera_scale, device,
    )

    # Discover files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(image_dir, ext))])
    masks_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(masks_dir, ext))])
    n_total = min(len(images_list), len(masks_list))
    images_list, masks_list = images_list[:n_total], masks_list[:n_total]
    if n_total == 0:
        raise FileNotFoundError("Found no images or masks to process.")

    # Slice for partial runs
    fs_idx = args.frame_start if args.frame_start is not None else 0
    fe_idx = args.frame_end if args.frame_end is not None else n_total
    fs_idx = max(0, min(fs_idx, n_total))
    fe_idx = max(fs_idx, min(fe_idx, n_total))

    if is_partial:
        print(f"[INFO] Partial run: processing frames [{fs_idx}, {fe_idx}) out of {n_total} total")
        images_list = images_list[fs_idx:fe_idx]
        masks_list = masks_list[fs_idx:fe_idx]

    n = len(images_list)
    if n == 0:
        raise FileNotFoundError("Frame range is empty — nothing to process.")

    # Scan for corrupted frames
    corrupted_set, valid_indices, valid_images, valid_masks, n_valid = scan_corrupted_frames(
        images_list, masks_list, n,
    )

    # Optional: mask centroids sanity check (skip for partial runs)
    if not is_partial:
        try:
            from utils.plot_mask_centroids import compute_mask_centroids, plot_mask_centroids

            def _to_actual(cid, fidx):
                return to_actual_pid(cid, fidx, segment_id_mappings)

            centroid_data = compute_mask_centroids(masks_list, _to_actual)
            plot_mask_centroids(
                centroid_data, os.path.join(input_dir, "mask_centroids.png"),
                title_prefix="Stage 1 Mask Centroids (real IDs)",
            )
        except Exception as e:
            print(f"[WARN] Failed to plot mask centroids: {e}")

    # Batch inference
    idx_path, idx_dict, mhr_shape_scale_dict, occ_dict = {}, {}, {}, {}
    valid_frames: List[Dict[str, Any]] = []
    all_mask_ids: set = set()
    all_output_ids: set = set()
    batch_size = int(args.batch_size)

    for start in range(0, n_valid, batch_size):
        end = min(n_valid, start + batch_size)
        batch_images = valid_images[start:end]
        batch_masks = valid_masks[start:end]

        outputs, id_batch, empty_frame_list = process_image_with_mask(
            estimator, batch_images, batch_masks,
            idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, cam_int,
        )

        batch_mask_ids = set()
        for ids_in_frame in id_batch:
            if ids_in_frame:
                batch_mask_ids.update(ids_in_frame)
                all_mask_ids.update(ids_in_frame)
        if start == 0:
            print(f"[DEBUG] First batch: id_batch={len(id_batch)} frames, "
                  f"outputs={len(outputs)} frames, empty={empty_frame_list}")
            print(f"[DEBUG] First batch mask IDs: {sorted(batch_mask_ids)}")

        num_empty = 0
        for bi in range(len(batch_images)):
            frame_name = os.path.basename(batch_images[bi])[:-4]
            frame_idx = int(frame_name)

            # Load original mask to track all people
            mask_path = batch_masks[bi]
            mask = np.array(Image.open(mask_path).convert('P'))
            mask_obj_ids = sorted(np.unique(mask)[np.unique(mask) != 0].astype(int).tolist())
            people_in_mask = [
                {"tracking_id": cid, "real_id": to_actual_pid(cid, frame_idx, segment_id_mappings)}
                for cid in mask_obj_ids
            ]

            if bi in empty_frame_list:
                num_empty += 1
                excluded = [
                    {**p, "reason": "margin_filtered"} for p in people_in_mask
                ]
                valid_frames.append({
                    "frame": frame_name,
                    "people": [],
                    "obj_ids": [],
                    "people_in_mask": people_in_mask,
                    "excluded": excluded,
                })
                continue

            out_list = outputs[bi - num_empty]
            ids = id_batch[bi - num_empty]

            # Track people sent to model
            people_sent_to_model = [
                {"tracking_id": cid, "real_id": to_actual_pid(cid, frame_idx, segment_id_mappings)}
                for cid in (ids if ids is not None else [])
            ]

            people = []
            obj_ids = []
            prediction_tracking_ids = set()
            for pid, person in enumerate(out_list):
                consecutive_id = int(ids[pid]) if ids is not None and pid < len(ids) else int(pid + 1)
                actual_pid = to_actual_pid(consecutive_id, frame_idx, segment_id_mappings)
                obj_ids.append(actual_pid)
                prediction_tracking_ids.add(consecutive_id)
                people.append({
                    "obj_id": actual_pid,
                    "tracking_id": consecutive_id,
                    "global_rot": person.get("global_rot", None),
                    "body_pose": person.get("body_pose_params", None),
                    "hand": person.get("hand_pose_params", None),
                    "scale": person.get("scale_params", None),
                    "shape": person.get("shape_params", None),
                    "face": person.get("expr_params", None),
                    "pred_cam_t": person.get("pred_cam_t", None),
                    "focal_length": person.get("focal_length", None),
                })

            # Determine exclusions
            mask_tracking_ids = set(mask_obj_ids)
            sent_tracking_ids = set(ids if ids is not None else [])
            
            margin_filtered = [
                {**p, "reason": "margin_filtered"}
                for p in people_in_mask
                if p["tracking_id"] not in sent_tracking_ids
            ]
            model_failed = [
                {**p, "reason": "model_failed"}
                for p in people_sent_to_model
                if p["tracking_id"] not in prediction_tracking_ids
            ]
            excluded = margin_filtered + model_failed

            valid_frames.append({
                "frame": frame_name,
                "people": people,
                "obj_ids": obj_ids,
                "people_in_mask": people_in_mask,
                "people_sent_to_model": people_sent_to_model,
                "excluded": excluded,
            })
            all_output_ids.update(obj_ids)

    # Interpolate corrupted frames
    frames, num_interpolated = interpolate_corrupted_frames(
        valid_frames, valid_indices, images_list, n,
    )

    # ID tracking summary
    print(f"\n[DEBUG] === ID TRACKING SUMMARY ===")
    print(f"[DEBUG] Total frames: {n} (valid: {n_valid}, corrupted/interpolated: {num_interpolated})")
    print(f"[DEBUG] Unique IDs from masks: {sorted(all_mask_ids)}")
    print(f"[DEBUG] Unique IDs in output:  {sorted(all_output_ids)}")
    missing_ids = all_mask_ids - all_output_ids
    extra_ids = all_output_ids - all_mask_ids
    if missing_ids:
        print(f"[DEBUG] MISSING IDs (in masks but not output): {sorted(missing_ids)}")
    if extra_ids:
        print(f"[DEBUG] EXTRA IDs (in output but not masks): {sorted(extra_ids)}")
    print(f"[DEBUG] ==============================\n")

    if is_partial:
        # ---- Partial run: save part file only ----
        part_label = args.part_label or f"{fs_idx}_{fe_idx}"
        parts_dir = os.path.join(input_dir, "raw_mhr_parts")
        os.makedirs(parts_dir, exist_ok=True)
        part_path = args.out or os.path.join(parts_dir, f"raw_mhr_part_{part_label}.pt")
        payload = {
            "frames": frames,
            "meta": {
                "input_dir": input_dir,
                "config_path": cfg_path,
                "camera_intrinsics": camera_intrinsics_path,
                "camera_scale": float(args.camera_scale),
                "note": f"Partial raw params (frames [{fs_idx}, {fe_idx})) with SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING=1",
                "segment_id_mappings": segment_id_mappings,
                "num_corrupted_interpolated": num_interpolated,
                "frame_start": fs_idx,
                "frame_end": fe_idx,
                "part_label": part_label,
            },
        }
        save_raw_mhr(part_path, payload)
        print(f"[INFO] Saved partial raw params to: {part_path}")

        # Auto-merge when all expected parts are present
        if all_parts_ready(input_dir):
            print(f"\n[INFO] All {NUM_EXPECTED_PARTS} parts detected — auto-merging into raw_mhr.pt")
            _, merged_frames, merged_meta = merge_raw_mhr_parts(input_dir)
            _run_post_merge(input_dir, merged_frames, merged_meta)
            _rezip_input_dirs(input_dir)
        else:
            n_found = len(discover_part_files(input_dir))
            print(f"[INFO] {n_found}/{NUM_EXPECTED_PARTS} parts ready. "
                  f"Remaining parts will trigger auto-merge, or run with --merge manually.")
    else:
        # ---- Full run: save everything ----
        save_mesh_prediction_summary(
            frames, segment_id_mappings,
            os.path.join(input_dir, "mesh_prediction_summary.json"),
        )
        if segment_id_mappings:
            save_per_segment_raw_mhr(
                frames, segment_id_mappings, input_dir,
                cfg_path, camera_intrinsics_path, float(args.camera_scale),
            )

        out_path = args.out or os.path.join(input_dir, "raw_mhr.pt")
        payload = {
            "frames": frames,
            "meta": {
                "input_dir": input_dir,
                "config_path": cfg_path,
                "camera_intrinsics": camera_intrinsics_path,
                "camera_scale": float(args.camera_scale),
                "note": "Raw params dumped with SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING=1",
                "segment_id_mappings": segment_id_mappings,
                "num_corrupted_interpolated": num_interpolated,
            },
        }
        save_raw_mhr(out_path, payload)
        print(f"[INFO] Saved combined raw params to: {out_path}")
        _rezip_input_dirs(input_dir)

    mem = cuda_mem_snapshot()
    mem["wall_time_sec"] = float(time.time() - t0)
    mem_label = f"gpu_mem_stage2{'_part_' + (args.part_label or f'{fs_idx}_{fe_idx}') if is_partial else ''}.json"
    write_json(os.path.join(input_dir, mem_label), mem)
    print(f"[INFO] Peak GPU memory (stage2): {mem}")

if __name__ == "__main__":
    main()
