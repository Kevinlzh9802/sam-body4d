#!/usr/bin/env python3
"""
Stage 2 (decoupled): masks/images -> raw MHR params (no temporal smoothing).

This preserves the original pipeline (`run_sam3d_body_meshes.py`) by adding a new script.

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
from utils.proj_utils import read_camera_intrinsics

# Ensure sam_3d_body package importable when running from repo root
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_DIR, "models", "sam_3d_body"))

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import save_raw_mhr
from utils.gpu_profiler import cuda_mem_snapshot, cuda_reset_peak_memory_stats, write_json
from utils.zip_utils import unzip_if_needed, zip_and_remove_dir


def adjust_K(K: np.ndarray, scale: float) -> np.ndarray:
    return np.array(
        [
            [K[0, 0] * scale, 0, K[0, 2] * scale],
            [0, K[1, 1] * scale, K[1, 2] * scale],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def read_camera_intrinsics_new(intrinsic_file: str):
    with open(intrinsic_file, "r") as f:
        intrinsic_data = json.load(f)
        params = intrinsic_data['Calibration']['cameras'][0]['model']['ptr_wrapper']['data']['parameters']

        f = params['f']['val']
        cx = params['cx']['val']
        cy = params['cy']['val']
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        ks = [params[f'k{i}']['val'] for i in range(1, 5)]
        dist_coeffs = np.array(ks)

    return K, dist_coeffs

def build_sam3d_body_from_config(cfg, device: torch.device) -> SAM3DBodyEstimator:
    mhr_path = cfg.sam_3d_body.get("mhr_path", "")
    fov_path = cfg.sam_3d_body.get("fov_path", "")
    model, model_cfg = load_sam_3d_body(cfg.sam_3d_body["ckpt_path"], device=device, mhr_path=mhr_path)

    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

    fov_estimator = None
    if fov_path:
        fov_estimator = FOVEstimator(name="moge2", device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    return estimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: dump raw MHR params from masks/images")
    parser.add_argument("--input", required=True, help="Stage1 output dir (contains images/, masks/)")
    parser.add_argument(
        "--intrinsic-path",
        default=None,
        help="Path to camera intrinsics JSON (e.g. parameters-camera-04.json). If unset, derived from input path.",
    )
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per inference call (throughput only)")
    parser.add_argument("--out", default=None, help="Output .pt path (default: <input>/raw_mhr.pt)")
    parser.add_argument("--camera-scale", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.time()

    input_dir = args.input
    image_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")

    # Auto-extract from zip archives produced by Stage 1
    for d in (image_dir, masks_dir):
        zip_p = d + ".zip"
        if not os.path.isdir(d) and os.path.isfile(zip_p):
            unzip_if_needed(zip_p, d)

    if not os.path.isdir(image_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing images/ or masks/ in: {input_dir}")

    # Load ID mapping from Stage 1 (per-segment or legacy global)
    id_mapping_path = os.path.join(input_dir, "id_mapping.json")
    segment_id_mappings: List[Dict[str, Any]] = []
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            id_mapping_data = json.load(f)
        if "segments" in id_mapping_data:
            for seg in id_mapping_data["segments"]:
                segment_id_mappings.append({
                    "frame_start": int(seg["frame_start"]),
                    "frame_end": int(seg["frame_end"]),
                    "consecutive_to_actual": {int(k): int(v) for k, v in seg["consecutive_to_actual"].items()},
                })
            print(f"[INFO] Loaded per-segment ID mappings ({len(segment_id_mappings)} segments)")
            for seg in segment_id_mappings:
                print(f"[INFO]   frames [{seg['frame_start']}, {seg['frame_end']}]: {seg['consecutive_to_actual']}")
        elif "consecutive_to_actual" in id_mapping_data:
            # Legacy global format — wrap as a single segment spanning all frames
            global_mapping = {int(k): int(v) for k, v in id_mapping_data["consecutive_to_actual"].items()}
            segment_id_mappings.append({
                "frame_start": 0,
                "frame_end": 999_999_999,
                "consecutive_to_actual": global_mapping,
            })
            print(f"[INFO] Loaded legacy global ID mapping: {global_mapping}")
    else:
        print(f"[WARN] No id_mapping.json found in {input_dir}; IDs will not be converted.")

    def to_actual_pid(consecutive_id: int, frame_idx: int) -> int:
        """Convert consecutive ID to actual PID using the per-segment mapping."""
        for seg in segment_id_mappings:
            if seg["frame_start"] <= frame_idx <= seg["frame_end"]:
                mapping = seg["consecutive_to_actual"]
                if int(consecutive_id) in mapping:
                    return mapping[int(consecutive_id)]
                break
        return consecutive_id

    cfg_path = args.config or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    estimator = build_sam3d_body_from_config(cfg, device=device)
    cuda_reset_peak_memory_stats()

    # Disable temporal smoothing inside the model to produce "raw" params.
    # This preserves original pipeline defaults (env var unset).
    os.environ["SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING"] = "1"

    # Camera intrinsics: use --intrinsic-path if set; otherwise derive from input path.
    if args.intrinsic_path and os.path.isfile(args.intrinsic_path):
        camera_intrinsics_path = args.intrinsic_path
    else:
        if args.intrinsic_path:
            raise FileNotFoundError(f"Intrinsic path not found: {args.intrinsic_path}")
        # Derive: <...>/outputs/<...> -> <...>/inputs/camera_params_new/parameters-camera-04.json
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
    K = adjust_K(K, scale=float(args.camera_scale))
    # K, _ = read_camera_intrinsics(camera_intrinsics_path, scale=float(args.camera_scale))
    # SAM-3D-Body expects batched intrinsics: shape [B, 3, 3] (not [3, 3]).
    # We use a single K shared across frames, so B=1 here; the estimator will
    # concat per-frame batches into [num_frames, 3, 3] internally.
    cam_int = torch.from_numpy(K).to(device).unsqueeze(0)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(image_dir, ext))])
    masks_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(masks_dir, ext))])
    n = min(len(images_list), len(masks_list))
    images_list = images_list[:n]
    masks_list = masks_list[:n]
    if n == 0:
        raise FileNotFoundError("Found no images or masks to process.")

    # --------------- Pre-scan for corrupted (0-byte) files ---------------
    corrupted_set: set = set()
    for i in range(n):
        img_size = os.path.getsize(images_list[i])
        msk_size = os.path.getsize(masks_list[i])
        if img_size == 0 or msk_size == 0:
            corrupted_set.add(i)
            print(f"[WARN] Corrupted frame {i}: "
                  f"image={img_size}B ({os.path.basename(images_list[i])}), "
                  f"mask={msk_size}B ({os.path.basename(masks_list[i])}) -> will interpolate")

    if corrupted_set:
        print(f"[INFO] {len(corrupted_set)} corrupted frames out of {n} total")

    valid_indices = [i for i in range(n) if i not in corrupted_set]
    valid_images = [images_list[i] for i in valid_indices]
    valid_masks = [masks_list[i] for i in valid_indices]
    n_valid = len(valid_images)

    if n_valid == 0:
        raise FileNotFoundError("All images/masks are corrupted — nothing to process.")

    # Plot mask centroids (x, y over frames) as a quick sanity check for Stage 1 + annotations
    try:
        from utils.plot_mask_centroids import compute_mask_centroids, plot_mask_centroids
        centroid_data = compute_mask_centroids(masks_list, to_actual_pid)
        centroid_plot_path = os.path.join(input_dir, "mask_centroids.png")
        plot_mask_centroids(centroid_data, centroid_plot_path,
                           title_prefix="Stage 1 Mask Centroids (real IDs)")
    except Exception as e:
        print(f"[WARN] Failed to plot mask centroids: {e}")

    # Minimal placeholders expected by process_image_with_mask
    idx_path, idx_dict, mhr_shape_scale_dict, occ_dict = {}, {}, {}, {}

    # Collect per-frame results for valid (non-corrupted) frames only.
    valid_frames: List[Dict[str, Any]] = []

    all_mask_ids: set = set()
    all_output_ids: set = set()
    batch_size = int(args.batch_size)

    for start in range(0, n_valid, batch_size):
        end = min(n_valid, start + batch_size)
        batch_images = valid_images[start:end]
        batch_masks = valid_masks[start:end]

        outputs, id_batch, empty_frame_list = process_image_with_mask(
            estimator,
            batch_images,
            batch_masks,
            idx_path,
            idx_dict,
            mhr_shape_scale_dict,
            occ_dict,
            cam_int,
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

            if bi in empty_frame_list:
                num_empty += 1
                valid_frames.append({"frame": frame_name, "people": [], "obj_ids": []})
                continue

            out_list = outputs[bi - num_empty]
            ids = id_batch[bi - num_empty]

            people = []
            obj_ids = []
            for pid, person in enumerate(out_list):
                if ids is not None and pid < len(ids):
                    consecutive_id = int(ids[pid])
                else:
                    consecutive_id = int(pid + 1)
                actual_pid = to_actual_pid(consecutive_id, frame_idx)
                obj_ids.append(actual_pid)
                people.append(
                    {
                        "obj_id": actual_pid,
                        "global_rot": person.get("global_rot", None),
                        "body_pose": person.get("body_pose_params", None),
                        "hand": person.get("hand_pose_params", None),
                        "scale": person.get("scale_params", None),
                        "shape": person.get("shape_params", None),
                        "face": person.get("expr_params", None),
                        "pred_cam_t": person.get("pred_cam_t", None),
                        "focal_length": person.get("focal_length", None),
                    }
                )

            valid_frames.append({"frame": frame_name, "people": people, "obj_ids": obj_ids})
            all_output_ids.update(obj_ids)

    # --------------- Fill corrupted frames by interpolation ---------------
    # Place valid results at their original positions, then fill gaps with
    # the most recent valid frame's output (forward-fill).
    all_frames: List[Optional[Dict[str, Any]]] = [None] * n
    for vi, frame_result in enumerate(valid_frames):
        all_frames[valid_indices[vi]] = frame_result

    last_valid_result: Optional[Dict[str, Any]] = None
    num_interpolated = 0
    for i in range(n):
        if all_frames[i] is not None:
            last_valid_result = all_frames[i]
        else:
            frame_name = os.path.basename(images_list[i])[:-4]
            if last_valid_result is not None:
                interpolated = copy.deepcopy(last_valid_result)
                interpolated["frame"] = frame_name
                interpolated["interpolated"] = True
                all_frames[i] = interpolated
            else:
                all_frames[i] = {
                    "frame": frame_name,
                    "people": [],
                    "obj_ids": [],
                    "interpolated": True,
                }
            num_interpolated += 1

    frames: List[Dict[str, Any]] = [f for f in all_frames if f is not None]

    if num_interpolated > 0:
        print(f"[INFO] Interpolated {num_interpolated} corrupted frames from nearest valid neighbor")

    # Print summary
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

    # --- Save per-segment raw_mhr files for easier debugging ---
    if segment_id_mappings:
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
                    "camera_scale": float(args.camera_scale),
                    "note": f"Per-segment raw params (segment {seg_idx})",
                    "segment_id_mappings": [seg],
                    "segment_index": seg_idx,
                    "frame_range": [fs, fe],
                    "actual_ids_in_segment": seg_actual_ids,
                },
            }
            seg_path = os.path.join(seg_dir, f"raw_mhr_seg_{seg_idx}.pt")
            save_raw_mhr(seg_path, seg_payload)
            print(f"[INFO] Saved segment {seg_idx} ({len(seg_frames)} frames, "
                  f"frames [{fs},{fe}], IDs {seg_actual_ids}) -> {seg_path}")

    # --- Save combined raw_mhr.pt (all segments concatenated) ---
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
    if segment_id_mappings:
        print(f"[INFO] IDs converted using per-segment mappings ({len(segment_id_mappings)} segments).")

    mem = cuda_mem_snapshot()
    mem["wall_time_sec"] = float(time.time() - t0)
    write_json(os.path.join(input_dir, "gpu_mem_stage2.json"), mem)
    print(f"[INFO] Peak GPU memory (stage2): {mem}")

    # Re-zip images/ and masks/ to reduce inode count for the cluster
    for d in (image_dir, masks_dir):
        if os.path.isdir(d):
            try:
                zip_and_remove_dir(d)
            except Exception as e:
                print(f"[WARN] Failed to zip {d}: {e}")


if __name__ == "__main__":
    main()

