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
    if not os.path.isdir(image_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing images/ or masks/ in: {input_dir}")

    # Load ID mapping from Stage 1 (consecutive -> actual PIDs)
    id_mapping_path = os.path.join(input_dir, "id_mapping.json")
    consecutive_to_actual: Dict[int, int] = {}
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            id_mapping = json.load(f)
        consecutive_to_actual = {int(k): int(v) for k, v in id_mapping.get("consecutive_to_actual", {}).items()}
        print(f"[INFO] Loaded ID mapping from: {id_mapping_path}")
        print(f"[INFO] ID mapping (consecutive -> actual): {consecutive_to_actual}")
    else:
        print(f"[WARN] No id_mapping.json found in {input_dir}; IDs will not be converted.")
    
    def to_actual_pid(consecutive_id: int) -> int:
        """Convert consecutive ID to actual PID using the mapping."""
        if consecutive_to_actual and int(consecutive_id) in consecutive_to_actual:
            return consecutive_to_actual[int(consecutive_id)]
        return consecutive_id  # fallback to original ID

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

    # Minimal placeholders expected by process_image_with_mask
    idx_path, idx_dict, mhr_shape_scale_dict, occ_dict = {}, {}, {}, {}

    # We accumulate "mhr" tensors by re-running on the whole clip, then re-pack.
    # We use the model output stored inside estimator.model during inference; the
    # easiest stable artifact is to run per chunk and collect per-frame per-person
    # param dicts, then stack into tensors at the end.
    #
    # For minimal risk, we dump the per-frame per-person params as a list of frames.
    frames: List[Dict[str, Any]] = []

    # DEBUG: Track all IDs seen from masks vs IDs in final output
    all_mask_ids: set = set()
    all_output_ids: set = set()
    id_mismatch_frames: List[Dict[str, Any]] = []

    for start in range(0, n, int(args.batch_size)):
        end = min(n, start + int(args.batch_size))
        batch_images = images_list[start:end]
        batch_masks = masks_list[start:end]

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

        # DEBUG: Log batch-level info
        batch_mask_ids = set()
        for ids_in_frame in id_batch:
            if ids_in_frame:
                batch_mask_ids.update(ids_in_frame)
                all_mask_ids.update(ids_in_frame)
        if start == 0:
            print(f"[DEBUG] First batch: id_batch has {len(id_batch)} frames, "
                  f"outputs has {len(outputs)} frames, empty_frame_list={empty_frame_list}")
            print(f"[DEBUG] First batch mask IDs: {sorted(batch_mask_ids)}")

        num_empty = 0
        for bi in range(len(batch_images)):
            frame_name = os.path.basename(batch_images[bi])[:-4]
            if bi in empty_frame_list:
                num_empty += 1
                frames.append({"frame": frame_name, "people": [], "obj_ids": []})
                continue

            out_list = outputs[bi - num_empty]
            ids = id_batch[bi - num_empty]
            
            # DEBUG: Check for mismatch between model outputs and mask IDs
            if ids is not None and len(out_list) != len(ids):
                mismatch_info = {
                    "frame": frame_name,
                    "mask_ids": list(ids) if ids else [],
                    "num_model_outputs": len(out_list),
                    "num_mask_ids": len(ids) if ids else 0,
                }
                id_mismatch_frames.append(mismatch_info)
                if len(id_mismatch_frames) <= 5:  # Log first 5 mismatches
                    print(f"[DEBUG] ID MISMATCH in {frame_name}: "
                          f"mask has {len(ids)} IDs {ids}, model output has {len(out_list)} people")
            
            people = []
            obj_ids = []
            for pid, person in enumerate(out_list):
                # These are numpy arrays already (process_frames converts to numpy)
                # Get consecutive ID from mask, then convert to actual PID
                if ids is not None and pid < len(ids):
                    consecutive_id = int(ids[pid])
                else:
                    consecutive_id = int(pid + 1)
                actual_pid = to_actual_pid(consecutive_id)
                obj_ids.append(actual_pid)
                # keep only "raw param" fields we need for Stage3
                people.append(
                    {
                        "obj_id": actual_pid,  # Store actual PID
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

            frames.append({"frame": frame_name, "people": people, "obj_ids": obj_ids})
            all_output_ids.update(obj_ids)

    # DEBUG: Print summary of ID tracking
    print(f"\n[DEBUG] === ID TRACKING SUMMARY ===")
    print(f"[DEBUG] Total unique IDs from masks (id_batch): {len(all_mask_ids)}")
    print(f"[DEBUG] Mask IDs: {sorted(all_mask_ids)}")
    print(f"[DEBUG] Total unique IDs in output: {len(all_output_ids)}")
    print(f"[DEBUG] Output IDs: {sorted(all_output_ids)}")
    missing_ids = all_mask_ids - all_output_ids
    extra_ids = all_output_ids - all_mask_ids
    if missing_ids:
        print(f"[DEBUG] MISSING IDs (in masks but not output): {sorted(missing_ids)}")
    if extra_ids:
        print(f"[DEBUG] EXTRA IDs (in output but not masks): {sorted(extra_ids)}")
    print(f"[DEBUG] Frames with ID count mismatch: {len(id_mismatch_frames)}")
    if id_mismatch_frames:
        print(f"[DEBUG] First mismatch details: {id_mismatch_frames[0]}")
    print(f"[DEBUG] ==============================\n")

    out_path = args.out or os.path.join(input_dir, "raw_mhr.pt")
    payload = {
        "frames": frames,
        "meta": {
            "input_dir": input_dir,
            "config_path": cfg_path,
            "camera_intrinsics": camera_intrinsics_path,
            "camera_scale": float(args.camera_scale),
            "note": "Raw params dumped with SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING=1",
            "consecutive_to_actual": consecutive_to_actual,
        },
    }
    save_raw_mhr(out_path, payload)
    print(f"[INFO] Saved raw params to: {out_path}")
    if consecutive_to_actual:
        print(f"[INFO] IDs converted from consecutive to actual PIDs using mapping.")

    mem = cuda_mem_snapshot()
    mem["wall_time_sec"] = float(time.time() - t0)
    write_json(os.path.join(input_dir, "gpu_mem_stage2.json"), mem)
    print(f"[INFO] Peak GPU memory (stage2): {mem}")


if __name__ == "__main__":
    main()

