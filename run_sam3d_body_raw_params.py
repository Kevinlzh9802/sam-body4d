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


def read_camera_intrinsics(intrinsic_file: str, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    with open(intrinsic_file, "r", encoding="utf-8") as f:
        intrinsic_data = json.load(f)
    K = np.array(intrinsic_data["intrinsic"], dtype=np.float32)
    dist_coeffs = np.array(intrinsic_data.get("distortion_coefficients", []), dtype=np.float32)
    return adjust_K(K, scale=scale), dist_coeffs


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

    # Camera intrinsics are loaded by mapping:
    #   <...>/outputs/<...>  ->  <...>/inputs/camera_intrinsics/intrinsic_4.json
    #
    # i.e. we locate the "outputs" directory level inside input_dir and replace the
    # remainder with the fixed intrinsics path under "inputs".
    p = Path(os.path.normpath(input_dir))
    parts = list(p.parts)
    out_idx = None
    for i, part in enumerate(parts):
        if part in {"outputs", "output"}:
            out_idx = i
            break
    if out_idx is None:
        raise FileNotFoundError(f'Cannot derive camera intrinsics path')
    dataset_root = Path(*parts[:out_idx])
    camera_intrinsics_path = str(dataset_root / "inputs" / "camera_params" / "intrinsic_4.json")
    K, dist = read_camera_intrinsics(camera_intrinsics_path, scale=float(args.camera_scale))
    cam_int = (torch.from_numpy(K).to(device), torch.from_numpy(dist).to(device))

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

        num_empty = 0
        for bi in range(len(batch_images)):
            frame_name = os.path.basename(batch_images[bi])[:-4]
            if bi in empty_frame_list:
                num_empty += 1
                frames.append({"frame": frame_name, "people": [], "obj_ids": []})
                continue

            out_list = outputs[bi - num_empty]
            ids = id_batch[bi - num_empty]
            people = []
            obj_ids = []
            for pid, person in enumerate(out_list):
                # These are numpy arrays already (process_frames converts to numpy)
                if ids is not None and pid < len(ids):
                    obj_id = int(ids[pid])
                else:
                    obj_id = int(pid + 1)
                obj_ids.append(obj_id)
                # keep only "raw param" fields we need for Stage3
                people.append(
                    {
                        "obj_id": obj_id,
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

    out_path = args.out or os.path.join(input_dir, "raw_mhr.pt")
    payload = {
        "frames": frames,
        "meta": {
            "input_dir": input_dir,
            "config_path": cfg_path,
            "camera_intrinsics": camera_intrinsics_path,
            "camera_scale": float(args.camera_scale),
            "note": "Raw params dumped with SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING=1",
        },
    }
    save_raw_mhr(out_path, payload)
    print(f"[INFO] Saved raw params to: {out_path}")

    mem = cuda_mem_snapshot()
    mem["wall_time_sec"] = float(time.time() - t0)
    write_json(os.path.join(input_dir, "gpu_mem_stage2.json"), mem)
    print(f"[INFO] Peak GPU memory (stage2): {mem}")


if __name__ == "__main__":
    main()

