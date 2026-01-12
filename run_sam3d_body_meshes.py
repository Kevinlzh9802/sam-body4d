#!/usr/bin/env python3
"""
Stage 2/2: Run SAM 3D Body using *saved* masks/images (no SAM-3).

Expected inputs: an output folder produced by `run_sam3_masklets.py`, containing:
  - images/*.jpg
  - masks/*.png  (P-mode palette; pixel values are obj_ids)
  - masklets_meta.json

Outputs (in the same output folder by default):
  - mesh_4d_individual/<obj_id>/<frame>.ply
  - rendered_frames/*.jpg (optional)
  - rendered_frames_individual/<obj_id>/*.jpg (optional)
  - gpu_mem_stage2.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Make sure sam_3d_body package is importable when running from repo root
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models", "sam_3d_body"))

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import (
    process_image_with_mask,
    save_mesh_results,
)
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
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


def main():
    parser = argparse.ArgumentParser(description="Stage 2: SAM 3D Body meshes from saved masks")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Stage1 output dir (contains images/, masks/, masklets_meta.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (defaults to masklets_meta.json config_path if present)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size override (else cfg.sam_3d_body.batch_size or 1)"
    )
    parser.add_argument(
        "--camera-intrinsics",
        type=str,
        default=None,
        help="Optional camera intrinsics JSON (will be scaled by --camera-scale)",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.5,
        help="Scale factor to apply to intrinsics (matches 0.5x images in this pipeline)",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip saving rendered preview frames")
    args = parser.parse_args()

    input_dir = args.input
    meta_path = os.path.join(input_dir, "masklets_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    cfg_path = args.config or meta.get("config_path") or os.path.join("configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
    cfg = OmegaConf.load(cfg_path)

    image_dir = meta.get("image_dir") or os.path.join(input_dir, "images")
    masks_dir = meta.get("masks_dir") or os.path.join(input_dir, "masks")
    out_obj_ids = meta.get("out_obj_ids", None)

    if not os.path.isdir(image_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing images/ or masks/ in: {input_dir}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(image_dir, ext))])
    masks_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(masks_dir, ext))])
    if len(images_list) == 0 or len(masks_list) == 0:
        raise FileNotFoundError("Found no images or masks to process.")
    if len(images_list) != len(masks_list):
        print(f"[WARN] images ({len(images_list)}) != masks ({len(masks_list)}); will process min length.")

    n = min(len(images_list), len(masks_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading SAM 3D Body from config: {cfg_path}")
    estimator = build_sam3d_body_from_config(cfg, device=device)

    cuda_reset_peak_memory_stats()

    cam_int = None
    if args.camera_intrinsics:
        K, dist = read_camera_intrinsics(args.camera_intrinsics, scale=float(args.camera_scale))
        cam_int = (K, dist)

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg.sam_3d_body.get("batch_size", 1))
    )

    # Output dirs
    rendered_dir = os.path.join(input_dir, "rendered_frames")
    rendered_individual_dir = os.path.join(input_dir, "rendered_frames_individual")
    mesh_dir = os.path.join(input_dir, "mesh_4d_individual")
    os.makedirs(mesh_dir, exist_ok=True)
    if not args.no_render:
        os.makedirs(rendered_dir, exist_ok=True)
        os.makedirs(rendered_individual_dir, exist_ok=True)

    # Minimal placeholders for completion-related args expected by process_image_with_mask
    idx_path, idx_dict, mhr_shape_scale_dict, occ_dict = {}, {}, {}, {}

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_images = images_list[start:end]
        batch_masks = masks_list[start:end]

        # process_image_with_mask will compute per-object bboxes from the label mask.
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

        # Map outputs back to original batch frame indices (skip empty frames)
        num_empty = 0
        for bi in range(len(batch_images)):
            image_path = batch_images[bi]
            frame_name = os.path.basename(image_path)[:-4]

            if bi in empty_frame_list:
                num_empty += 1
                continue

            mask_output = outputs[bi - num_empty]
            id_current = id_batch[bi - num_empty]

            # Save meshes
            save_mesh_results(
                outputs=mask_output,
                faces=estimator.faces,
                save_dir=mesh_dir,
                image_path=image_path,
                id_current=id_current,
            )

            # Optional rendering previews
            if not args.no_render:
                img = cv2.imread(image_path)
                rend_img = visualize_sample_together(img, mask_output, estimator.faces, id_current)
                cv2.imwrite(os.path.join(rendered_dir, f"{frame_name}.jpg"), rend_img.astype(np.uint8))

                rend_img_list = visualize_sample(img, mask_output, estimator.faces, id_current)
                for ri, rend_img_i in enumerate(rend_img_list):
                    if id_current is not None and ri < len(id_current):
                        obj_id = int(id_current[ri])
                    else:
                        obj_id = int(ri + 1)
                    obj_dir = os.path.join(rendered_individual_dir, str(obj_id))
                    os.makedirs(obj_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(obj_dir, f"{frame_name}_{obj_id}.jpg"), rend_img_i.astype(np.uint8))

    mem = cuda_mem_snapshot()
    write_json(os.path.join(input_dir, "gpu_mem_stage2.json"), mem)
    print(f"[INFO] Peak GPU memory (stage2): {mem}")
    print(f"[INFO] Done. Meshes in: {mesh_dir}")


if __name__ == "__main__":
    main()


