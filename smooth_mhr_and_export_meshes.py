#!/usr/bin/env python3
"""
Stage 3 (decoupled): raw MHR params -> temporal smoothing -> mhr_forward -> meshes.

Input:
  - raw_mhr.pt produced by run_sam3d_body_raw_params.py

Output:
  - meshes_4d_individual/<pid>.npz  (one compressed archive per person containing
    ``vertices`` (T_vis, V, 3) float32, ``faces`` (F, 3) int32, and
    ``frame_names`` (T_vis,) string array).

Notes:
  - This stage does NOT run the heavy image encoder. It only loads the SAM-3D-Body
    checkpoint to access the MHR forward (`head_pose.mhr_forward`) and faces.
  - Temporal smoothing is applied over the full sequence (or any window you choose).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure imports work regardless of where this script is invoked from.
import sys


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root for `models.*` imports
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
# Also allow importing the standalone `sam_3d_body` package directly if needed
sam3d_body_pkg_dir = os.path.join(REPO_DIR, "models", "sam_3d_body")
if sam3d_body_pkg_dir not in sys.path:
    sys.path.insert(0, sam3d_body_pkg_dir)

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.sam_3d_body.visualization.renderer import Renderer
from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.stage3_core import (
    Stage3Config,
    stack_frames_to_tensors,
    freeze_shape_scale_first_frame,
    run_stage3_post_optimizations,
)
from smoothing.ground_plane_opt import load_extrinsics_json
from smoothing.feet_z_plot import plot_feet_z_from_stage3
from smoothing.projection_2d_plot import plot_2d_projections_from_stage3
from smoothing.reproj_opt import ReprojOptConfig
from smoothing.ground_plane_opt import GroundOptConfig
from smoothing.mask_reproj_opt import MaskReprojOptConfig
from smoothing.reproj_overlay_plot import plot_reproj_overlays_from_stage3
from utils import kalman_smooth_mhr_params_per_obj_id_adaptive, ema_smooth_global_rot_per_obj_id_adaptive
from utils.extract_mesh_ground_info import run_extract_ground_info
from utils.plot_ground_info import run_plot_ground_info
from utils.zip_utils import unzip_if_needed, cleanup_extracted_dir


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
    parser = argparse.ArgumentParser(description="Stage 3: smooth raw params and export meshes")
    parser.add_argument("--raw", required=True, help="raw_mhr.pt from run_sam3d_body_raw_params.py")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--out", default=None, help="Output dir (default: alongside raw file)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    # Option selection
    parser.add_argument("--no-option1", action="store_true", help="Disable built-in Option1 smoothing (EMA/Kalman + shape/scale freeze).")
    parser.add_argument("--enable-ground", action="store_true", help="Enable ground-plane/contact optimization (adjust pred_cam_t in world coords).")
    
    # Mask-based reprojection (default ON) - uses Stage 1 masks
    parser.add_argument("--no-mask-reproj", action="store_true", 
                        help="Disable mask-based reprojection optimization. Default: ON (uses Stage 1 masks).")
    parser.add_argument("--mask-reproj-iters", type=int, default=150)
    parser.add_argument("--mask-reproj-lr", type=float, default=0.02)
    parser.add_argument("--mask-reproj-lambda-vertex", type=float, default=1.0,
                        help="Weight for vertex-in-mask loss")
    parser.add_argument("--mask-reproj-lambda-coverage", type=float, default=0.5,
                        help="Weight for mask coverage loss")
    parser.add_argument("--mask-reproj-lambda-prior", type=float, default=0.1)
    parser.add_argument("--mask-reproj-lambda-vel", type=float, default=1.0,
                        help="Weight for velocity smoothness (1st order)")
    parser.add_argument("--mask-reproj-lambda-accel", type=float, default=0.5,
                        help="Weight for acceleration smoothness (2nd order, reduces jitter)")
    parser.add_argument("--mask-reproj-num-verts", type=int, default=500,
                        help="Number of vertices to sample for mask reproj (0 = use all)")
    
    # Legacy keypoint-based reprojection (default OFF) - requires external bbox/kps pkl
    parser.add_argument("--enable-kps-reproj", action="store_true", 
                        help="Enable legacy keypoint-based reprojection (requires --bbox-kps-pkl). Default: OFF.")
    parser.add_argument("--bbox-kps-pkl", default=None, help="bboxes_kps_data pickle (observed 2D kps with kp_idx in mhr70).")
    parser.add_argument("--camera-intrinsics-json", default=None, help="Camera intrinsics json (scaled by --camera-scale).")
    parser.add_argument("--camera-scale", type=float, default=0.5)
    parser.add_argument("--reproj-iters", type=int, default=200)
    parser.add_argument("--reproj-lr", type=float, default=0.05)
    parser.add_argument("--reproj-huber-delta", type=float, default=10.0)
    parser.add_argument("--reproj-lambda-prior", type=float, default=0.1)
    parser.add_argument("--reproj-lambda-vel", type=float, default=1.0)
    parser.add_argument("--reproj-lambda-accel", type=float, default=0.5)
    parser.add_argument("--obs-scale-cands", nargs="*", type=float, default=[1.0, 0.5, 2.0])

    # Ground inputs/knobs
    parser.add_argument("--extrinsics-json", default=None, help="Extrinsics json (OpenCV solvePnP: rotation, translation).")
    parser.add_argument("--ground-iters", type=int, default=200)
    parser.add_argument("--ground-lr", type=float, default=0.05)
    parser.add_argument("--ground-lambda-plane", type=float, default=5.0)
    parser.add_argument("--ground-lambda-slide", type=float, default=1.0)
    parser.add_argument("--ground-lambda-prior", type=float, default=0.2)
    parser.add_argument("--ground-lambda-vel", type=float, default=1.0)
    parser.add_argument("--ground-lambda-accel", type=float, default=0.5,
                        help="Acceleration smoothness for ground optimization (reduces jitter)")
    parser.add_argument("--contact-z-thresh", type=float, default=3.0,
                        help="Contact detection threshold for foot Z height in world units (default: 3.0 cm)")
    parser.add_argument("--contact-vxy-thresh", type=float, default=5.0,
                        help="Contact detection threshold for foot XY velocity in world units (default: 5.0 cm/frame)")
    parser.add_argument("--mhr-batch-size", type=int, default=256, help="Batch size for MHR forward pass (reduce if OOM)")
    parser.add_argument("--export-camera-space", action="store_true", 
                        help="Export meshes in camera coordinates instead of world coordinates. "
                             "By default, meshes are exported in world space (requires --extrinsics-json) "
                             "where the ground plane is at z=0.")
    parser.add_argument("--world-scale", type=float, default=100.0,
                        help="Scale factor to apply to mesh coordinates before world-space transformation. "
                             "Default 100.0 converts SMPL-X meters to centimeters (use if extrinsics are in cm). "
                             "Set to 1.0 if extrinsics are already in meters.")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.raw)
    os.makedirs(out_dir, exist_ok=True)
    # Log run options for reproducibility
    opts_path = os.path.join(out_dir, "stage3_run_options.txt")
    with open(opts_path, "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload.get("frames", None)
    if frames is None:
        raise ValueError("Invalid raw payload: missing 'frames'.")

    # Load per-segment ID mappings (consecutive mask pixel ID <-> actual PID) from raw_mhr metadata.
    # Also try loading from id_mapping.json alongside raw_mhr.pt as fallback.
    meta = payload.get("meta", {})
    segment_id_mappings: Optional[List[Dict[str, Any]]] = meta.get("segment_id_mappings", None)
    if not segment_id_mappings:
        id_mapping_path = os.path.join(os.path.dirname(args.raw), "id_mapping.json")
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, "r", encoding="utf-8") as _f:
                _id_data = json.load(_f)
            if "segments" in _id_data:
                segment_id_mappings = [
                    {"frame_start": int(s["frame_start"]), "frame_end": int(s["frame_end"]),
                     "consecutive_to_actual": {int(k): int(v) for k, v in s["consecutive_to_actual"].items()}}
                    for s in _id_data["segments"]
                ]
            elif "consecutive_to_actual" in _id_data:
                segment_id_mappings = [{
                    "frame_start": 0, "frame_end": 999_999_999,
                    "consecutive_to_actual": {int(k): int(v) for k, v in _id_data["consecutive_to_actual"].items()},
                }]
    if segment_id_mappings:
        print(f"[INFO] Loaded {len(segment_id_mappings)} segment ID mapping(s) for mask reproj")
    else:
        print("[INFO] No segment ID mappings found; mask reproj will use obj_id as mask pixel ID directly")

    cfg_path = args.config or meta.get("config_path") or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    estimator = build_sam3d_body_from_config(cfg, device=device)

    mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all = stack_frames_to_tensors(frames, device=device)
    T = len(frame_names)
    N = len(obj_ids_all)
    print(f"[INFO] Loaded raw: T={T}, N={N}, obj_ids={obj_ids_all[:10]}{'...' if N>10 else ''}")

    # --- Option 1 smoothing over full sequence ---
    if not args.no_option1:
        mhr = kalman_smooth_mhr_params_per_obj_id_adaptive(
            mhr_dict=mhr,
            num_frames=T,
            frame_obj_ids=frame_obj_ids_slots,
            keys_to_smooth=[k for k in ["body_pose", "hand", "pred_cam_t"] if k in mhr],
            kalman_cfg=None,
            vis_flags=vis_flags,
        )
        freeze_shape_scale_first_frame(mhr, T=T, N=N)
        if "global_rot" in mhr:
            mhr = ema_smooth_global_rot_per_obj_id_adaptive(
                mhr_dict=mhr,
                num_frames=T,
                frame_obj_ids=frame_obj_ids_slots,
                vis_flags=vis_flags,
                key_name="global_rot",
            )

    # --- Recompute meshes via MHR forward ---
    head_pose = estimator.model.head_pose
    global_rot = mhr.get("global_rot", None)
    body_pose = mhr.get("body_pose", None)
    hand = mhr.get("hand", None)
    scale = mhr.get("scale", None)
    shape = mhr.get("shape", None)
    face = mhr.get("face", None)
    pred_cam_t = mhr.get("pred_cam_t", None)
    focal = mhr.get("focal_length", None)

    if global_rot is None or body_pose is None or scale is None or shape is None or pred_cam_t is None:
        raise ValueError("Raw payload missing required keys (need global_rot, body_pose, scale, shape, pred_cam_t).")

    if hand is None:
        # allow hand missing; head_pose expects hand_pose_params, can pass zeros with right dim if needed
        hand = torch.zeros((T * N, 108), dtype=torch.float32, device=device)
    if face is None:
        face = torch.zeros((T * N, 72), dtype=torch.float32, device=device)

    # --- Batched MHR forward to avoid OOM ---
    # Total samples = T * N (e.g., 1201 frames × 17 people = 20,417)
    # Process in smaller batches to fit in GPU memory
    total_samples = T * N
    mhr_batch_size = int(args.mhr_batch_size)
    print(f"[INFO] Running MHR forward in batches: {total_samples} samples, batch_size={mhr_batch_size}")
    
    verts_list = []
    j3d_list = []
    
    num_batches = (total_samples + mhr_batch_size - 1) // mhr_batch_size
    for batch_start in tqdm(range(0, total_samples, mhr_batch_size), total=num_batches, desc="MHR forward"):
        batch_end = min(batch_start + mhr_batch_size, total_samples)
        
        mhr_out = head_pose.mhr_forward(
            global_trans=global_rot[batch_start:batch_end] * 0,
            global_rot=global_rot[batch_start:batch_end],
            body_pose_params=body_pose[batch_start:batch_end],
            hand_pose_params=hand[batch_start:batch_end],
            scale_params=scale[batch_start:batch_end],
            shape_params=shape[batch_start:batch_end],
            expr_params=face[batch_start:batch_end],
            return_keypoints=True,
            return_joint_coords=False,
            return_model_params=False,
            return_joint_rotations=False,
        )
        # mhr_forward signature can vary across checkpoints/versions.
        # Expect at least (verts, j3d); ignore any extra returns.
        if isinstance(mhr_out, (tuple, list)) and len(mhr_out) >= 2:
            batch_verts, batch_j3d = mhr_out[0], mhr_out[1]
        else:
            raise ValueError(f"Unexpected mhr_forward output: {type(mhr_out)}")
        
        # Move to CPU immediately to free GPU memory
        verts_list.append(batch_verts.cpu())
        j3d_list.append(batch_j3d.cpu())
        
        # Clear cache periodically
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Concatenate all batches — verts stays on CPU (~18 GB for large sequences), j3d is small enough for GPU
    verts = torch.cat(verts_list, dim=0)          # CPU — (T*N, V, 3)
    j3d = torch.cat(j3d_list, dim=0).to(device)   # GPU — (T*N, 70, 3), ~115 MB
    del verts_list, j3d_list
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"[INFO] MHR forward complete: verts shape={verts.shape} (CPU), j3d shape={j3d.shape} ({j3d.device})")
    
    # MHR can return 308 keypoints; clip to 70 (body-only) to match the rest of the pipeline
    if j3d.shape[1] > 70:
        j3d = j3d[:, :70].contiguous()
    
    # Camera system difference (match existing pipeline)
    verts[..., [1, 2]] *= -1   # CPU — no GPU memory
    j3d[..., [1, 2]] *= -1

    # Save faces before freeing model, then release GPU memory occupied by model weights (~1-3 GB)
    faces_np = np.asarray(estimator.faces, dtype=np.int32)
    del estimator, head_pose
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[INFO] Released MHR model from GPU")

    # Determine which optimizations to run
    enable_mask_reproj = not args.no_mask_reproj
    enable_kps_reproj = args.enable_kps_reproj
    enable_ground = args.enable_ground

    # Auto-detect mask directory from raw_mhr.pt path: <exp>/masklets/raw_mhr.pt -> <exp>/masklets/masks
    raw_parent = os.path.dirname(args.raw)  # e.g., <exp>/masklets
    mask_dir = os.path.join(raw_parent, "masks")

    # Auto-extract masks from zip if the directory doesn't exist yet
    mask_zip = mask_dir + ".zip"
    if not os.path.isdir(mask_dir) and os.path.isfile(mask_zip):
        unzip_if_needed(mask_zip, mask_dir)

    # Validate required inputs for mask reproj
    if enable_mask_reproj:
        if not os.path.isdir(mask_dir):
            print(f"[WARN] Mask directory not found: {mask_dir}")
            print("[WARN] Disabling mask-based reprojection optimization.")
            enable_mask_reproj = False
        elif not args.camera_intrinsics_json:
            print("[WARN] --camera-intrinsics-json is required for mask-based reprojection.")
            print("[WARN] Disabling mask-based reprojection optimization.")
            enable_mask_reproj = False

    # Run post-optimizations if any enabled
    if enable_mask_reproj or enable_kps_reproj or enable_ground:
        cfg3 = Stage3Config(
            enable_option1=(not args.no_option1),
            enable_mask_reproj=enable_mask_reproj,
            enable_kps_reproj=enable_kps_reproj,
            enable_ground=enable_ground,
            # Mask reproj config
            mask_dir=mask_dir if enable_mask_reproj else None,
            mask_reproj_cfg=MaskReprojOptConfig(
                iters=int(args.mask_reproj_iters),
                lr=float(args.mask_reproj_lr),
                lambda_vertex_in_mask=float(args.mask_reproj_lambda_vertex),
                lambda_mask_coverage=float(args.mask_reproj_lambda_coverage),
                lambda_prior=float(args.mask_reproj_lambda_prior),
                lambda_vel=float(args.mask_reproj_lambda_vel),
                lambda_accel=float(args.mask_reproj_lambda_accel),
                num_sample_vertices=int(args.mask_reproj_num_verts),
            ),
            # Legacy kps reproj config
            bbox_kps_pkl=args.bbox_kps_pkl,
            camera_intrinsics_json=args.camera_intrinsics_json,
            camera_scale=float(args.camera_scale),
            reproj_cfg=ReprojOptConfig(
                iters=int(args.reproj_iters),
                lr=float(args.reproj_lr),
                huber_delta_px=float(args.reproj_huber_delta),
                lambda_prior=float(args.reproj_lambda_prior),
                lambda_vel=float(args.reproj_lambda_vel),
                lambda_accel=float(args.reproj_lambda_accel),
                obs_scale_candidates=tuple(float(x) for x in args.obs_scale_cands),
            ),
            # Ground config
            extrinsics_json=args.extrinsics_json,
            ground_cfg=GroundOptConfig(
                iters=int(args.ground_iters),
                lr=float(args.ground_lr),
                lambda_plane=float(args.ground_lambda_plane),
                lambda_slide=float(args.ground_lambda_slide),
                lambda_prior=float(args.ground_lambda_prior),
                lambda_vel=float(args.ground_lambda_vel),
                lambda_accel=float(args.ground_lambda_accel),
                contact_z_thresh=float(args.contact_z_thresh),
                contact_v_xy_thresh=float(args.contact_vxy_thresh),
                world_scale=float(args.world_scale),
            ),
        )

        # keypoints3d_local: (T*N,70,3) without translation; that's `j3d` here.
        # vertices_local: (T*N,V,3) mesh vertices for mask reproj
        mhr, opt_summary = run_stage3_post_optimizations(
            cfg=cfg3,
            device=device,
            mhr=mhr,
            frame_names=frame_names,
            obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            vis_flags=vis_flags,
            keypoints3d_local=j3d,
            vertices_local=verts if enable_mask_reproj else None,
            segment_id_mappings=segment_id_mappings,
        )

        # Apply updated pred_cam_t to verts for export (do not change verts topology)
        pred_cam_t = mhr["pred_cam_t"]
        out_dir = args.out or os.path.dirname(args.raw)
        with open(os.path.join(out_dir, "stage3_opt_summary.json"), "w", encoding="utf-8") as f:
            json.dump(opt_summary, f, indent=2)

    out_dir = args.out or os.path.dirname(args.raw)
    mesh_dir = os.path.join(out_dir, "meshes_4d_individual")
    os.makedirs(mesh_dir, exist_ok=True)

    # Load extrinsics for world-space export (default) unless --export-camera-space is set
    extr = None
    export_world = not args.export_camera_space
    if export_world:
        if not args.extrinsics_json:
            print("[WARN] World-space export requires --extrinsics-json. Falling back to camera space.")
            export_world = False
        else:
            extr = load_extrinsics_json(args.extrinsics_json, device=device)
            print("[INFO] Exporting meshes in WORLD coordinates (ground plane at z=0)")
    if not export_world:
        print("[INFO] Exporting meshes in CAMERA coordinates")
    
    world_scale = float(args.world_scale)
    if export_world and world_scale != 1.0:
        print(f"[INFO] Applying world scale factor: {world_scale} (mesh coordinates will be scaled before world transform)")

    # Export per-person NPZ: one compressed archive per obj_id containing
    # vertices (T_visible, V, 3), faces (F, 3), and frame_names (T_visible,).
    # Precompute extrinsics as CPU numpy (avoids GPU round-trip per sample).
    if extr is not None:
        Rt_np = extr.R.transpose(0, 1).cpu().numpy()   # (3,3)
        t_np = extr.t.cpu().numpy().reshape(1, 3)       # (1,3)

    for oid in obj_ids_all:
        si = list(obj_ids_all).index(oid)
        verts_list_oid: List[np.ndarray] = []
        names_list_oid: List[str] = []
        for ti in range(T):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            v = verts[bi].detach().float().numpy()
            camt = pred_cam_t[bi].detach().float().cpu().numpy()
            v_cam = v + camt
            if extr is not None:
                v_cam_scaled = v_cam * world_scale
                mesh_vertices = (v_cam_scaled - t_np) @ Rt_np
            else:
                mesh_vertices = v_cam
            verts_list_oid.append(mesh_vertices)
            names_list_oid.append(frame_names[ti])

        if not verts_list_oid:
            continue
        verts_stacked = np.stack(verts_list_oid, axis=0).astype(np.float32)
        npz_path = os.path.join(mesh_dir, f"{oid}.npz")
        np.savez_compressed(
            npz_path,
            vertices=verts_stacked,
            faces=faces_np,
            frame_names=np.array(names_list_oid),
        )
        print(f"[INFO] Saved {oid}.npz: {verts_stacked.shape[0]} frames, "
              f"{verts_stacked.shape[1]} verts, {faces_np.shape[0]} faces")

    print(f"[INFO] Exported per-person NPZ archives to: {mesh_dir}")

    # Extract 2D ground-plane info (positions + orientations) from keypoints; save one pkl (+ csv) per output folder (world space only)
    ground_rows: List[Dict[str, Any]] = []
    if extr is not None:
        try:
            _pkl, ground_rows = run_extract_ground_info(
                keypoints3d_local=j3d.detach().cpu().numpy(),
                pred_cam_t=pred_cam_t.detach().cpu().numpy(),
                extr=extr,
                world_scale=world_scale,
                frame_names=frame_names,
                obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                T=T,
                N=N,
                output_dir=out_dir,
                basename="ground_plane_info",
                device=device,
            )
            if _pkl:
                print(f"[INFO] Saved ground-plane info: {_pkl}")
        except Exception as e:
            print(f"[WARN] Ground-plane info extraction failed: {e}")

    # Plot ground-plane positions and orientations every 200 frames (before zipping)
    if ground_rows:
        try:
            plot_dir = run_plot_ground_info(
                rows=ground_rows,
                frame_names=frame_names,
                output_dir=out_dir,
                frame_interval=200,
                plot_subdir="ground_plane_plots",
            )
            print(f"[INFO] Saved ground-plane plots: {plot_dir}")
        except Exception as e:
            print(f"[WARN] Ground-plane plotting failed: {e}")

    # Plot feet z-coordinates in world space (only if extrinsics available)
    if extr is not None:
        feet_z_plot_path = os.path.join(out_dir, "feet_z_world.png")
        try:
            plot_feet_z_from_stage3(
                keypoints3d_local=j3d,
                pred_cam_t=pred_cam_t,
                extr=extr,
                T=T,
                N=N,
                obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                output_path=feet_z_plot_path,
                title="Average Feet Z-Coordinate (World Space) - Post Stage 3",
                world_scale=world_scale,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate feet z-coordinate plot: {e}")
        
        # Plot 2D projections (bird's eye view) every 100 frames
        projection_2d_dir = os.path.join(out_dir, "projection_2d")
        try:
            plot_2d_projections_from_stage3(
                keypoints3d_local=j3d,
                pred_cam_t=pred_cam_t,
                extr=extr,
                T=T,
                N=N,
                obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                output_dir=projection_2d_dir,
                frame_interval=100,
                world_scale=world_scale,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate 2D projection plots: {e}")

    # --- Reprojection overlay: projected mesh + keypoints on image + mask --- #
    # Requires camera intrinsics + images directory from the raw payload.
    meta = payload.get("meta", {})
    input_dir = meta.get("input_dir", "")
    images_dir = os.path.join(input_dir, "images") if input_dir else ""
    masks_dir = os.path.join(input_dir, "masks") if input_dir else ""

    # Auto-extract from zip archives if directories don't exist yet
    for d in (images_dir, masks_dir):
        if d:
            z = d + ".zip"
            if not os.path.isdir(d) and os.path.isfile(z):
                unzip_if_needed(z, d)

    has_images = images_dir and os.path.isdir(images_dir)
    has_intrinsics = bool(args.camera_intrinsics_json)

    if has_images and has_intrinsics:
        from smoothing.stage3_core import read_camera_intrinsics_new, adjust_K as adjust_K_fn

        reproj_overlay_dir = os.path.join(out_dir, "reproj_overlay")
        try:
            K_np, _dist = read_camera_intrinsics_new(args.camera_intrinsics_json)
            K_np = adjust_K_fn(K_np, scale=float(args.camera_scale))

            # Load observed 2D keypoints if available
            bbox_data = None
            oid_to_bidx = None
            if args.bbox_kps_pkl:
                from smoothing.obs_kps import load_bboxes_kps_pkl, build_obj_id_to_bbox_idx
                bbox_data = load_bboxes_kps_pkl(args.bbox_kps_pkl)
                oid_to_bidx = build_obj_id_to_bbox_idx(bbox_data)

            plot_reproj_overlays_from_stage3(
                verts=verts,
                keypoints3d_local=j3d,
                pred_cam_t=pred_cam_t,
                faces=faces_np,
                K=K_np,
                images_dir=images_dir,
                masks_dir=masks_dir if os.path.isdir(masks_dir) else None,
                T=T,
                N=N,
                obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                frame_names=frame_names,
                output_dir=reproj_overlay_dir,
                frame_interval=100,
                bboxes_kps_data=bbox_data,
                obj_id_to_bbox_idx=oid_to_bidx,
            )
        except Exception as e:
            import traceback
            print(f"[WARN] Failed to generate reproj overlay plots: {e}")
            traceback.print_exc()
    else:
        reasons = []
        if not has_images:
            reasons.append(f"images dir not found ({images_dir!r})")
        if not has_intrinsics:
            reasons.append("--camera-intrinsics-json not provided")
        print(f"[INFO] Skipping reproj overlay: {'; '.join(reasons)}")

    # Clean up extracted images/masks (only if the zip still exists on disk)
    for d in dict.fromkeys([mask_dir, images_dir, masks_dir]):
        if d:
            cleanup_extracted_dir(d)


if __name__ == "__main__":
    main()

