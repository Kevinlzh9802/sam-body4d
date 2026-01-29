#!/usr/bin/env python3
"""
Stage 3 (decoupled): raw MHR params -> temporal smoothing -> mhr_forward -> meshes.

Input:
  - raw_mhr.pt produced by run_sam3d_body_raw_params.py

Output:
  - mesh_4d_individual/<obj_id>/<frame>.ply in an output folder

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
from smoothing.reproj_opt import ReprojOptConfig
from smoothing.ground_plane_opt import GroundOptConfig
from utils import kalman_smooth_mhr_params_per_obj_id_adaptive, ema_smooth_global_rot_per_obj_id_adaptive


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
    parser.add_argument("--enable-reproj", action="store_true", help="Enable Option2 robust 2D reprojection optimization (adjust pred_cam_t).")
    parser.add_argument("--enable-ground", action="store_true", help="Enable ground-plane/contact optimization (adjust pred_cam_t in world coords).")

    # Reprojection inputs/knobs
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
    parser.add_argument("--contact-z-thresh", type=float, default=0.03)
    parser.add_argument("--contact-vxy-thresh", type=float, default=0.05)
    args = parser.parse_args()

    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload.get("frames", None)
    if frames is None:
        raise ValueError("Invalid raw payload: missing 'frames'.")

    cfg_path = args.config or payload.get("meta", {}).get("config_path") or os.path.join(REPO_DIR, "configs", "body4d.yaml")
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

    mhr_out = head_pose.mhr_forward(
        global_trans=global_rot * 0,
        global_rot=global_rot,
        body_pose_params=body_pose,
        hand_pose_params=hand,
        scale_params=scale,
        shape_params=shape,
        expr_params=face,
        return_keypoints=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
    )
    # mhr_forward signature can vary across checkpoints/versions.
    # Expect at least (verts, j3d); ignore any extra returns.
    if isinstance(mhr_out, (tuple, list)) and len(mhr_out) >= 2:
        verts, j3d = mhr_out[0], mhr_out[1]
    else:
        raise ValueError(f"Unexpected mhr_forward output: {type(mhr_out)}")
    # Camera system difference (match existing pipeline)
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1

    # Option2 / ground: operate on pred_cam_t, using keypoints3d before translation.
    if args.enable_reproj or args.enable_ground:
        cfg3 = Stage3Config(
            enable_option1=(not args.no_option1),
            enable_reproj=bool(args.enable_reproj),
            enable_ground=bool(args.enable_ground),
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
            extrinsics_json=args.extrinsics_json,
            ground_cfg=GroundOptConfig(
                iters=int(args.ground_iters),
                lr=float(args.ground_lr),
                lambda_plane=float(args.ground_lambda_plane),
                lambda_slide=float(args.ground_lambda_slide),
                lambda_prior=float(args.ground_lambda_prior),
                lambda_vel=float(args.ground_lambda_vel),
                contact_z_thresh=float(args.contact_z_thresh),
                contact_v_xy_thresh=float(args.contact_vxy_thresh),
            ),
        )

        # keypoints3d_local: (T*N,70,3) without translation; that's `j3d` here.
        mhr, opt_summary = run_stage3_post_optimizations(
            cfg=cfg3,
            device=device,
            mhr=mhr,
            frame_names=frame_names,
            obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            vis_flags=vis_flags,
            keypoints3d_local=j3d,
        )

        # Apply updated pred_cam_t to verts for export (do not change verts topology)
        pred_cam_t = mhr["pred_cam_t"]
        out_dir = args.out or os.path.join(os.path.dirname(args.raw), "smoothed_export")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "stage3_opt_summary.json"), "w", encoding="utf-8") as f:
            json.dump(opt_summary, f, indent=2)

    out_dir = args.out or os.path.join(os.path.dirname(args.raw), "smoothed_export")
    mesh_dir = os.path.join(out_dir, "mesh_4d_individual")
    os.makedirs(mesh_dir, exist_ok=True)

    # Export per frame/per obj_id (only when present)
    faces_np = estimator.faces
    for ti in range(T):
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            v = verts[bi].detach().float().cpu().numpy()
            camt = pred_cam_t[bi].detach().float().cpu().numpy()
            fl = float(focal[bi, 0].detach().cpu().item()) if focal is not None else 1000.0
            renderer = Renderer(focal_length=fl, faces=faces_np)
            mesh = renderer.vertices_to_trimesh(v, camt, mesh_base_color=(0.65, 0.74, 0.86))
            obj_out_dir = os.path.join(mesh_dir, str(oid))
            os.makedirs(obj_out_dir, exist_ok=True)
            mesh.export(os.path.join(obj_out_dir, f"{frame_names[ti]}.ply"))

    print(f"[INFO] Exported smoothed meshes to: {mesh_dir}")


if __name__ == "__main__":
    main()

