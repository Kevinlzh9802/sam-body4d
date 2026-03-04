#!/usr/bin/env python3
"""
Stage 3 (decoupled): raw MHR params -> temporal smoothing -> mhr_forward -> meshes.

Input:
  - raw_mhr.pt produced by run_sam3d_body_raw_params.py

Output:
  - meshes_4d_individual/<pid>.npz  (one compressed archive per person containing
    ``vertices`` (T_vis, V, 3) float32, ``faces`` (F, 3) int32, and
    ``frame_names`` (T_vis,) string array).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sam3d_body_pkg_dir = os.path.join(REPO_DIR, "models", "sam_3d_body")
if sam3d_body_pkg_dir not in sys.path:
    sys.path.insert(0, sam3d_body_pkg_dir)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.stage3_core import (
    Stage3Config,
    stack_frames_to_tensors,
    freeze_shape_scale_first_frame,
    run_stage3_post_optimizations,
)
from smoothing.ground_plane_opt import load_extrinsics_json, GroundOptConfig
from smoothing.feet_z_plot import plot_feet_z_from_stage3, plot_body_landmarks_z
from smoothing.reproj_opt import ReprojOptConfig
from smoothing.mask_reproj_opt import MaskReprojOptConfig
from smoothing.reproj_overlay_plot import plot_reproj_overlays_from_stage3
from utils import kalman_smooth_mhr_params_per_obj_id_adaptive, ema_smooth_global_rot_per_obj_id_adaptive
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.extract_mesh_ground_info import run_extract_ground_info
from utils.id_mapping import load_segment_id_mappings_from_meta
from utils.model_factory import build_sam3d_body_from_config
from utils.plot_ground_info import run_plot_ground_info
from utils.plot_mesh_heights import plot_mesh_heights_from_stage3
from utils.zip_utils import unzip_if_needed, cleanup_extracted_dir
from scripts.diagnose_pred_cam_t import run_diagnostics as run_pred_cam_t_diagnostics


# ---------------------------------------------------------------------------
# Batched MHR forward
# ---------------------------------------------------------------------------

def run_batched_mhr_forward(
    estimator,
    mhr: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    T: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Run MHR forward in batches, return ``(verts_cpu, j3d_device, faces_np)``.

    The caller is responsible for deleting ``estimator`` afterward to free GPU memory.
    """
    head_pose = estimator.model.head_pose
    global_rot = mhr["global_rot"]
    body_pose = mhr["body_pose"]
    _hand = mhr.get("hand")
    hand = _hand if _hand is not None else torch.zeros((T * N, 108), dtype=torch.float32, device=device)
    scale = mhr["scale"]
    shape = mhr["shape"]
    _face = mhr.get("face")
    face = _face if _face is not None else torch.zeros((T * N, 72), dtype=torch.float32, device=device)

    total_samples = T * N
    print(f"[INFO] Running MHR forward in batches: {total_samples} samples, batch_size={batch_size}")

    verts_list, j3d_list = [], []
    num_batches = (total_samples + batch_size - 1) // batch_size
    for bs in tqdm(range(0, total_samples, batch_size), total=num_batches, desc="MHR forward"):
        be = min(bs + batch_size, total_samples)
        mhr_out = head_pose.mhr_forward(
            global_trans=global_rot[bs:be] * 0,
            global_rot=global_rot[bs:be],
            body_pose_params=body_pose[bs:be],
            hand_pose_params=hand[bs:be],
            scale_params=scale[bs:be],
            shape_params=shape[bs:be],
            expr_params=face[bs:be],
            return_keypoints=True,
            return_joint_coords=False,
            return_model_params=False,
            return_joint_rotations=False,
        )
        if isinstance(mhr_out, (tuple, list)) and len(mhr_out) >= 2:
            batch_verts, batch_j3d = mhr_out[0], mhr_out[1]
        else:
            raise ValueError(f"Unexpected mhr_forward output: {type(mhr_out)}")
        verts_list.append(batch_verts.cpu())
        j3d_list.append(batch_j3d.cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verts = torch.cat(verts_list, dim=0)
    j3d = torch.cat(j3d_list, dim=0).to(device)
    del verts_list, j3d_list
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[INFO] MHR forward complete: verts shape={verts.shape} (CPU), j3d shape={j3d.shape} ({j3d.device})")

    if j3d.shape[1] > 70:
        j3d = j3d[:, :70].contiguous()

    # Camera system difference
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1

    faces_np = np.asarray(estimator.faces, dtype=np.int32)
    # NOTE: do NOT `del estimator` here — it only drops the local reference.
    # The caller must delete its own reference to actually free GPU memory.
    return verts, j3d, faces_np


# ---------------------------------------------------------------------------
# Per-person mesh NPZ export
# ---------------------------------------------------------------------------

def export_per_person_meshes(
    verts: torch.Tensor,
    pred_cam_t: torch.Tensor,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    frame_names: List[str],
    faces_np: np.ndarray,
    extr,
    world_scale: float,
    mesh_dir: str,
    T: int,
    N: int,
) -> None:
    """Export one ``.npz`` per person with vertices, faces, and frame_names."""
    os.makedirs(mesh_dir, exist_ok=True)

    Rt_np = t_np = None
    if extr is not None:
        Rt_np = extr.R.transpose(0, 1).cpu().numpy()
        t_np = extr.t.cpu().numpy().reshape(1, 3)

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
            if Rt_np is not None:
                mesh_vertices = (v_cam * world_scale - t_np) @ Rt_np
            else:
                mesh_vertices = v_cam
            verts_list_oid.append(mesh_vertices)
            names_list_oid.append(frame_names[ti])

        if not verts_list_oid:
            continue
        verts_stacked = np.stack(verts_list_oid, axis=0).astype(np.float32)
        npz_path = os.path.join(mesh_dir, f"{oid}.npz")
        np.savez_compressed(npz_path, vertices=verts_stacked, faces=faces_np, frame_names=np.array(names_list_oid))
        print(f"[INFO] Saved {oid}.npz: {verts_stacked.shape[0]} frames, "
              f"{verts_stacked.shape[1]} verts, {faces_np.shape[0]} faces")
    print(f"[INFO] Exported per-person NPZ archives to: {mesh_dir}")


# ---------------------------------------------------------------------------
# Reprojection overlay generation
# ---------------------------------------------------------------------------

def generate_reproj_overlays(
    verts: torch.Tensor,
    j3d: torch.Tensor,
    pred_cam_t: torch.Tensor,
    faces_np: np.ndarray,
    camera_intrinsics_json: str,
    camera_scale: float,
    bbox_kps_pkl: Optional[str],
    payload_meta: Dict[str, Any],
    frame_names: List[str],
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    segment_id_mappings,
    out_dir: str,
    T: int,
    N: int,
) -> None:
    """Generate reprojection overlay plots if images and intrinsics are available."""
    input_dir = payload_meta.get("input_dir", "")
    images_dir = os.path.join(input_dir, "images") if input_dir else ""
    masks_dir = os.path.join(input_dir, "masks") if input_dir else ""

    for d in (images_dir, masks_dir):
        if d:
            z = d + ".zip"
            if not os.path.isdir(d) and os.path.isfile(z):
                unzip_if_needed(z, d)

    has_images = images_dir and os.path.isdir(images_dir)
    has_intrinsics = bool(camera_intrinsics_json)

    if not (has_images and has_intrinsics):
        reasons = []
        if not has_images:
            reasons.append(f"images dir not found ({images_dir!r})")
        if not has_intrinsics:
            reasons.append("--camera-intrinsics-json not provided")
        print(f"[INFO] Skipping reproj overlay: {'; '.join(reasons)}")
        return

    reproj_overlay_dir = os.path.join(out_dir, "reproj_overlay")
    try:
        K_np, _dist = read_camera_intrinsics_new(camera_intrinsics_json)
        K_np = adjust_K(K_np, scale=float(camera_scale))

        bbox_data = None
        oid_to_bidx = None
        if bbox_kps_pkl:
            from smoothing.obs_kps import load_bboxes_kps_pkl, build_obj_id_to_bbox_idx
            bbox_data = load_bboxes_kps_pkl(bbox_kps_pkl)
            oid_to_bidx = build_obj_id_to_bbox_idx(bbox_data)

        plot_reproj_overlays_from_stage3(
            verts=verts, keypoints3d_local=j3d, pred_cam_t=pred_cam_t,
            faces=faces_np, K=K_np, images_dir=images_dir,
            masks_dir=masks_dir if os.path.isdir(masks_dir) else None,
            T=T, N=N, obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            frame_names=frame_names, output_dir=reproj_overlay_dir,
            frame_interval=100, bboxes_kps_data=bbox_data,
            obj_id_to_bbox_idx=oid_to_bidx,
            segment_id_mappings=segment_id_mappings,
        )
    except Exception as e:
        import traceback
        print(f"[WARN] Failed to generate reproj overlay plots: {e}")
        traceback.print_exc()

    # Return the dirs so caller can clean up
    return images_dir, masks_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: smooth raw params and export meshes")
    parser.add_argument("--raw", required=True, help="raw_mhr.pt from run_sam3d_body_raw_params.py")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--out", default=None, help="Output dir (default: alongside raw file)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--no-option1", action="store_true", help="Disable built-in Option1 smoothing.")
    parser.add_argument("--enable-ground", action="store_true", help="Enable ground-plane/contact optimization.")
    # Mask-based reprojection (default ON)
    parser.add_argument("--no-mask-reproj", action="store_true", help="Disable mask-based reprojection optimization.")
    parser.add_argument("--mask-reproj-iters", type=int, default=200)
    parser.add_argument("--mask-reproj-lr", type=float, default=0.01)
    parser.add_argument("--mask-reproj-lambda-vertex", type=float, default=1.0,
                        help="Weight for vertex-in-mask loss (vertices should project inside mask)")
    parser.add_argument("--mask-reproj-lambda-coverage", type=float, default=0.5,
                        help="Weight for mask coverage loss (mask pixels should be near projected vertices)")
    parser.add_argument("--mask-reproj-lambda-prior", type=float, default=0.05)
    parser.add_argument("--mask-reproj-lambda-vel", type=float, default=0.5)
    parser.add_argument("--mask-reproj-lambda-accel", type=float, default=2.0)
    parser.add_argument("--mask-reproj-num-verts", type=int, default=500)
    parser.add_argument("--mask-reproj-num-mask-pts", type=int, default=200,
                        help="Number of mask points to sample for coverage loss (0 = disable)")
    # Legacy keypoint-based reprojection (default OFF)
    parser.add_argument("--enable-kps-reproj", action="store_true")
    parser.add_argument("--bbox-kps-pkl", default=None)
    parser.add_argument("--camera-intrinsics-json", default=None)
    parser.add_argument("--camera-scale", type=float, default=0.5)
    parser.add_argument("--reproj-iters", type=int, default=200)
    parser.add_argument("--reproj-lr", type=float, default=0.05)
    parser.add_argument("--reproj-huber-delta", type=float, default=10.0)
    parser.add_argument("--reproj-lambda-prior", type=float, default=0.1)
    parser.add_argument("--reproj-lambda-vel", type=float, default=1.0)
    parser.add_argument("--reproj-lambda-accel", type=float, default=0.5)
    parser.add_argument("--obs-scale-cands", nargs="*", type=float, default=[1.0, 0.5, 2.0])
    # Ground inputs
    parser.add_argument("--extrinsics-json", default=None)
    parser.add_argument("--ground-iters", type=int, default=200)
    parser.add_argument("--ground-lr", type=float, default=0.05)
    parser.add_argument("--ground-lambda-plane", type=float, default=5.0)
    parser.add_argument("--ground-lambda-slide", type=float, default=1.0)
    parser.add_argument("--ground-lambda-prior", type=float, default=0.2)
    parser.add_argument("--ground-lambda-vel", type=float, default=1.0)
    parser.add_argument("--ground-lambda-accel", type=float, default=0.5)
    parser.add_argument("--contact-z-thresh", type=float, default=3.0)
    parser.add_argument("--contact-vxy-thresh", type=float, default=5.0)
    parser.add_argument("--mhr-batch-size", type=int, default=256)
    parser.add_argument("--export-camera-space", action="store_true")
    parser.add_argument("--world-scale", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.raw)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage3_run_options.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload.get("frames", None)
    if frames is None:
        raise ValueError("Invalid raw payload: missing 'frames'.")

    meta = payload.get("meta", {})
    segment_id_mappings = load_segment_id_mappings_from_meta(meta, fallback_dir=os.path.dirname(args.raw))
    if segment_id_mappings:
        print(f"[INFO] Loaded {len(segment_id_mappings)} segment ID mapping(s)")
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
    T, N = len(frame_names), len(obj_ids_all)
    print(f"[INFO] Loaded raw: T={T}, N={N}, obj_ids={obj_ids_all[:10]}{'...' if N>10 else ''}")

    # Diagnostics: visualise raw pred_cam_t before any smoothing
    diag_dir = os.path.join(out_dir, "diagnostics")
    run_pred_cam_t_diagnostics(
        mhr=mhr, frame_obj_ids_slots=frame_obj_ids_slots,
        frame_names=frame_names, obj_ids_all=obj_ids_all,
        segment_id_mappings=segment_id_mappings, out_dir=diag_dir, label="raw",
    )

    # Option 1 smoothing
    if not args.no_option1:
        mhr = kalman_smooth_mhr_params_per_obj_id_adaptive(
            mhr_dict=mhr, num_frames=T, frame_obj_ids=frame_obj_ids_slots,
            keys_to_smooth=[k for k in ["body_pose", "hand", "pred_cam_t"] if k in mhr],
            kalman_cfg=None, vis_flags=vis_flags,
        )
        freeze_shape_scale_first_frame(mhr, T=T, N=N)
        if "global_rot" in mhr:
            mhr = ema_smooth_global_rot_per_obj_id_adaptive(
                mhr_dict=mhr, num_frames=T, frame_obj_ids=frame_obj_ids_slots,
                vis_flags=vis_flags, key_name="global_rot",
            )

    # Validate required MHR keys
    for k in ("global_rot", "body_pose", "scale", "shape", "pred_cam_t"):
        if k not in mhr:
            raise ValueError(f"Raw payload missing required key: {k}")

    # Batched MHR forward
    verts, j3d, faces_np = run_batched_mhr_forward(
        estimator, mhr, device, int(args.mhr_batch_size), T, N,
    )
    del estimator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[INFO] Released MHR model from GPU")
    pred_cam_t = mhr.get("pred_cam_t", None)

    # Post-optimizations
    enable_mask_reproj = not args.no_mask_reproj
    enable_kps_reproj = args.enable_kps_reproj
    enable_ground = args.enable_ground

    raw_parent = os.path.dirname(args.raw)
    mask_dir = os.path.join(raw_parent, "masks")
    mask_zip = mask_dir + ".zip"
    if not os.path.isdir(mask_dir) and os.path.isfile(mask_zip):
        unzip_if_needed(mask_zip, mask_dir)
    if enable_mask_reproj:
        if not os.path.isdir(mask_dir):
            print(f"[WARN] Mask directory not found: {mask_dir} — disabling mask reproj.")
            enable_mask_reproj = False
        elif not args.camera_intrinsics_json:
            print("[WARN] --camera-intrinsics-json required for mask reproj — disabling.")
            enable_mask_reproj = False

    if enable_mask_reproj or enable_kps_reproj or enable_ground:
        cfg3 = Stage3Config(
            enable_option1=(not args.no_option1),
            enable_mask_reproj=enable_mask_reproj,
            enable_kps_reproj=enable_kps_reproj,
            enable_ground=enable_ground,
            mask_dir=mask_dir if enable_mask_reproj else None,
            mask_reproj_cfg=MaskReprojOptConfig(
                iters=int(args.mask_reproj_iters), lr=float(args.mask_reproj_lr),
                lambda_vertex_in_mask=float(args.mask_reproj_lambda_vertex),
                lambda_mask_coverage=float(args.mask_reproj_lambda_coverage),
                lambda_prior=float(args.mask_reproj_lambda_prior),
                lambda_vel=float(args.mask_reproj_lambda_vel),
                lambda_accel=float(args.mask_reproj_lambda_accel),
                num_sample_vertices=int(args.mask_reproj_num_verts),
                num_sample_mask_points=int(args.mask_reproj_num_mask_pts),
            ),
            bbox_kps_pkl=args.bbox_kps_pkl,
            camera_intrinsics_json=args.camera_intrinsics_json,
            camera_scale=float(args.camera_scale),
            reproj_cfg=ReprojOptConfig(
                iters=int(args.reproj_iters), lr=float(args.reproj_lr),
                huber_delta_px=float(args.reproj_huber_delta),
                lambda_prior=float(args.reproj_lambda_prior),
                lambda_vel=float(args.reproj_lambda_vel),
                lambda_accel=float(args.reproj_lambda_accel),
                obs_scale_candidates=tuple(float(x) for x in args.obs_scale_cands),
            ),
            extrinsics_json=args.extrinsics_json,
            ground_cfg=GroundOptConfig(
                iters=int(args.ground_iters), lr=float(args.ground_lr),
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
        mhr, opt_summary = run_stage3_post_optimizations(
            cfg=cfg3, device=device, mhr=mhr, frame_names=frame_names,
            obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
            vis_flags=vis_flags, keypoints3d_local=j3d,
            vertices_local=verts if enable_mask_reproj else None,
            segment_id_mappings=segment_id_mappings,
        )
        pred_cam_t = mhr["pred_cam_t"]
        with open(os.path.join(out_dir, "stage3_opt_summary.json"), "w", encoding="utf-8") as f:
            json.dump(opt_summary, f, indent=2)

    # Extrinsics / world-space setup
    extr = None
    export_world = not args.export_camera_space
    if export_world:
        if not args.extrinsics_json:
            print("[WARN] World-space export requires --extrinsics-json. Falling back to camera space.")
            export_world = False
        else:
            extr = load_extrinsics_json(args.extrinsics_json, device=device)
            print("[INFO] Exporting meshes in WORLD coordinates")
    if not export_world:
        print("[INFO] Exporting meshes in CAMERA coordinates")
    world_scale = float(args.world_scale)

    # Export meshes
    export_per_person_meshes(
        verts=verts, pred_cam_t=pred_cam_t,
        obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
        frame_names=frame_names, faces_np=faces_np, extr=extr,
        world_scale=world_scale,
        mesh_dir=os.path.join(out_dir, "meshes_4d_individual"),
        T=T, N=N,
    )

    # Ground-plane info + plots
    ground_rows: List[Dict[str, Any]] = []
    if extr is not None:
        try:
            _pkl, ground_rows = run_extract_ground_info(
                keypoints3d_local=j3d.detach().cpu().numpy(),
                pred_cam_t=pred_cam_t.detach().cpu().numpy(),
                extr=extr, world_scale=world_scale,
                frame_names=frame_names, obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots, T=T, N=N,
                output_dir=out_dir, basename="ground_plane_info", device=device,
            )
            if _pkl:
                print(f"[INFO] Saved ground-plane info: {_pkl}")
        except Exception as e:
            print(f"[WARN] Ground-plane info extraction failed: {e}")
    if ground_rows:
        try:
            plot_dir = run_plot_ground_info(
                rows=ground_rows, frame_names=frame_names,
                output_dir=out_dir, frame_interval=200, plot_subdir="ground_plane_plots",
            )
            print(f"[INFO] Saved ground-plane plots: {plot_dir}")
        except Exception as e:
            print(f"[WARN] Ground-plane plotting failed: {e}")

    # Mesh height plot (diagnostic: should be ~160-185 cm for adults)
    if extr is not None:
        try:
            plot_mesh_heights_from_stage3(
                vertices=verts, pred_cam_t=pred_cam_t, extr=extr,
                world_scale=world_scale, T=T, N=N,
                obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                output_path=os.path.join(out_dir, "mesh_heights_world.png"),
                title="Mesh Height (World Space) — Post Stage 3",
            )
        except Exception as e:
            print(f"[WARN] Failed to generate mesh height plot: {e}")

    # Feet z-coordinate plot
    if extr is not None:
        try:
            plot_feet_z_from_stage3(
                keypoints3d_local=j3d, pred_cam_t=pred_cam_t, extr=extr,
                T=T, N=N, obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                output_path=os.path.join(out_dir, "feet_z_world.png"),
                title="Average Feet Z-Coordinate (World Space) - Post Stage 3",
                world_scale=world_scale,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate feet z-coordinate plot: {e}")

    # Head / pelvis / feet z-coordinate combined plot
    if extr is not None:
        try:
            plot_body_landmarks_z(
                keypoints3d_local=j3d, pred_cam_t=pred_cam_t, extr=extr,
                T=T, N=N, obj_ids_all=obj_ids_all,
                frame_obj_ids_slots=frame_obj_ids_slots,
                world_scale=world_scale,
                output_path=os.path.join(out_dir, "body_landmarks_z_world.png"),
                title="Body Landmark Z (World) — Post Stage 3",
            )
        except Exception as e:
            print(f"[WARN] Failed to generate body landmark z-plot: {e}")

    # Reprojection overlays
    overlay_result = generate_reproj_overlays(
        verts=verts, j3d=j3d, pred_cam_t=pred_cam_t, faces_np=faces_np,
        camera_intrinsics_json=args.camera_intrinsics_json,
        camera_scale=float(args.camera_scale),
        bbox_kps_pkl=args.bbox_kps_pkl,
        payload_meta=payload.get("meta", {}),
        frame_names=frame_names, obj_ids_all=obj_ids_all,
        frame_obj_ids_slots=frame_obj_ids_slots,
        segment_id_mappings=segment_id_mappings,
        out_dir=out_dir, T=T, N=N,
    )

    # Clean up extracted dirs
    cleanup_dirs = [mask_dir]
    if overlay_result:
        cleanup_dirs.extend(overlay_result)
    for d in dict.fromkeys(cleanup_dirs):
        if d:
            cleanup_extracted_dir(d)


if __name__ == "__main__":
    main()
