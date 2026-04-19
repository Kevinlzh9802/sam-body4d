#!/usr/bin/env python3
"""
Simple mesh placement: mask centroids + fixed world-height assumption → meshes.

No optimization.  For each person in each frame, the 2D mask centroid is
unprojected to a camera ray, and the depth along the ray is chosen so that the
pelvis world-z equals a fixed assumed height (default 1 m above ground).
This gives a ``pred_cam_t`` that is then used to place the MHR mesh.

Inputs:
  - raw_mhr.pt from Stage 2
  - Stage 1 masks directory
  - Camera intrinsics + extrinsics
  - For fisheye cameras, pass ``--centroid-ray-model fisheye`` so mask
    centroids are converted to rays with ``cv2.fisheye.undistortPoints``.

Outputs (same layout as smooth_mhr_and_export_meshes.py):
  - meshes_4d_individual/<pid>.npz
  - ground_plane_plots/
  - reproj_overlay/
  - feet_z_world.png
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
from PIL import Image
from tqdm import tqdm

import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sam3d_body_pkg_dir = os.path.join(REPO_DIR, "models", "sam_3d_body")
if sam3d_body_pkg_dir not in sys.path:
    sys.path.insert(0, sam3d_body_pkg_dir)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.stage3_core import stack_frames_to_tensors, freeze_shape_scale_first_frame
from smoothing.ground_plane_opt import load_extrinsics_json, Extrinsics
from smoothing.feet_z_plot import plot_feet_z_from_stage3, plot_body_landmarks_z
from smoothing.reproj_overlay_plot import plot_reproj_overlays_from_stage3
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.extract_mesh_ground_info import run_extract_ground_info
from utils.id_mapping import load_segment_id_mappings_from_meta
from utils.model_factory import build_sam3d_body_from_config
from utils.plot_ground_info import run_plot_ground_info
from utils.plot_mesh_heights import plot_mesh_heights_from_stage3
from utils.zip_utils import unzip_if_needed, cleanup_extracted_dir


# ---------------------------------------------------------------------------
# Mask centroid computation
# ---------------------------------------------------------------------------

def compute_mask_centroids(
    mask_dir: str,
    frame_names: List[str],
    segment_id_mappings: Optional[List[Dict[str, Any]]],
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """
    Compute 2D pixel centroid (cx, cy) per person per frame from mask PNGs.

    Returns ``{actual_pid: {frame_idx: (cx, cy)}}``.
    """
    # Build per-frame consecutive→actual mapping
    T = len(frame_names)
    c2a_per_frame: List[Dict[int, int]] = [{} for _ in range(T)]
    if segment_id_mappings:
        for seg in segment_id_mappings:
            c2a = {int(k): int(v) for k, v in seg["consecutive_to_actual"].items()}
            fs, fe = int(seg["frame_start"]), int(seg["frame_end"])
            for t in range(max(0, fs), min(T, fe + 1)):
                c2a_per_frame[t] = c2a

    centroids: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for ti, fname in enumerate(tqdm(frame_names, desc="Mask centroids")):
        mask_path = os.path.join(mask_dir, f"{fname}.png")
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        for cid in np.unique(mask):
            if cid == 0:
                continue
            ys, xs = np.where(mask == cid)
            cx, cy = float(xs.mean()), float(ys.mean())
            actual_pid = c2a_per_frame[ti].get(int(cid), int(cid))
            centroids.setdefault(actual_pid, {})[ti] = (cx, cy)
    return centroids


# ---------------------------------------------------------------------------
# Analytical pred_cam_t from 2D centroids + world height
# ---------------------------------------------------------------------------

def compute_pred_cam_t_from_centroids(
    centroids: Dict[int, Dict[int, Tuple[float, float]]],
    j3d_local: torch.Tensor,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    extr: Extrinsics,
    world_scale: float,
    assumed_height_world: float,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    T: int,
    N: int,
    hip_indices: Tuple[int, int] = (9, 10),
    centroid_ray_model: str = "pinhole",
) -> torch.Tensor:
    """
    For each (person, frame) compute ``pred_cam_t`` such that:
    - the hip midpoint projects to the mask centroid in the image, and
    - the hip midpoint world z-coordinate equals ``assumed_height_world``.

    MHR70 has no explicit pelvis joint; we use the midpoint of left_hip (9)
    and right_hip (10) as a proxy.

    Falls back to zero ``pred_cam_t`` when no centroid is available.
    """
    device = j3d_local.device
    pred_cam_t = torch.zeros(T * N, 3, device=device)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx_k, cy_k = float(K[0, 2]), float(K[1, 2])

    Rt = extr.R.transpose(0, 1).cpu().numpy()          # (3,3) = R^T
    t_extr = extr.t.cpu().numpy().reshape(3)            # (3,)
    Rt_col2 = Rt[:, 2]                                  # (3,)

    n_placed = 0
    for si, oid in enumerate(obj_ids_all):
        oid_centroids = centroids.get(oid, {})
        for ti in range(T):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            if ti not in oid_centroids:
                continue
            cx_px, cy_px = oid_centroids[ti]
            # Hip midpoint as pelvis proxy (MHR70 joints 9=left_hip, 10=right_hip)
            pelvis_local = j3d_local[bi, list(hip_indices)].cpu().numpy().mean(axis=0)  # (3,)

            # Unnormalised camera ray (z=1). For fisheye/standard distortion,
            # OpenCV undistortPoints returns normalized image coordinates.
            r = camera_ray_from_pixel(
                cx_px,
                cy_px,
                K,
                dist_coeffs,
                model=centroid_ray_model,
                pinhole_params=(fx, fy, cx_k, cy_k),
            )

            # Solve for depth z along ray such that world z == assumed_height_world
            # X_world = (z * r * world_scale - t) @ R^T
            # X_world[2] = z * ws * (r @ Rt_col2) - t @ Rt_col2
            denom = world_scale * float(r @ Rt_col2)
            if abs(denom) < 1e-8:
                continue
            z = (assumed_height_world + float(t_extr @ Rt_col2)) / denom
            if z <= 0.05:
                continue

            pelvis_cam = (z * r).astype(np.float32)
            t_new = pelvis_cam - pelvis_local
            pred_cam_t[bi] = torch.from_numpy(t_new).to(device)
            n_placed += 1

    print(f"[INFO] Placed {n_placed} person-frames via centroid + height assumption")
    return pred_cam_t


def camera_ray_from_pixel(
    x_px: float,
    y_px: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    model: str,
    pinhole_params: Tuple[float, float, float, float],
) -> np.ndarray:
    """Return a camera ray with z=1 from a distorted or undistorted pixel."""
    model = str(model).lower().strip()
    fx, fy, cx_k, cy_k = pinhole_params
    if model == "pinhole":
        return np.array([(x_px - cx_k) / fx, (y_px - cy_k) / fy, 1.0], dtype=np.float64)

    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dist.size == 0:
        raise ValueError(f"--centroid-ray-model {model!r} requires distortion coefficients.")

    import cv2  # type: ignore

    pixel = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
    K64 = np.asarray(K, dtype=np.float64)
    if model == "fisheye":
        if dist.size < 4:
            dist = np.pad(dist, (0, 4 - dist.size), mode="constant")
        undist = cv2.fisheye.undistortPoints(pixel, K64, dist[:4].reshape(4, 1))
    elif model == "standard":
        undist = cv2.undistortPoints(pixel, K64, dist)
    else:
        raise ValueError(
            f"Unknown centroid ray model {model!r}; expected 'pinhole', 'standard', or 'fisheye'."
        )
    x_n, y_n = undist[0, 0]
    return np.array([float(x_n), float(y_n), 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Batched MHR forward (same as smooth_mhr_and_export_meshes.py)
# ---------------------------------------------------------------------------

def run_batched_mhr_forward(estimator, mhr, device, batch_size, T, N):
    head_pose = estimator.model.head_pose
    global_rot = mhr["global_rot"]
    body_pose = mhr["body_pose"]
    _hand = mhr.get("hand")
    hand = _hand if _hand is not None else torch.zeros((T * N, 108), dtype=torch.float32, device=device)
    scale = mhr["scale"]
    shape = mhr["shape"]
    _face = mhr.get("face")
    face = _face if _face is not None else torch.zeros((T * N, 72), dtype=torch.float32, device=device)

    total = T * N
    verts_list, j3d_list = [], []
    for bs in tqdm(range(0, total, batch_size), desc="MHR forward"):
        be = min(bs + batch_size, total)
        out = head_pose.mhr_forward(
            global_trans=global_rot[bs:be] * 0, global_rot=global_rot[bs:be],
            body_pose_params=body_pose[bs:be], hand_pose_params=hand[bs:be],
            scale_params=scale[bs:be], shape_params=shape[bs:be],
            expr_params=face[bs:be], return_keypoints=True,
            return_joint_coords=False, return_model_params=False, return_joint_rotations=False,
        )
        verts_list.append(out[0].cpu())
        j3d_list.append(out[1].cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verts = torch.cat(verts_list, dim=0)
    j3d = torch.cat(j3d_list, dim=0)
    del verts_list, j3d_list
    if j3d.shape[1] > 70:
        j3d = j3d[:, :70].contiguous()
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1
    faces_np = np.asarray(estimator.faces, dtype=np.int32)
    return verts, j3d, faces_np


# ---------------------------------------------------------------------------
# Per-person NPZ export (same as smooth_mhr_and_export_meshes.py)
# ---------------------------------------------------------------------------

def export_per_person_meshes(verts, pred_cam_t, obj_ids_all, frame_obj_ids_slots,
                             frame_names, faces_np, extr, world_scale, mesh_dir, T, N):
    os.makedirs(mesh_dir, exist_ok=True)
    Rt_np = t_np = None
    if extr is not None:
        Rt_np = extr.R.transpose(0, 1).cpu().numpy()
        t_np = extr.t.cpu().numpy().reshape(1, 3)

    for oid in obj_ids_all:
        si = list(obj_ids_all).index(oid)
        vl, nl = [], []
        for ti in range(T):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            v = verts[bi].detach().float().numpy()
            camt = pred_cam_t[bi].detach().float().cpu().numpy()
            v_cam = v + camt
            if Rt_np is not None:
                v_world = (v_cam * world_scale - t_np) @ Rt_np
            else:
                v_world = v_cam
            vl.append(v_world)
            nl.append(frame_names[ti])
        if not vl:
            continue
        vs = np.stack(vl, axis=0).astype(np.float32)
        npz_path = os.path.join(mesh_dir, f"{oid}.npz")
        np.savez_compressed(npz_path, vertices=vs, faces=faces_np, frame_names=np.array(nl))
        print(f"[INFO] {oid}.npz: {vs.shape[0]} frames, {vs.shape[1]} verts")
    print(f"[INFO] Exported meshes to: {mesh_dir}")


# ---------------------------------------------------------------------------
# Reprojection overlay (reuses the shared module)
# ---------------------------------------------------------------------------

def generate_reproj_overlays(verts, j3d, pred_cam_t, faces_np,
                             camera_intrinsics_json, camera_scale,
                             payload_meta, frame_names, obj_ids_all,
                             frame_obj_ids_slots, segment_id_mappings,
                             out_dir, T, N):
    input_dir = payload_meta.get("input_dir", "")
    images_dir = os.path.join(input_dir, "images") if input_dir else ""
    masks_dir = os.path.join(input_dir, "masks") if input_dir else ""
    for d in (images_dir, masks_dir):
        if d:
            z = d + ".zip"
            if not os.path.isdir(d) and os.path.isfile(z):
                unzip_if_needed(z, d)

    if not (images_dir and os.path.isdir(images_dir) and camera_intrinsics_json):
        print("[INFO] Skipping reproj overlay (missing images or intrinsics)")
        return images_dir, masks_dir

    try:
        K_np, _ = read_camera_intrinsics_new(camera_intrinsics_json)
        K_np = adjust_K(K_np, scale=float(camera_scale))
        plot_reproj_overlays_from_stage3(
            verts=verts, keypoints3d_local=j3d, pred_cam_t=pred_cam_t,
            faces=faces_np, K=K_np, images_dir=images_dir,
            masks_dir=masks_dir if os.path.isdir(masks_dir) else None,
            T=T, N=N, obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            frame_names=frame_names,
            output_dir=os.path.join(out_dir, "reproj_overlay"),
            frame_interval=100, segment_id_mappings=segment_id_mappings,
        )
    except Exception as e:
        import traceback
        print(f"[WARN] Reproj overlay failed: {e}")
        traceback.print_exc()
    return images_dir, masks_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple mesh placement from mask centroids + fixed height assumption",
    )
    parser.add_argument("--raw", required=True, help="raw_mhr.pt from Stage 2")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--out", default=None, help="Output dir (default: alongside raw file)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera-intrinsics-json", required=True)
    parser.add_argument("--camera-scale", type=float, default=0.5)
    parser.add_argument("--extrinsics-json", required=True)
    parser.add_argument("--world-scale", type=float, default=100.0,
                        help="Scale from SMPL-X metres to extrinsic units (default 100 = cm)")
    parser.add_argument("--assumed-height", type=float, default=1.0,
                        help="Assumed pelvis height above ground in metres (default 1.0)")
    parser.add_argument("--centroid-ray-model", choices=["pinhole", "standard", "fisheye"], default="pinhole",
                        help="How to convert mask-centroid pixels to camera rays "
                             "(pinhole ignores distortion; fisheye uses cv2.fisheye.undistortPoints)")
    parser.add_argument("--mhr-batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.raw)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "simple_run_options.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # ---- Load raw_mhr.pt --------------------------------------------------
    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload.get("frames")
    if frames is None:
        raise ValueError("Invalid raw payload: missing 'frames'.")
    meta = payload.get("meta", {})

    segment_id_mappings = load_segment_id_mappings_from_meta(meta, fallback_dir=os.path.dirname(args.raw))
    if segment_id_mappings:
        print(f"[INFO] Loaded {len(segment_id_mappings)} segment ID mapping(s)")

    cfg_path = args.config or meta.get("config_path") or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # ---- Stack frames, build model, MHR forward ---------------------------
    mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all = stack_frames_to_tensors(frames, device=device)
    T, N = len(frame_names), len(obj_ids_all)
    print(f"[INFO] T={T}, N={N}, obj_ids={obj_ids_all[:10]}{'...' if N > 10 else ''}")

    freeze_shape_scale_first_frame(mhr, T=T, N=N)

    estimator = build_sam3d_body_from_config(cfg, device=device)
    verts, j3d, faces_np = run_batched_mhr_forward(estimator, mhr, device, int(args.mhr_batch_size), T, N)
    del estimator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[INFO] Released MHR model from GPU")

    # ---- Load camera / extrinsics -----------------------------------------
    K_np, dist_np = read_camera_intrinsics_new(args.camera_intrinsics_json)
    K_np = adjust_K(K_np, scale=float(args.camera_scale))
    extr = load_extrinsics_json(args.extrinsics_json, device=torch.device("cpu"))
    world_scale = float(args.world_scale)
    assumed_height_world = float(args.assumed_height) * world_scale
    print(f"[INFO] Assumed pelvis height: {args.assumed_height} m = {assumed_height_world} world units")
    print(f"[INFO] Centroid ray model: {args.centroid_ray_model}")

    # ---- Compute mask centroids -------------------------------------------
    mask_dir = os.path.join(os.path.dirname(args.raw), "masks")
    mask_zip = mask_dir + ".zip"
    if not os.path.isdir(mask_dir) and os.path.isfile(mask_zip):
        unzip_if_needed(mask_zip, mask_dir)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    centroids = compute_mask_centroids(mask_dir, frame_names, segment_id_mappings)
    print(f"[INFO] Centroids for {len(centroids)} person(s)")

    # ---- Compute pred_cam_t analytically ----------------------------------
    pred_cam_t = compute_pred_cam_t_from_centroids(
        centroids=centroids, j3d_local=j3d, K=K_np, dist_coeffs=dist_np, extr=extr,
        world_scale=world_scale, assumed_height_world=assumed_height_world,
        obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
        T=T, N=N, centroid_ray_model=args.centroid_ray_model,
    )

    # Move j3d to device for downstream plotting (pred_cam_t stays CPU-friendly)
    j3d = j3d.to(device)
    extr_device = load_extrinsics_json(args.extrinsics_json, device=device)

    # ---- Export NPZ meshes ------------------------------------------------
    export_per_person_meshes(
        verts=verts, pred_cam_t=pred_cam_t,
        obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
        frame_names=frame_names, faces_np=faces_np, extr=extr,
        world_scale=world_scale,
        mesh_dir=os.path.join(out_dir, "meshes_4d_individual"),
        T=T, N=N,
    )

    # ---- Ground-plane info + plots ----------------------------------------
    ground_rows: List[Dict[str, Any]] = []
    try:
        _pkl, ground_rows = run_extract_ground_info(
            keypoints3d_local=j3d.detach().cpu().numpy(),
            pred_cam_t=pred_cam_t.detach().cpu().numpy(),
            extr=extr_device, world_scale=world_scale,
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

    # ---- Mesh height plot ---------------------------------------------------
    try:
        plot_mesh_heights_from_stage3(
            vertices=verts, pred_cam_t=pred_cam_t, extr=extr,
            world_scale=world_scale, T=T, N=N,
            obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            output_path=os.path.join(out_dir, "mesh_heights_world.png"),
            title="Mesh Height (World Space) — simple centroid placement",
        )
    except Exception as e:
        print(f"[WARN] Failed to generate mesh height plot: {e}")

    # ---- Feet z-coordinate plot -------------------------------------------
    try:
        plot_feet_z_from_stage3(
            keypoints3d_local=j3d, pred_cam_t=pred_cam_t.to(device), extr=extr_device,
            T=T, N=N, obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            output_path=os.path.join(out_dir, "feet_z_world.png"),
            title="Feet Z (World) — simple centroid placement",
            world_scale=world_scale,
        )
    except Exception as e:
        print(f"[WARN] Feet z-coordinate plot failed: {e}")

    # ---- Head / pelvis / feet z-coordinate combined plot ------------------
    try:
        plot_body_landmarks_z(
            keypoints3d_local=j3d, pred_cam_t=pred_cam_t.to(device), extr=extr_device,
            T=T, N=N, obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            world_scale=world_scale,
            output_path=os.path.join(out_dir, "body_landmarks_z_world.png"),
            title="Body Landmark Z (World) — simple centroid placement",
        )
    except Exception as e:
        print(f"[WARN] Body landmark z-plot failed: {e}")

    # ---- Reprojection overlays --------------------------------------------
    overlay_result = generate_reproj_overlays(
        verts=verts, j3d=j3d, pred_cam_t=pred_cam_t.to(device), faces_np=faces_np,
        camera_intrinsics_json=args.camera_intrinsics_json,
        camera_scale=float(args.camera_scale),
        payload_meta=meta, frame_names=frame_names,
        obj_ids_all=obj_ids_all, frame_obj_ids_slots=frame_obj_ids_slots,
        segment_id_mappings=segment_id_mappings,
        out_dir=out_dir, T=T, N=N,
    )

    # ---- Cleanup ----------------------------------------------------------
    cleanup_dirs = [mask_dir]
    if overlay_result:
        cleanup_dirs.extend(overlay_result)
    for d in dict.fromkeys(cleanup_dirs):
        if d:
            cleanup_extracted_dir(d)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
