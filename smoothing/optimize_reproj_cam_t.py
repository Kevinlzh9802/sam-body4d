#!/usr/bin/env python3
"""
Option 2: Per-person per-sequence optimization with robust 2D keypoint reprojection.

This script is a *post-processing* step for stage2 output folders (masks/images),
and it does NOT modify `run_sam3d_body_meshes.py`.

Pipeline:
  1) Load images/ and masks/ from a stage2 folder.
  2) Run SAM-3D-Body (mask-conditioned) to get per-frame predictions:
     - pred_vertices (V, 3)
     - pred_keypoints_3d (70, 3)
     - pred_cam_t (3,)
     - focal_length / cam_int
  3) For each tracked person (obj_id), optimize a translation time-series t_t (3,)
     that minimizes:
        robust reprojection loss to observed 2D keypoints
      + temporal smoothness (velocity + optional acceleration)
      + prior to stay close to initial pred_cam_t
  4) Export corrected meshes (vertices + optimized t_t) into a new output folder.

Observed keypoints format (from your bboxes_kps_data):
  bboxes_kps_data[frame]["kps"] is (N, K, 3) where last dim is (x, y, kp_idx),
  and kp_idx refers to the mhr70 indexing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

# Ensure sam_3d_body package importable when running from repo root
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_DIR, "models", "sam_3d_body"))

from models.sam_3d_body.notebook.utils import process_image_with_mask
from models.sam_3d_body.sam_3d_body.visualization.renderer import Renderer
from utils.camera_utils import adjust_K, read_camera_intrinsics
from utils.model_factory import build_sam3d_body_from_config


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    # x: (...,) >= 0
    d = float(delta)
    return torch.where(x <= d, 0.5 * x * x, d * (x - 0.5 * d))


def _project_points(K: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    K: (3,3) or (T,3,3)
    X: (...,3) camera coords
    returns: (...,2) pixels
    """
    # (..,3)
    if K.ndim == 2:
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    else:
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]

    z = X[..., 2].clamp(min=1e-6)
    u = fx * (X[..., 0] / z) + cx
    v = fy * (X[..., 1] / z) + cy
    return torch.stack([u, v], dim=-1)


@dataclass
class ObsKps:
    # kp_idx refers to mhr70 index
    kp_idx: torch.LongTensor  # (M,)
    xy: torch.FloatTensor     # (M,2)


def _load_bboxes_kps_pkl(path: str) -> Any:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _build_obj_id_to_bbox_idx(bboxes_kps_data: Any) -> Optional[Dict[int, int]]:
    """
    Try to build a stable mapping from obj_id (pid) to bbox index using frame 0.
    Expected: bboxes_kps_data[0]["pids"] exists.
    """
    if bboxes_kps_data is None:
        return None
    try:
        rec0 = bboxes_kps_data[0]
        if isinstance(rec0, dict) and "pids" in rec0:
            pids = rec0["pids"]
            return {int(pid): int(i) for i, pid in enumerate(pids)}
    except Exception:
        return None
    return None


def _get_obs_for_person(
    bboxes_kps_data: Any,
    frame_idx: int,
    obj_id: int,
    obj_id_to_bbox_idx: Optional[Dict[int, int]],
    device: torch.device,
) -> Optional[ObsKps]:
    if bboxes_kps_data is None:
        return None
    if frame_idx < 0:
        return None
    try:
        if frame_idx >= len(bboxes_kps_data):
            return None
    except Exception:
        return None

    rec = bboxes_kps_data[frame_idx]
    if not isinstance(rec, dict):
        return None
    kps_all = rec.get("kps", None)
    if kps_all is None:
        return None
    kps_all = np.asarray(kps_all, dtype=np.float32)
    if kps_all.ndim != 3 or kps_all.shape[-1] < 3:
        return None

    if obj_id_to_bbox_idx is None:
        return None
    if int(obj_id) not in obj_id_to_bbox_idx:
        return None
    bbox_idx = int(obj_id_to_bbox_idx[int(obj_id)])
    if bbox_idx < 0 or bbox_idx >= kps_all.shape[0]:
        return None

    obs = kps_all[bbox_idx]  # (K,3): (x,y,kp_idx)
    kp_idx_list: List[int] = []
    xy_list: List[List[float]] = []
    for row in obs:
        x, y, ki = float(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(ki):
            continue
        kp_idx = int(ki)
        if kp_idx < 0:
            continue
        kp_idx_list.append(kp_idx)
        xy_list.append([x, y])

    if not kp_idx_list:
        return None
    return ObsKps(
        kp_idx=torch.tensor(kp_idx_list, dtype=torch.long, device=device),
        xy=torch.tensor(xy_list, dtype=torch.float32, device=device),
    )


def optimize_translation_sequence(
    *,
    K: torch.Tensor,                 # (3,3)
    X3d: torch.Tensor,               # (T, 70, 3) pred_keypoints_3d (pre-translation)
    t0: torch.Tensor,                # (T, 3) initial pred_cam_t
    obs_by_t: List[Optional[ObsKps]],# length T
    obs_scale: float,
    iters: int,
    lr: float,
    huber_delta: float,
    lambda_prior: float,
    lambda_vel: float,
    lambda_accel: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns optimized translation t_opt (T,3) and a small metrics dict.
    """
    device = X3d.device
    T = X3d.shape[0]

    dt = torch.zeros((T, 3), dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.Adam([dt], lr=float(lr))

    def reproj_loss(t: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=device)
        count = 0
        for ti in range(T):
            obs = obs_by_t[ti]
            if obs is None:
                continue
            Xi = X3d[ti]  # (70,3)
            ti_vec = t[ti].view(1, 3)
            pts = Xi[obs.kp_idx] + ti_vec  # (M,3)
            uv = _project_points(K, pts)   # (M,2)
            resid = uv - (obs.xy * float(obs_scale))
            r = torch.sqrt((resid * resid).sum(dim=-1) + 1e-8)
            loss = loss + _huber(r, huber_delta).mean()
            count += 1
        if count == 0:
            return loss
        return loss / float(count)

    def smoothness(t: torch.Tensor) -> torch.Tensor:
        if T <= 1:
            return torch.zeros((), device=device)
        v = t[1:] - t[:-1]
        l = (v * v).sum(dim=-1).mean()
        if lambda_accel > 0.0 and T >= 3:
            a = t[2:] - 2.0 * t[1:-1] + t[:-2]
            l = l + (a * a).sum(dim=-1).mean() * float(lambda_accel) / max(float(lambda_vel), 1e-8)
        return l

    for _it in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt
        loss_reproj = reproj_loss(t)
        loss_prior = ((t - t0) ** 2).sum(dim=-1).mean()
        loss_smooth = smoothness(t)
        loss = float(lambda_prior) * loss_prior + float(lambda_vel) * loss_smooth + loss_reproj
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_opt = t0 + dt
        metrics = {
            "loss_reproj": float(reproj_loss(t_opt).detach().cpu().item()),
            "loss_prior": float(((t_opt - t0) ** 2).sum(dim=-1).mean().detach().cpu().item()),
            "loss_vel": float(((t_opt[1:] - t_opt[:-1]) ** 2).sum(dim=-1).mean().detach().cpu().item()) if T > 1 else 0.0,
        }
    return t_opt.detach(), metrics


def pick_best_obs_scale(
    *,
    K: torch.Tensor,
    X3d: torch.Tensor,
    t0: torch.Tensor,
    obs_by_t: List[Optional[ObsKps]],
    candidates: Sequence[float],
    huber_delta: float,
) -> float:
    device = X3d.device
    T = X3d.shape[0]
    best_s = float(candidates[0]) if candidates else 1.0
    best_loss = float("inf")
    for s in candidates:
        loss = torch.zeros((), device=device)
        count = 0
        for ti in range(T):
            obs = obs_by_t[ti]
            if obs is None:
                continue
            pts = X3d[ti][obs.kp_idx] + t0[ti].view(1, 3)
            uv = _project_points(K, pts)
            resid = uv - (obs.xy * float(s))
            r = torch.sqrt((resid * resid).sum(dim=-1) + 1e-8)
            loss = loss + _huber(r, huber_delta).mean()
            count += 1
        if count == 0:
            continue
        loss_val = float((loss / float(count)).detach().cpu().item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_s = float(s)
    return best_s


def export_corrected_meshes(
    *,
    out_dir: str,
    faces: np.ndarray,
    frame_name: str,
    obj_id: int,
    pred_vertices: Any,
    cam_t: np.ndarray,
    focal_length: float,
) -> None:
    os.makedirs(os.path.join(out_dir, str(obj_id)), exist_ok=True)
    if isinstance(pred_vertices, torch.Tensor):
        v = pred_vertices.detach().float().cpu().numpy()
    else:
        v = np.asarray(pred_vertices, dtype=np.float32)
    renderer = Renderer(focal_length=float(focal_length), faces=faces)
    color = (0.65, 0.74, 0.86)  # light blue-ish
    tmesh = renderer.vertices_to_trimesh(v, cam_t.astype(np.float32), color)
    tmesh.export(os.path.join(out_dir, str(obj_id), f"{frame_name}.ply"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Option 2: optimize pred_cam_t with robust 2D keypoint reprojection")
    parser.add_argument("--input", required=True, help="Stage2 folder with images/ and masks/")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--bbox-kps-pkl", default=None, help="Path to bboxes_kps_data pkl (provides observed keypoints)")
    parser.add_argument("--camera-intrinsics", default=None, help="Camera intrinsics json (scaled by --camera-scale)")
    parser.add_argument("--camera-scale", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--out", default=None, help="Output dir (default: <input>/optimized_reproj)")

    # Optimization knobs
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--lambda-prior", type=float, default=0.1, help="Stay close to original pred_cam_t")
    parser.add_argument("--lambda-vel", type=float, default=1.0, help="Velocity smoothness weight")
    parser.add_argument("--lambda-accel", type=float, default=0.5, help="Acceleration smoothness (relative)")
    parser.add_argument("--obs-scale-cands", nargs="*", type=float, default=[1.0, 0.5, 2.0])
    args = parser.parse_args()

    input_dir = args.input
    image_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")
    if not os.path.isdir(image_dir) or not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing images/ or masks/ in: {input_dir}")

    cfg_path = args.config or os.path.join(REPO_DIR, "configs", "body4d.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO_DIR, cfg_path)
    cfg = OmegaConf.load(cfg_path)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    estimator = build_sam3d_body_from_config(cfg, device=device)

    bboxes_kps_data = None
    obj_id_to_bbox_idx = None
    if args.bbox_kps_pkl:
        bboxes_kps_data = _load_bboxes_kps_pkl(args.bbox_kps_pkl)
        obj_id_to_bbox_idx = _build_obj_id_to_bbox_idx(bboxes_kps_data)
        print(f"[INFO] Loaded bbox/kps pkl. obj_id_to_bbox_idx: {len(obj_id_to_bbox_idx or {})} ids")
    else:
        print("[WARN] --bbox-kps-pkl not provided; reprojection term will be inactive.")

    cam_int = None
    if args.camera_intrinsics:
        K_np, dist = read_camera_intrinsics(args.camera_intrinsics, scale=float(args.camera_scale))
        cam_int = (torch.from_numpy(K_np).to(device), torch.from_numpy(dist).to(device))
        K_t = cam_int[0]
    else:
        # If not provided, estimator will run FOV estimator; for reprojection we still need K.
        K_t = None

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(image_dir, ext))])
    masks_list = sorted([p for ext in image_extensions for p in glob.glob(os.path.join(masks_dir, ext))])
    n = min(len(images_list), len(masks_list))
    images_list = images_list[:n]
    masks_list = masks_list[:n]
    if n == 0:
        raise FileNotFoundError("Found no images or masks to process.")

    out_dir = args.out or os.path.join(input_dir, "optimized_reproj")
    mesh_out_dir = os.path.join(out_dir, "mesh_4d_individual")
    os.makedirs(mesh_out_dir, exist_ok=True)

    # --- Run model over whole sequence, collecting per-frame per-person outputs ---
    idx_path, idx_dict, mhr_shape_scale_dict, occ_dict = {}, {}, {}, {}
    per_frame: List[Dict[int, Dict[str, Any]]] = []
    frame_names: List[str] = []

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
            image_path = batch_images[bi]
            frame_name = os.path.basename(image_path)[:-4]
            frame_idx = start + bi
            if frame_idx != len(per_frame):
                # ensure we append in order even if empty frames exist
                while len(per_frame) < frame_idx:
                    per_frame.append({})
                    frame_names.append(f"{len(frame_names):08d}")

            if bi in empty_frame_list:
                num_empty += 1
                per_frame.append({})
                frame_names.append(frame_name)
                continue

            out_list = outputs[bi - num_empty]
            ids = id_batch[bi - num_empty]

            frame_dict: Dict[int, Dict[str, Any]] = {}
            for pid, out in enumerate(out_list):
                if ids is not None and pid < len(ids):
                    obj_id = int(ids[pid])
                else:
                    obj_id = int(pid + 1)
                frame_dict[obj_id] = out
            per_frame.append(frame_dict)
            frame_names.append(frame_name)

    T = len(per_frame)
    print(f"[INFO] Collected model outputs for {T} frames")

    # Determine K if not explicitly provided
    if K_t is None:
        # Try to grab from first available output
        K_t = None
        for fd in per_frame:
            for _oid, out in fd.items():
                # camera_head returns pred_keypoints_2d in original image space using batch cam_int
                # but K itself is not stored; we rely on provided intrinsics for proper reprojection.
                pass
        raise ValueError("Provide --camera-intrinsics for this optimization (needed for reprojection).")

    K_t = K_t.detach().float()

    # --- Build per-person sequences and optimize translations ---
    all_obj_ids = sorted({oid for fd in per_frame for oid in fd.keys()})
    print(f"[INFO] Optimizing {len(all_obj_ids)} person(s): {all_obj_ids}")

    summary: Dict[str, Any] = {"input": input_dir, "out": out_dir, "people": {}}

    faces_np = estimator.faces

    for obj_id in all_obj_ids:
        # Collect sequences for this person
        X_list = []
        t0_list = []
        vtx_list = []
        f_list = []
        present_mask = []
        obs_by_t: List[Optional[ObsKps]] = []

        for ti in range(T):
            out = per_frame[ti].get(obj_id, None)
            if out is None:
                present_mask.append(False)
                # dummy placeholders to keep shape; will be masked by obs=None
                X_list.append(torch.zeros((70, 3), dtype=torch.float32))
                t0_list.append(torch.zeros((3,), dtype=torch.float32))
                vtx_list.append(None)
                f_list.append(None)
                obs_by_t.append(None)
                continue
            present_mask.append(True)
            X_list.append(out["pred_keypoints_3d"].detach().float().cpu())
            t0_list.append(out["pred_cam_t"].detach().float().cpu())
            vtx_list.append(out["pred_vertices"])
            f_list.append(float(out.get("focal_length", 0.0)))
            obs_by_t.append(_get_obs_for_person(bboxes_kps_data, ti, obj_id, obj_id_to_bbox_idx, device=torch.device("cpu")))

        X3d = torch.stack(X_list, dim=0).to(torch.device("cpu"))
        t0 = torch.stack(t0_list, dim=0).to(torch.device("cpu"))

        # If no observations, skip optimization and just export original meshes
        has_obs = any(o is not None for o in obs_by_t)
        if not has_obs:
            print(f"[WARN] obj_id={obj_id}: no observed keypoints; skipping reprojection optimization.")
            t_opt = t0
            best_scale = None
            metrics = {}
        else:
            # Pick best scale
            best_scale = pick_best_obs_scale(
                K=K_t.to(torch.device("cpu")),
                X3d=X3d,
                t0=t0,
                obs_by_t=obs_by_t,
                candidates=args.obs_scale_cands,
                huber_delta=float(args.huber_delta),
            )

            t_opt, metrics = optimize_translation_sequence(
                K=K_t.to(torch.device("cpu")),
                X3d=X3d,
                t0=t0,
                obs_by_t=obs_by_t,
                obs_scale=float(best_scale),
                iters=int(args.iters),
                lr=float(args.lr),
                huber_delta=float(args.huber_delta),
                lambda_prior=float(args.lambda_prior),
                lambda_vel=float(args.lambda_vel),
                lambda_accel=float(args.lambda_accel),
            )

        # Export corrected meshes for frames where person exists
        for ti in range(T):
            out = per_frame[ti].get(obj_id, None)
            if out is None:
                continue
            frame_name = frame_names[ti]
            focal = float(out.get("focal_length", 0.0))
            cam_t_np = t_opt[ti].detach().cpu().numpy().astype(np.float32)
            export_corrected_meshes(
                out_dir=mesh_out_dir,
                faces=faces_np,
                frame_name=frame_name,
                obj_id=obj_id,
                pred_vertices=out["pred_vertices"],
                cam_t=cam_t_np,
                focal_length=focal,
            )

        summary["people"][str(obj_id)] = {
            "best_obs_scale": best_scale,
            "metrics": metrics,
        }
        print(f"[OK] obj_id={obj_id}: best_obs_scale={best_scale} metrics={metrics}")

    with open(os.path.join(out_dir, "optimize_reproj_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Done. Corrected meshes in: {mesh_out_dir}")
    print(f"[INFO] Summary: {os.path.join(out_dir, 'optimize_reproj_summary.json')}")


if __name__ == "__main__":
    main()

