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


def _stack_frames_to_tensors(
    frames: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[List[int]], Dict[int, List[int]], List[str], List[int]]:
    """
    Convert raw frames payload into flattened tensors shaped (T*N, D).
    We use a fixed slot order across all frames: sorted(all_obj_ids).

    Returns:
      mhr_dict: dict of tensors (B, D)
      frame_obj_ids_slots: List[List[int]] length T, each list length N where missing -> 0
      vis_flags: dict[obj_id] -> List[0/1] length T
      frame_names: List[str] length T
      obj_ids_all: sorted list of obj_ids (slot order)
    """
    T = len(frames)
    frame_names = [str(fr.get("frame", f"{i:08d}")) for i, fr in enumerate(frames)]
    # gather all obj_ids
    obj_set = set()
    for fr in frames:
        for p in fr.get("people", []):
            if p is None:
                continue
            oid = p.get("obj_id", None)
            if oid is not None:
                obj_set.add(int(oid))
    obj_ids_all = sorted(obj_set)
    N = len(obj_ids_all)
    if N == 0:
        raise ValueError("No people found in raw payload.")

    # slot mapping
    slot_of = {oid: si for si, oid in enumerate(obj_ids_all)}

    # Build per-frame slot list and visibility flags
    frame_obj_ids_slots: List[List[int]] = []
    vis_flags: Dict[int, List[int]] = {oid: [0] * T for oid in obj_ids_all}
    for ti, fr in enumerate(frames):
        slots = [0] * N
        for p in fr.get("people", []):
            if p is None:
                continue
            oid = int(p.get("obj_id"))
            si = slot_of.get(oid, None)
            if si is None:
                continue
            slots[si] = oid
            vis_flags[oid][ti] = 1
        frame_obj_ids_slots.append(slots)

    # Infer dimensions from first present entry
    def first_shape(key: str) -> int:
        for fr in frames:
            for p in fr.get("people", []):
                v = p.get(key, None)
                if v is not None:
                    arr = np.asarray(v)
                    return int(arr.reshape(-1).shape[0])
        return 0

    dims = {
        "global_rot": first_shape("global_rot"),
        "body_pose": first_shape("body_pose"),
        "hand": first_shape("hand"),
        "scale": first_shape("scale"),
        "shape": first_shape("shape"),
        "face": first_shape("face"),
        "pred_cam_t": first_shape("pred_cam_t"),
        "focal_length": 1,
    }

    B = T * N
    mhr: Dict[str, torch.Tensor] = {}
    for k, d in dims.items():
        if d <= 0:
            continue
        mhr[k] = torch.zeros((B, d), dtype=torch.float32, device=device)

    # Fill tensors
    for ti, fr in enumerate(frames):
        for p in fr.get("people", []):
            oid = int(p.get("obj_id"))
            si = slot_of[oid]
            bi = ti * N + si
            for k in ["global_rot", "body_pose", "hand", "scale", "shape", "face", "pred_cam_t"]:
                if k not in mhr:
                    continue
                v = p.get(k, None)
                if v is None:
                    continue
                vv = torch.from_numpy(np.asarray(v, dtype=np.float32).reshape(-1)).to(device)
                if vv.numel() == mhr[k].shape[1]:
                    mhr[k][bi] = vv
            if "focal_length" in mhr:
                fl = p.get("focal_length", None)
                if fl is not None:
                    mhr["focal_length"][bi, 0] = float(np.asarray(fl).reshape(-1)[0])

    return mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all


def _freeze_shape_scale_first_frame(mhr: Dict[str, torch.Tensor], T: int, N: int) -> None:
    # In-place: set shape/scale for each slot to first frame values across time.
    if "shape" in mhr:
        shp = mhr["shape"].view(T, N, -1)
        first = shp[0].clone()
        shp[:] = first[None, :, :]
        mhr["shape"] = shp.view(T * N, -1)
    if "scale" in mhr:
        sc = mhr["scale"].view(T, N, -1)
        first = sc[0].clone()
        sc[:] = first[None, :, :]
        mhr["scale"] = sc.view(T * N, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: smooth raw params and export meshes")
    parser.add_argument("--raw", required=True, help="raw_mhr.pt from run_sam3d_body_raw_params.py")
    parser.add_argument("--config", default=None, help="Config YAML (default: configs/body4d.yaml)")
    parser.add_argument("--out", default=None, help="Output dir (default: alongside raw file)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
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

    mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all = _stack_frames_to_tensors(frames, device=device)
    T = len(frame_names)
    N = len(obj_ids_all)
    print(f"[INFO] Loaded raw: T={T}, N={N}, obj_ids={obj_ids_all[:10]}{'...' if N>10 else ''}")

    # --- Option 1 smoothing over full sequence ---
    mhr = kalman_smooth_mhr_params_per_obj_id_adaptive(
        mhr_dict=mhr,
        num_frames=T,
        frame_obj_ids=frame_obj_ids_slots,
        keys_to_smooth=[k for k in ["body_pose", "hand", "pred_cam_t"] if k in mhr],
        kalman_cfg=None,
        vis_flags=vis_flags,
    )
    _freeze_shape_scale_first_frame(mhr, T=T, N=N)
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

    verts, j3d, _jcoords, _mhr_params, _joint_rots = head_pose.mhr_forward(
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
    # Camera system difference (match existing pipeline)
    verts[..., [1, 2]] *= -1
    j3d[..., [1, 2]] *= -1

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

