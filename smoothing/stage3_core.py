from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils import kalman_smooth_mhr_params_per_obj_id_adaptive, ema_smooth_global_rot_per_obj_id_adaptive

from .obs_kps import ObsKps, build_obj_id_to_bbox_idx, get_obs_for_person, load_bboxes_kps_pkl
from .reproj_opt import ReprojOptConfig, optimize_pred_cam_t
from .ground_plane_opt import GroundOptConfig, load_extrinsics_json, optimize_ground_plane_translation
from .mask_reproj_opt import MaskReprojOptConfig, load_masks_for_sequence, optimize_mask_reprojection


@dataclass
class Stage3Config:
    enable_option1: bool = True
    enable_mask_reproj: bool = True  # NEW: mask-based reproj (default ON)
    enable_kps_reproj: bool = False  # Legacy keypoint-based reproj (default OFF)
    enable_ground: bool = False

    # Mask reprojection inputs (uses Stage 1 masks)
    mask_dir: Optional[str] = None  # Auto-detected from raw_mhr.pt path
    mask_reproj_cfg: MaskReprojOptConfig = field(default_factory=MaskReprojOptConfig)

    # Legacy keypoint reprojection inputs
    bbox_kps_pkl: Optional[str] = None
    camera_intrinsics_json: Optional[str] = None
    camera_scale: float = 0.5
    reproj_cfg: ReprojOptConfig = field(default_factory=ReprojOptConfig)

    # Ground inputs
    extrinsics_json: Optional[str] = None
    ground_cfg: GroundOptConfig = field(default_factory=GroundOptConfig)


# Re-export from canonical location for backward compatibility
from utils.camera_utils import read_camera_intrinsics, read_camera_intrinsics_new, adjust_K

def stack_frames_to_tensors(
    frames: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[List[int]], Dict[int, List[int]], List[str], List[int]]:
    """
    Convert raw frames payload into flattened tensors shaped (T*N, D).
    Slot order is fixed across frames: sorted(all_obj_ids).
    """
    T = len(frames)
    frame_names = [str(fr.get("frame", f"{i:08d}")) for i, fr in enumerate(frames)]
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

    slot_of = {oid: si for si, oid in enumerate(obj_ids_all)}

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


def freeze_shape_scale_first_frame(mhr: Dict[str, torch.Tensor], T: int, N: int) -> None:
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


def run_stage3_post_optimizations(
    *,
    cfg: Stage3Config,
    device: torch.device,
    mhr: Dict[str, torch.Tensor],
    frame_names: List[str],
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    vis_flags: Dict[int, List[int]],
    # required for recompute/projection
    keypoints3d_local: torch.Tensor,  # (T*N, 70, 3) after mhr_forward camera-axis flips, BEFORE adding pred_cam_t
    vertices_local: Optional[torch.Tensor] = None,  # (T*N, V, 3) mesh vertices (for mask reproj)
    segment_id_mappings: Optional[List[Dict[str, Any]]] = None,  # per-segment actual->consecutive mapping
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Apply optional post-optimizations on top of base Option1 smoothing.
    Currently both optimizers adjust pred_cam_t only, and they are NOT mutually exclusive.
    They can be applied sequentially:
      - mask reprojection (if enabled, default ON)
      - legacy keypoint reprojection (if enabled)
      - ground-plane/contact (if enabled)
    """
    summary: Dict[str, Any] = {"mask_reproj": {}, "kps_reproj": {}, "ground": {}}
    T = len(frame_names)
    N = len(obj_ids_all)

    # NEW: Mask-based reprojection optimization (default ON)
    if cfg.enable_mask_reproj:
        if not cfg.mask_dir:
            raise ValueError("mask_dir is required for mask-based reprojection optimization.")
        if not cfg.camera_intrinsics_json:
            raise ValueError("--camera-intrinsics-json is required for mask-based reprojection optimization.")
        if vertices_local is None:
            raise ValueError("vertices_local is required for mask-based reprojection optimization.")

        K_np, _dist = read_camera_intrinsics_new(cfg.camera_intrinsics_json)
        K_np = adjust_K(K_np, scale=float(cfg.camera_scale))
        K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32)

        # Load masks from Stage 1 output
        print(f"[INFO] Loading masks from: {cfg.mask_dir}")
        masks = load_masks_for_sequence(cfg.mask_dir, frame_names, device=device)
        print(f"[INFO] Loaded {len(masks)} mask frames")

        pred_cam_t = mhr["pred_cam_t"].view(T, N, 3).contiguous()
        V = vertices_local.shape[1] if vertices_local.dim() == 3 else vertices_local.numel() // (T * N * 3)
        # Keep verts_all on whatever device vertices_local is on (may be CPU); move per-person slices to GPU on demand
        verts_all = vertices_local.view(T, N, V, 3).contiguous()

        # Build per-frame actual->consecutive reverse mapping for mask pixel lookup.
        # Masks store consecutive IDs (1,2,3...) while obj_ids_all has actual PIDs.
        # segment_id_mappings[i] = {frame_start, frame_end, consecutive_to_actual: {consec->actual}}
        # We invert to actual->consecutive per frame.
        actual_to_consec_per_frame: Optional[List[Dict[int, int]]] = None
        if segment_id_mappings:
            actual_to_consec_per_frame = [{} for _ in range(T)]
            for seg in segment_id_mappings:
                inv = {int(v): int(k) for k, v in seg["consecutive_to_actual"].items()}
                fs, fe = int(seg["frame_start"]), int(seg["frame_end"])
                for t in range(max(0, fs), min(T, fe + 1)):
                    actual_to_consec_per_frame[t] = inv

        for si, oid in enumerate(obj_ids_all):
            present = torch.tensor([1 if frame_obj_ids_slots[t][si] == oid else 0 for t in range(T)], device=device)
            if int(present.sum().item()) == 0:
                continue

            # Build per-frame mask pixel ID for this person
            person_mask_ids: Optional[Dict[int, int]] = None
            if actual_to_consec_per_frame is not None:
                person_mask_ids = {}
                for t in range(T):
                    mapping = actual_to_consec_per_frame[t]
                    if oid in mapping:
                        person_mask_ids[t] = mapping[oid]

            t0 = pred_cam_t[:, si, :]  # (T, 3) — already on device
            V_person = verts_all[:, si, :, :].to(device)  # (T, V, 3) — move one person to GPU
            t_opt, metrics = optimize_mask_reprojection(
                K=K,
                vertices_local=V_person,
                t0=t0,
                masks=masks,
                obj_id=oid,
                mask_ids=person_mask_ids,
                cfg=cfg.mask_reproj_cfg,
            )
            pred_cam_t[:, si, :] = t_opt
            del V_person
            summary["mask_reproj"][str(oid)] = metrics

        mhr["pred_cam_t"] = pred_cam_t.view(T * N, 3)

    # Legacy keypoint-based reprojection optimization (default OFF)
    if cfg.enable_kps_reproj:
        if not cfg.camera_intrinsics_json:
            raise ValueError("--camera-intrinsics-json is required for keypoint reprojection optimization.")
        if not cfg.bbox_kps_pkl:
            raise ValueError("--bbox-kps-pkl is required for keypoint reprojection optimization.")

        # K_np, _dist = read_camera_intrinsics(cfg.camera_intrinsics_json, scale=float(cfg.camera_scale))
        K_np, _dist = read_camera_intrinsics_new(cfg.camera_intrinsics_json)
        K_np = adjust_K(K_np, scale=float(cfg.camera_scale))
        K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32)

        bboxes_kps_data = load_bboxes_kps_pkl(cfg.bbox_kps_pkl)
        obj_id_to_bbox_idx = build_obj_id_to_bbox_idx(bboxes_kps_data)
        if not obj_id_to_bbox_idx:
            raise ValueError("Could not build obj_id_to_bbox_idx from bbox/kps pkl (expected frame0['pids']).")

        pred_cam_t = mhr["pred_cam_t"].view(T, N, 3).contiguous()
        # MHR can return 70 (body only) or 308 (full SMPL-X) keypoints.
        # Reproj optimization only needs the first 70 (body joints) where observed kp indices are defined.
        K_kps = keypoints3d_local.shape[1] if keypoints3d_local.dim() == 3 else keypoints3d_local.numel() // (T * N * 3)
        X3d_full = keypoints3d_local.view(T, N, K_kps, 3).contiguous()
        K_body = min(K_kps, 70)
        X3d = X3d_full[:, :, :K_body, :].contiguous()

        for si, oid in enumerate(obj_ids_all):
            # Only optimize if present at least once
            present = torch.tensor([1 if frame_obj_ids_slots[t][si] == oid else 0 for t in range(T)], device=device)
            if int(present.sum().item()) == 0:
                continue

            obs_by_t: List[Optional[ObsKps]] = []
            for t in range(T):
                if frame_obj_ids_slots[t][si] != oid:
                    obs_by_t.append(None)
                    continue
                obs_by_t.append(get_obs_for_person(bboxes_kps_data, t, oid, obj_id_to_bbox_idx, device=device))

            t0 = pred_cam_t[:, si, :]
            X = X3d[:, si, :, :]
            t_opt, metrics = optimize_pred_cam_t(K=K, X3d=X, t0=t0, obs_by_t=obs_by_t, cfg=cfg.reproj_cfg)
            pred_cam_t[:, si, :] = t_opt
            summary["kps_reproj"][str(oid)] = metrics

        mhr["pred_cam_t"] = pred_cam_t.view(T * N, 3)

    # Ground plane/contact optimization
    if cfg.enable_ground:
        if not cfg.extrinsics_json:
            raise ValueError("--extrinsics-json is required for ground-plane/contact optimization.")
        extr = load_extrinsics_json(cfg.extrinsics_json, device=device)

        pred_cam_t = mhr["pred_cam_t"].view(T, N, 3).contiguous()
        # MHR can return 70 (body only) or 308 (full SMPL-X) keypoints.
        # Ground optimization only needs the first 70 (body joints) where foot indices are defined.
        K_kps = keypoints3d_local.shape[1] if keypoints3d_local.dim() == 3 else keypoints3d_local.numel() // (T * N * 3)
        X3d_full = keypoints3d_local.view(T, N, K_kps, 3).contiguous()
        # Use only the first 70 keypoints (body joints) for ground optimization
        K_body = min(K_kps, 70)
        X3d = X3d_full[:, :, :K_body, :].contiguous()

        for si, oid in enumerate(obj_ids_all):
            present = torch.tensor([1 if frame_obj_ids_slots[t][si] == oid else 0 for t in range(T)], device=device)
            if int(present.sum().item()) == 0:
                continue

            t0 = pred_cam_t[:, si, :]
            X = X3d[:, si, :, :]
            t_opt, metrics = optimize_ground_plane_translation(extr=extr, X3d=X, t0=t0, cfg=cfg.ground_cfg)
            pred_cam_t[:, si, :] = t_opt
            summary["ground"][str(oid)] = metrics

        mhr["pred_cam_t"] = pred_cam_t.view(T * N, 3)

    return mhr, summary

