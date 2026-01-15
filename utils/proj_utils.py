"""
Projection / reprojection helper utilities.

This file is intentionally lightweight so it can be imported from inference scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import json

def read_camera_intrinsics(intrinsic_file: str, scale):
    with open(intrinsic_file, "r") as f:
        intrinsic_data = json.load(f)
        K = np.array(intrinsic_data["intrinsic"])
        dist_coeffs = np.array(intrinsic_data["distortion_coefficients"])
        # Scale K to match 0.5x resolution images fed to SAM3D
        K = adjust_K(K, scale=scale)
    return K, dist_coeffs

def adjust_K(K, scale):
    K_resized = np.array([[K[0,0]*scale, 0,           K[0,2]*scale],
             [0,           K[1,1]*scale, K[1,2]*scale],
             [0,           0,         1]])
    return K_resized


def _to_jsonable(x: Any) -> Any:
    """
    Convert common numeric containers (torch/numpy) to JSON-serializable python types.
    """
    if x is None:
        return None
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).tolist()
    try:
        return list(x)
    except Exception:
        return x


def _as_numpy_keypoints_2d(pred_keypoints_2d: Any) -> Optional[np.ndarray]:
    """
    Convert predicted 2D keypoints to a numpy array (K, 2) in float32.
    Accepts torch.Tensor or np.ndarray-like objects.
    """
    if pred_keypoints_2d is None:
        return None
    try:
        import torch  # type: ignore

        if isinstance(pred_keypoints_2d, torch.Tensor):
            return pred_keypoints_2d.detach().float().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    try:
        arr = np.asarray(pred_keypoints_2d, dtype=np.float32)
        return arr
    except Exception:
        return None


def compute_pelvis_proxy(
    pred_keypoints_3d: Any,
    pred_cam_t: Any,
    *,
    left_hip_idx: int = 9,
    right_hip_idx: int = 10,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Compute a simple pelvis proxy as mean of (left_hip, right_hip) from mhr70 keypoints.
    Returns:
      pelvis_local: [3] in the model's local coords (before translation)
      pelvis_cam:   [3] after adding pred_cam_t (approx camera coords used for mesh export)
    """
    pelvis_local = None
    pelvis_cam = None
    if pred_keypoints_3d is None:
        return pelvis_local, pelvis_cam

    try:
        import torch  # type: ignore

        if isinstance(pred_keypoints_3d, torch.Tensor):
            k3d = pred_keypoints_3d.detach().float().cpu().numpy()
        else:
            k3d = np.asarray(pred_keypoints_3d, dtype=np.float32)
    except Exception:
        k3d = None

    if k3d is None or k3d.ndim != 2 or k3d.shape[1] != 3:
        return pelvis_local, pelvis_cam
    if max(left_hip_idx, right_hip_idx) >= k3d.shape[0]:
        return pelvis_local, pelvis_cam

    pelvis_local_np = 0.5 * (k3d[left_hip_idx] + k3d[right_hip_idx])
    pelvis_local = pelvis_local_np.astype(np.float32).tolist()

    if pred_cam_t is None:
        return pelvis_local, pelvis_cam

    try:
        import torch  # type: ignore

        if isinstance(pred_cam_t, torch.Tensor):
            camt = pred_cam_t.detach().float().cpu().numpy().reshape(3)
        else:
            camt = np.asarray(pred_cam_t, dtype=np.float32).reshape(3)
        pelvis_cam = (pelvis_local_np + camt).astype(np.float32).tolist()
    except Exception:
        pass

    return pelvis_local, pelvis_cam


def compute_reproj_2d_alignment_debug(
    *,
    bboxes_kps_data: Any,
    frame_idx: int,
    obj_id: int,
    pred_keypoints_2d: Any,
    obj_id_to_bbox_idx: Optional[Dict[int, int]] = None,
    obs_scale_candidates: Sequence[float] = (1.0, 0.5, 2.0),
) -> Optional[Dict[str, Any]]:
    """
    Compute a small reprojection / 2D alignment debug dict.

    Observations come from bboxes_kps_data[frame_idx]["kps"], shaped (N, K, 3):
      (x, y, kp_idx), where kp_idx indexes into the model's mhr70 set.

    We compare observed (x, y) (optionally scaled by obs_scale_candidates) to predicted
    keypoints at pred_keypoints_2d[kp_idx] (K=70 typically).

    Returns:
      dict with best candidate and per-candidate mean/median L2 errors, or None if not available.
    """
    if bboxes_kps_data is None:
        return None

    if frame_idx < 0:
        return None

    try:
        n_frames = len(bboxes_kps_data)
    except Exception:
        return None

    if frame_idx >= n_frames:
        return None

    frame_rec = bboxes_kps_data[frame_idx]
    kps_all = None
    if isinstance(frame_rec, dict):
        kps_all = frame_rec.get("kps", None)
    if kps_all is None:
        try:
            kps_all = frame_rec["kps"]  # type: ignore[index]
        except Exception:
            kps_all = None
    if kps_all is None:
        return None

    kps_all = np.asarray(kps_all)
    if kps_all.ndim != 3 or kps_all.shape[-1] < 3:
        return None

    # Map obj_id -> bbox_idx.
    # Prefer an explicit mapping (works for discontinuous IDs like [1,2,4,5,8,15]).
    # Fallback to legacy behavior: if no mapping is provided, treat obj_id as 1-based index.
    if obj_id_to_bbox_idx is not None:
        bbox_idx = int(obj_id_to_bbox_idx.get(int(obj_id), -1))
    else:
        bbox_idx = int(obj_id) - 1
    if bbox_idx < 0 or bbox_idx >= kps_all.shape[0]:
        return None

    obs = np.asarray(kps_all[bbox_idx], dtype=np.float32)  # (K, 3)
    pred2d = _as_numpy_keypoints_2d(pred_keypoints_2d)
    if pred2d is None or pred2d.ndim != 2 or pred2d.shape[1] != 2:
        return None

    # Collect valid observed points
    pairs: List[Tuple[float, float, int]] = []
    for row in obs:
        if row.shape[0] < 3:
            continue
        x_obs, y_obs, kp_idx_f = float(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(x_obs) or not np.isfinite(y_obs) or not np.isfinite(kp_idx_f):
            continue
        kp_idx = int(kp_idx_f)
        if kp_idx < 0 or kp_idx >= pred2d.shape[0]:
            continue
        pairs.append((x_obs, y_obs, kp_idx))

    if not pairs:
        return None

    errs: Dict[str, Dict[str, Any]] = {}
    for s_obs in obs_scale_candidates:
        e: List[float] = []
        for x_obs, y_obs, kp_idx in pairs:
            x_pred, y_pred = float(pred2d[kp_idx, 0]), float(pred2d[kp_idx, 1])
            dx = x_pred - (x_obs * float(s_obs))
            dy = y_pred - (y_obs * float(s_obs))
            e.append(float((dx * dx + dy * dy) ** 0.5))
        if e:
            errs[f"obs_scale_{float(s_obs):g}"] = {
                "count": int(len(e)),
                "mean_l2": float(np.mean(e)),
                "median_l2": float(np.median(e)),
            }

    if not errs:
        return None

    best_key = min(errs.keys(), key=lambda k: errs[k]["mean_l2"])
    return {"best": best_key, "candidates": errs}


def build_pred_cam_t_debug_record(
    *,
    frame_name: str,
    obj_id: int,
    person_output: Dict[str, Any],
    bboxes_kps_data: Any = None,
    obj_id_to_bbox_idx: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable debug record for pred_cam_t and optional 2D alignment metrics.
    """
    pred_cam_t = person_output.get("pred_cam_t", None)
    pred_cam = person_output.get("pred_cam", None)
    focal_length = person_output.get("focal_length", None)
    k3d = person_output.get("pred_keypoints_3d", None)
    k2d = person_output.get("pred_keypoints_2d", None)

    pelvis_local, pelvis_cam = compute_pelvis_proxy(k3d, pred_cam_t)

    rec: Dict[str, Any] = {
        "frame": frame_name,
        "obj_id": int(obj_id),
        "pred_cam_t": _to_jsonable(pred_cam_t),
        "pred_cam": _to_jsonable(pred_cam),
        "focal_length": _to_jsonable(focal_length),
        "pelvis_local": pelvis_local,
        "pelvis_cam": pelvis_cam,
    }

    # Optional 2D alignment debug
    try:
        frame_idx = int(frame_name)
    except Exception:
        frame_idx = -1
    reproj_dbg = compute_reproj_2d_alignment_debug(
        bboxes_kps_data=bboxes_kps_data,
        frame_idx=frame_idx,
        obj_id=int(obj_id),
        pred_keypoints_2d=k2d,
        obj_id_to_bbox_idx=obj_id_to_bbox_idx,
    )
    if reproj_dbg is not None:
        rec["reproj_2d_err"] = reproj_dbg

    return rec

