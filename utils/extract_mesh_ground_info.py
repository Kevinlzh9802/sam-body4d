"""
Extract 2D ground-plane positions and orientations from body keypoints (MHR70).

All positions are in world xy (cm). Orientations are in radians [0, 2*pi);
0 = (1,0), pi/2 = (0,1). Used after Stage 3 mesh export; expects keypoints
already in world space (same as exported meshes).
"""

from __future__ import annotations

import csv
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# MHR70 keypoint indices (from models/sam_3d_body/sam_3d_body/metadata/mhr70.py)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 9, 10
LEFT_ANKLE, RIGHT_ANKLE = 13, 14
LEFT_BIG_TOE, LEFT_SMALL_TOE = 15, 16
RIGHT_BIG_TOE, RIGHT_SMALL_TOE = 18, 19
NECK = 69


def _angle_from_xy(x: float, y: float) -> float:
    """Angle in [0, 2*pi). 0 = (1,0), pi/2 = (0,1)."""
    a = np.arctan2(float(y), float(x))
    if a < 0:
        a += 2.0 * np.pi
    return a


def _perp_angle_from_line(lx: float, ly: float, rx: float, ry: float) -> float:
    """Angle of perpendicular to line L->R (in 2D). Returns [0, 2*pi)."""
    vx, vy = rx - lx, ry - ly
    perp_x, perp_y = -vy, vx
    return _angle_from_xy(perp_x, perp_y)


def _mean_angle_rad(a1: float, a2: float) -> float:
    """Circular mean of two angles in radians, result in [0, 2*pi)."""
    s, c = np.sin(a1) + np.sin(a2), np.cos(a1) + np.cos(a2)
    a = np.arctan2(s, c)
    if a < 0:
        a += 2.0 * np.pi
    return a


def extract_ground_info_from_keypoints_2d(
    keypoints_xy: np.ndarray,
    frame_names: List[str],
    obj_ids_all: List[Any],
    frame_obj_ids_slots: List[List[Any]],
    T: int,
    N: int,
) -> List[Dict[str, Any]]:
    """
    Extract 2D ground-plane info from keypoints already in world xy (cm).

    keypoints_xy: (T*N, 70, 2) in world xy, units cm.
    frame_obj_ids_slots[t][s] == obj_ids_all[s] if that object is present in frame t.

    Returns a list of dicts, one per (frame, obj_id) present, with keys:
      frame_name, obj_id, head_xy, head_orient_rad,
      shoulder_left_xy, shoulder_right_xy, shoulder_orient_rad,
      hip_left_xy, hip_right_xy, hip_orient_rad,
      ankle_left_xy, ankle_right_xy, ankle_orient_rad,
      toe_left_xy, toe_right_xy, toe_orient_rad,
      foot_left_xy, foot_right_xy, foot_orient_rad.
    """
    rows: List[Dict[str, Any]] = []
    for ti in range(T):
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            k = keypoints_xy[bi]  # (70, 2)

            # Head: position = nose; orientation = neck -> nose (forward)
            head_xy = k[NOSE].copy()
            neck_xy = k[NECK]
            dx = head_xy[0] - neck_xy[0]
            dy = head_xy[1] - neck_xy[1]
            head_orient_rad = _angle_from_xy(dx, dy)

            # Shoulders
            ls_xy = k[LEFT_SHOULDER].copy()
            rs_xy = k[RIGHT_SHOULDER].copy()
            shoulder_orient_rad = _perp_angle_from_line(
                ls_xy[0], ls_xy[1], rs_xy[0], rs_xy[1]
            )

            # Hips
            lh_xy = k[LEFT_HIP].copy()
            rh_xy = k[RIGHT_HIP].copy()
            hip_orient_rad = _perp_angle_from_line(
                lh_xy[0], lh_xy[1], rh_xy[0], rh_xy[1]
            )

            # Ankles
            la_xy = k[LEFT_ANKLE].copy()
            ra_xy = k[RIGHT_ANKLE].copy()
            ankle_orient_rad = _perp_angle_from_line(
                la_xy[0], la_xy[1], ra_xy[0], ra_xy[1]
            )

            # Toes: left = mean(big, small), right = mean(big, small)
            toe_left_xy = (
                (k[LEFT_BIG_TOE] + k[LEFT_SMALL_TOE]) / 2.0
            ).astype(np.float64)
            toe_right_xy = (
                (k[RIGHT_BIG_TOE] + k[RIGHT_SMALL_TOE]) / 2.0
            ).astype(np.float64)
            toe_orient_rad = _perp_angle_from_line(
                toe_left_xy[0], toe_left_xy[1],
                toe_right_xy[0], toe_right_xy[1],
            )

            # Feet: centroid of 2 toe + 2 ankle per foot; orient = mean(ankle, toe)
            foot_left_xy = (
                k[LEFT_ANKLE] + k[LEFT_BIG_TOE] + k[LEFT_SMALL_TOE]
            ) / 3.0
            foot_right_xy = (
                k[RIGHT_ANKLE] + k[RIGHT_BIG_TOE] + k[RIGHT_SMALL_TOE]
            ) / 3.0
            foot_orient_rad = _mean_angle_rad(ankle_orient_rad, toe_orient_rad)

            rows.append({
                "frame_name": frame_names[ti],
                "obj_id": oid,
                "head_xy": head_xy,
                "head_orient_rad": float(head_orient_rad),
                "shoulder_left_xy": ls_xy,
                "shoulder_right_xy": rs_xy,
                "shoulder_orient_rad": float(shoulder_orient_rad),
                "hip_left_xy": lh_xy,
                "hip_right_xy": rh_xy,
                "hip_orient_rad": float(hip_orient_rad),
                "ankle_left_xy": la_xy,
                "ankle_right_xy": ra_xy,
                "ankle_orient_rad": float(ankle_orient_rad),
                "toe_left_xy": toe_left_xy,
                "toe_right_xy": toe_right_xy,
                "toe_orient_rad": float(toe_orient_rad),
                "foot_left_xy": foot_left_xy,
                "foot_right_xy": foot_right_xy,
                "foot_orient_rad": float(foot_orient_rad),
            })
    return rows


def save_ground_info(
    rows: List[Dict[str, Any]],
    output_dir: str,
    basename: str = "ground_plane_info",
    save_pkl: bool = True,
    save_csv: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Save extracted ground info to output_dir. Returns (pkl_path, csv_path).
    """
    pkl_path = os.path.join(output_dir, f"{basename}.pkl") if save_pkl else None
    csv_path = os.path.join(output_dir, f"{basename}.csv") if save_csv else None

    if save_pkl:
        with open(pkl_path, "wb") as f:
            pickle.dump(rows, f)

    if save_csv:
        if not rows:
            if csv_path:
                open(csv_path, "w").close()
        else:
            # Flatten first row to get column names (excluding array fields for CSV)
            flat_keys = [
                "frame_name", "obj_id",
                "head_x", "head_y", "head_orient_rad",
                "shoulder_left_x", "shoulder_left_y",
                "shoulder_right_x", "shoulder_right_y", "shoulder_orient_rad",
                "hip_left_x", "hip_left_y", "hip_right_x", "hip_right_y", "hip_orient_rad",
                "ankle_left_x", "ankle_left_y", "ankle_right_x", "ankle_right_y", "ankle_orient_rad",
                "toe_left_x", "toe_left_y", "toe_right_x", "toe_right_y", "toe_orient_rad",
                "foot_left_x", "foot_left_y", "foot_right_x", "foot_right_y", "foot_orient_rad",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    flat = {
                        "frame_name": r["frame_name"],
                        "obj_id": r["obj_id"],
                        "head_x": r["head_xy"][0], "head_y": r["head_xy"][1],
                        "head_orient_rad": r["head_orient_rad"],
                        "shoulder_left_x": r["shoulder_left_xy"][0], "shoulder_left_y": r["shoulder_left_xy"][1],
                        "shoulder_right_x": r["shoulder_right_xy"][0], "shoulder_right_y": r["shoulder_right_xy"][1],
                        "shoulder_orient_rad": r["shoulder_orient_rad"],
                        "hip_left_x": r["hip_left_xy"][0], "hip_left_y": r["hip_left_xy"][1],
                        "hip_right_x": r["hip_right_xy"][0], "hip_right_y": r["hip_right_xy"][1],
                        "hip_orient_rad": r["hip_orient_rad"],
                        "ankle_left_x": r["ankle_left_xy"][0], "ankle_left_y": r["ankle_left_xy"][1],
                        "ankle_right_x": r["ankle_right_xy"][0], "ankle_right_y": r["ankle_right_xy"][1],
                        "ankle_orient_rad": r["ankle_orient_rad"],
                        "toe_left_x": r["toe_left_xy"][0], "toe_left_y": r["toe_left_xy"][1],
                        "toe_right_x": r["toe_right_xy"][0], "toe_right_y": r["toe_right_xy"][1],
                        "toe_orient_rad": r["toe_orient_rad"],
                        "foot_left_x": r["foot_left_xy"][0], "foot_left_y": r["foot_left_xy"][1],
                        "foot_right_x": r["foot_right_xy"][0], "foot_right_y": r["foot_right_xy"][1],
                        "foot_orient_rad": r["foot_orient_rad"],
                    }
                    w.writerow(flat)

    return (pkl_path, csv_path)


def run_extract_ground_info(
    keypoints3d_local: np.ndarray,
    pred_cam_t: np.ndarray,
    extr: Optional[Any],
    world_scale: float,
    frame_names: List[str],
    obj_ids_all: List[Any],
    frame_obj_ids_slots: List[List[Any]],
    T: int,
    N: int,
    output_dir: str,
    basename: str = "ground_plane_info",
    device: Optional[Any] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert keypoints to world xy (cm), extract ground info, save to output_dir.

    keypoints3d_local: (T*N, 70, 3) in model local space (same as j3d from MHR forward).
    pred_cam_t: (T*N, 3). extr has .cam_to_world(X) method; if None, uses camera xy (no scale).
    world_scale: scale to apply before cam_to_world (e.g. 100 for m->cm).
    device: used when extr is not None so tensors match extr's device.

    Returns (pkl_path, rows) or (None, []) if skipped (e.g. no extrinsics).
    """
    import torch
    keypoints3d_local_t = torch.from_numpy(
        np.asarray(keypoints3d_local, dtype=np.float32)
    )
    pred_cam_t_t = torch.from_numpy(
        np.asarray(pred_cam_t, dtype=np.float32)
    )
    # pred_cam_t is (T*N, 3); keypoints are (T*N, K, 3) — unsqueeze for broadcast
    kp_cam = keypoints3d_local_t + pred_cam_t_t.unsqueeze(1)  # (T*N, K, 3)
    if extr is not None:
        dev = getattr(extr.R, "device", device) or device or torch.device("cpu")
        kp_cam = kp_cam.to(dev)
        kp_cam_scaled = kp_cam * world_scale
        kp_world = extr.cam_to_world(kp_cam_scaled.view(-1, 3))
        kp_world = kp_world.view(kp_cam.shape[0], kp_cam.shape[1], 3)
        kp_xy = kp_world.cpu().numpy()[..., :2]
    else:
        # Camera space: use xy as-is; scale by world_scale for consistent units if desired
        kp_xy = (kp_cam[..., :2].numpy() * world_scale)

    rows = extract_ground_info_from_keypoints_2d(
        kp_xy, frame_names, obj_ids_all, frame_obj_ids_slots, T, N
    )
    os.makedirs(output_dir, exist_ok=True)
    pkl_path, csv_path = save_ground_info(rows, output_dir, basename=basename)
    return pkl_path, rows
