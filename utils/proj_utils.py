"""
Projection / reprojection helper utilities.

This file is intentionally lightweight so it can be imported from inference scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import glob
import cv2

from utils.camera_utils import read_camera_intrinsics, adjust_K  # noqa: F401 — re-export


def _to_jsonable(x: Any) -> Any:
    """
    Convert common numeric containers (torch/numpy) to JSON-serializable python types.
    """
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).tolist()
    if isinstance(x, np.generic):
        return x.item()
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


def undistort_video(
    input_video_path: str,
    output_video_path: str,
    intrinsic_json_path: str,
    *,
    scale: float = 1.0,
    balance: float = 0.0,
    distortion_model: str = "standard",
    crop_to_roi: bool = False,
    overwrite: bool = False,
    codec: str = "mp4v",
    fps_fallback: float = 30.0,
) -> Dict[str, Any]:
    """
    Undistort a single video using camera intrinsics + distortion coefficients from a JSON file.

    Distortion model:
    - "standard": OpenCV radial/tangential model (cv2.undistort / initUndistortRectifyMap)
    - "fisheye": OpenCV fisheye model (cv2.fisheye.*), expects 4 coeffs (uses first 4 if longer)
    - "auto": choose "fisheye" if dist has length 4, else "standard"

    Args:
        input_video_path: Path to source video.
        output_video_path: Path to output video.
        intrinsic_json_path: Path to intrinsic json (expects fields "intrinsic" and "distortion_coefficients").
        scale: Optional scale factor applied to K (use if video resolution differs from calibration).
        balance: Undistortion balance / alpha in [0,1] (0 crops to valid region more; 1 keeps more FOV).
        distortion_model: "standard" | "fisheye" | "auto".
        overwrite: If False, skip if output already exists.
        codec: FourCC string for output encoding (default: "mp4v").
        fps_fallback: FPS used if input FPS is missing/invalid.

    Returns:
        Dict with basic metadata, e.g. {"status": "processed"|"skipped", "output": str, ...}.
    """
    # Keep this module lightweight: import heavy deps only when needed.
    from pathlib import Path

    import cv2  # type: ignore

    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    intr_path = Path(intrinsic_json_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input video not found: {in_path}")
    if not intr_path.is_file():
        raise FileNotFoundError(f"Intrinsics json not found: {intr_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return {"status": "skipped", "output": str(out_path)}

    K_np, dist_np = read_camera_intrinsics(str(intr_path), scale=scale)
    K_np = np.asarray(K_np, dtype=np.float64)
    dist_np = np.asarray(dist_np, dtype=np.float64).reshape(-1)
    # dist_np = np.array([dist_np[0], dist_np[1], dist_np[4], dist_np[2], dist_np[3]])

    distortion_model = str(distortion_model).lower().strip()
    if distortion_model not in {"standard", "fisheye", "auto"}:
        raise ValueError(f"Unknown distortion_model={distortion_model!r} (expected 'standard'|'fisheye'|'auto').")
    use_fisheye = (distortion_model == "fisheye") or (distortion_model == "auto" and dist_np.size == 4)

    # NOTE: You had some in-place debugging modifications (e.g. zeroing coefficients).
    # Keep those ONLY for the standard model to avoid corrupting fisheye parameters.
    if not use_fisheye:
        dist_np[0:2] = 0

    W, H = 960, 540  # image size
    pts = np.array([
        [823, 197],
        [W - 1, 0],
        [0, H - 1],
        [W - 1, H - 1],
        [W / 2, H / 2],
    ], dtype=np.float64).reshape(-1, 1, 2)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    writer = None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 1e-6:
            fps = float(fps_fallback)

        dim = (w, h)
        roi = (0, 0, w, h)
        new_K = None
        map1_std = map2_std = None
        map1_fish = map2_fish = None
        if dist_np.size != 0:
            if use_fisheye:
                dist4 = dist_np[:4].reshape(4, 1)
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K_np, dist4, dim, np.eye(3), balance=float(balance), new_size=dim
                )
                map1_fish, map2_fish = cv2.fisheye.initUndistortRectifyMap(
                    K_np, dist4, np.eye(3), new_K, dim, cv2.CV_16SC2
                )
                # Approx ROI via undistorted corners (axis-aligned bbox).
                corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64).reshape(-1, 1, 2)
                c_und = cv2.fisheye.undistortPoints(corners, K_np, dist4, P=new_K).reshape(-1, 2)
                x0, y0 = np.floor(np.min(c_und, axis=0)).astype(int)
                x1, y1 = np.ceil(np.max(c_und, axis=0)).astype(int)
                x0 = max(0, min(int(x0), w - 1))
                y0 = max(0, min(int(y0), h - 1))
                x1 = max(0, min(int(x1), w))
                y1 = max(0, min(int(y1), h))
                roi = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))
            else:
                # Standard OpenCV undistortion (matches the simple cv2.undistort snippet).
                new_K, roi = cv2.getOptimalNewCameraMatrix(
                    K_np, dist_np, dim, alpha=float(balance), newImgSize=dim
                )
                # Optional: precompute map for speed (also used for big-canvas debug below).
                map1_std, map2_std = cv2.initUndistortRectifyMap(
                    K_np, dist_np, None, new_K, dim, cv2.CV_16SC2
                )

        out_w, out_h = w, h
        if crop_to_roi and roi is not None:
            x, y, rw, rh = [int(v) for v in roi]
            if rw > 0 and rh > 0:
                out_w, out_h = rw, rh

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output writer: {out_path}")

        n_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if dist_np.size == 0:
                und = frame
            else:
                if use_fisheye:
                    und = cv2.remap(
                        frame,
                        map1_fish,
                        map2_fish,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                else:
                    # Use remap if available (faster & consistent), otherwise fall back.
                    if map1_std is not None and map2_std is not None:
                        und = cv2.remap(
                            frame,
                            map1_std,
                            map2_std,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                        )
                    else:
                        und = cv2.undistort(frame, K_np, dist_np, None, new_K)

            # --- debug: render onto a larger canvas so "outside" pixels are visible ---
            # IMPORTANT: `initUndistortRectifyMap(..., size=...)` controls the output size.
            # If you pass (W,H) here, `und_big` will still be W×H no matter what you do to newK2.
            H0, W0 = frame.shape[:2]
            s = 2
            W2, H2 = int(W0 * s), int(H0 * s)

            # Use a new camera matrix for the *larger output size*.
            if dist_np.size == 0:
                und_big = frame
            else:
                if use_fisheye:
                    dist4 = dist_np[:4].reshape(4, 1)
                    newK_big = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        K_np, dist4, (W0, H0), np.eye(3), balance=1.0, new_size=(W2, H2)
                    )
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                        K_np, dist4, np.eye(3), newK_big, (W2, H2), cv2.CV_16SC2
                    )
                else:
                    newK_big, roi_big = cv2.getOptimalNewCameraMatrix(
                        K_np, dist_np, (W0, H0), alpha=1.0, newImgSize=(W2, H2)
                    )
                    map1, map2 = cv2.initUndistortRectifyMap(
                        K_np, dist_np, None, newK_big, (W2, H2), cv2.CV_16SC2
                    )
                und_big = cv2.remap(
                    frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            und_rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
            und_big_rgb = cv2.cvtColor(und_big, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].imshow(frame_rgb)
            ax[0].set_title("Before (original)")
            ax[0].axis("off")

            ax[1].imshow(und_big_rgb)
            ax[1].set_title("After (undistorted)")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()

            if crop_to_roi and roi is not None:
                x, y, rw, rh = [int(v) for v in roi]
                if rw > 0 and rh > 0:
                    und = und[y : y + rh, x : x + rw]
            writer.write(und)
            n_frames += 1

        return {
            "status": "processed",
            "output": str(out_path),
            "frames": int(n_frames),
            "fps": float(fps),
            "size": [int(out_w), int(out_h)],
            "roi": [int(v) for v in roi] if roi is not None else None,
        }
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass


def undistort_videos_folder(
    input_folder: str,
    output_folder: str,
    camera_params_folder: str,
    *,
    video_exts: Tuple[str, ...] = (".mp4",),
    camera_name_prefix: str = "cam",
    intrinsic_json_template: str = "intrinsic_{cam_id}.json",
    scale: float = 1.0,
    balance: float = 0.0,
    distortion_model: str = "fisheye",
    overwrite: bool = False,
    codec: str = "mp4v",
) -> Dict[str, Any]:
    """
    Undistort all videos under a folder with structure:
      <input_folder>/<camX>/*.mp4
    and write them to <output_folder> while preserving the directory structure.

    Camera intrinsics/distortion are loaded from JSON files under <camera_params_folder>,
    using `intrinsic_json_template` (default: "intrinsic_{cam_id}.json").

    Distortion model:
    - Uses the standard OpenCV distortion model (radial/tangential).

    Args:
        input_folder: Root folder containing per-camera subfolders (e.g., cam2, cam4, ...).
        output_folder: Root folder to write undistorted videos into (mirrors input structure).
        camera_params_folder: Folder containing per-camera intrinsic JSON files.
        video_exts: Video file extensions to process.
        camera_name_prefix: Prefix for camera subfolders (default: "cam").
        intrinsic_json_template: Template for intrinsic json filename. Must contain "{cam_id}".
        scale: Optional scale factor applied to K (use if videos are resized vs calibration).
        balance: Undistortion balance / alpha in [0,1] (0 crops to valid region more; 1 keeps more FOV).
        overwrite: If False, skip outputs that already exist.
        codec: FourCC string for output encoding (default: "mp4v").

    Returns:
        A small stats dict: {"processed": int, "skipped": int, "failed": int, "outputs": [..]}.
    """
    # Keep this module lightweight: import heavy deps only when needed.
    import re
    from pathlib import Path

    in_root = Path(input_folder)
    out_root = Path(output_folder)
    cam_params_root = Path(camera_params_folder)

    stats: Dict[str, Any] = {"processed": 0, "skipped": 0, "failed": 0, "outputs": []}

    if not in_root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_root}")
    if not cam_params_root.is_dir():
        raise FileNotFoundError(f"Camera params folder not found: {cam_params_root}")

    # Find camera subfolders like cam2, cam4, ...
    cam_dirs = sorted([p for p in in_root.iterdir() if p.is_dir() and p.name.startswith(camera_name_prefix)])
    if not cam_dirs:
        raise FileNotFoundError(f"No camera subfolders like '{camera_name_prefix}*' found under: {in_root}")

    cam_id_re = re.compile(rf"^{re.escape(camera_name_prefix)}(\d+)$")

    for cam_dir in cam_dirs:
        m = cam_id_re.match(cam_dir.name)
        if not m:
            # Ignore non-matching folders (e.g. camX_extra)
            continue
        cam_id = m.group(1)
        # Only process the requested cameras explicitly.
        # if cam_id not in {"6"}:
        #     continue

        intrinsic_json = cam_params_root / "parameters-camera-04.json"

        # Process videos inside this camera folder (non-recursive, per the requested structure).
        for vid_path in sorted([p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]):
            rel = vid_path.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue
            try:
                res = undistort_video(
                    input_video_path=str(vid_path),
                    output_video_path=str(out_path),
                    intrinsic_json_path=str(intrinsic_json),
                    scale=float(scale),
                    balance=float(balance),
                    distortion_model=str(distortion_model),
                    overwrite=bool(overwrite),
                    codec=str(codec),
                )
                if res.get("status") == "processed":
                    stats["processed"] += 1
                    stats["outputs"].append(str(out_path))
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["failed"] += 1

    return stats


def sanity_check_camera_intrinsics_and_distortion(
    *,
    intrinsic_json_path: Optional[str] = None,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    image_size: Tuple[int, int],
    scale: float = 1.0,
    alpha: float = 0.0,
    sample_points_px: Optional[np.ndarray] = None,
    explode_thresh_px: float = 2000.0,
) -> Dict[str, Any]:
    """
    Quick sanity checks for camera intrinsics K and distortion coefficients.

    Intended use: detect obvious mismatches (wrong camera json, wrong scale/resolution,
    wrong coefficient order) BEFORE running full video undistortion.

    Args:
        intrinsic_json_path: Path to a json with keys "intrinsic" and "distortion_coefficients".
        K: Optional 3x3 intrinsic matrix (overrides intrinsic_json_path if provided).
        dist: Optional distortion array (overrides intrinsic_json_path if provided).
        image_size: (W, H) of the images/videos you will undistort.
        scale: If loading from json, apply the same scaling as `read_camera_intrinsics`.
        alpha: Passed to `cv2.getOptimalNewCameraMatrix` (0=crop more, 1=keep more FOV).
        sample_points_px: Optional (N,2) or (N,1,2) pixel points to test. If None, uses corners+center.
        explode_thresh_px: If undistorted pixel coords go beyond this magnitude, flag as "exploding".

    Returns:
        Dict with:
          - "K", "dist" (jsonable)
          - "metrics": focal/pp/fov estimates, coefficient magnitudes
          - "undistort_points": original vs undistorted (normalized and pixel with P=newK)
          - "warnings": list[str]
    """
    import cv2  # type: ignore

    W, H = int(image_size[0]), int(image_size[1])
    if W <= 0 or H <= 0:
        raise ValueError(f"Invalid image_size={image_size}; expected positive (W,H).")

    warnings: List[str] = []

    if K is None or dist is None:
        if intrinsic_json_path is None:
            raise ValueError("Provide either (intrinsic_json_path) or (K and dist).")
        K_np, dist_np = read_camera_intrinsics(intrinsic_json_path, scale=scale)
    else:
        K_np, dist_np = K, dist

    K_np = np.asarray(K_np, dtype=np.float64)
    dist_np = np.asarray(dist_np, dtype=np.float64).reshape(-1)

    # ---- Basic K checks ----
    if K_np.shape != (3, 3):
        warnings.append(f"K has shape {tuple(K_np.shape)}, expected (3,3).")
    else:
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        skew = float(K_np[0, 1])
        if not np.isfinite([fx, fy, cx, cy, skew]).all():
            warnings.append("K contains non-finite values.")
        if fx <= 0 or fy <= 0:
            warnings.append(f"Non-positive focal length: fx={fx}, fy={fy}.")
        if abs(float(K_np[2, 2]) - 1.0) > 1e-2:
            warnings.append(f"K[2,2]={float(K_np[2,2])} (expected ~1).")
        if abs(skew) > 1e-2:
            warnings.append(f"Non-zero skew K[0,1]={skew} (expected ~0).")
        if not (-0.5 * W <= cx <= 1.5 * W) or not (-0.5 * H <= cy <= 1.5 * H):
            warnings.append(f"Principal point looks off-image: cx={cx}, cy={cy} for W,H={W},{H}.")

    # ---- Distortion heuristic checks ----
    if dist_np.size == 0:
        warnings.append("distortion_coefficients is empty (no distortion will be applied).")
    elif dist_np.size == 5:
        # OpenCV order for 5: [k1,k2,p1,p2,k3]
        k1, k2, p1, p2, k3 = [float(x) for x in dist_np.tolist()]
        # Heuristic: tangential terms are usually small relative to radial terms.
        if abs(p1) > 0.05 or abs(p2) > 0.05:
            warnings.append(
                f"Tangential terms look large for OpenCV order: p1={p1}, p2={p2}. "
                f"If undistortion looks crazy, check coefficient order."
            )
        if abs(k3) > 1.0:
            warnings.append(f"|k3| is large ({k3}); undistortion may be very sensitive to scale/order.")
    else:
        warnings.append(
            f"distortion_coefficients has length {int(dist_np.size)}; this checker assumes standard OpenCV model. "
            f"Ensure your distortion model matches OpenCV."
        )

    # ---- FOV estimate (rough) ----
    metrics: Dict[str, Any] = {}
    if K_np.shape == (3, 3) and float(K_np[0, 0]) > 0 and float(K_np[1, 1]) > 0:
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        fov_x = float(2.0 * np.arctan((W / 2.0) / fx) * 180.0 / np.pi)
        fov_y = float(2.0 * np.arctan((H / 2.0) / fy) * 180.0 / np.pi)
        metrics.update(
            {
                "fx": fx,
                "fy": fy,
                "cx": float(K_np[0, 2]),
                "cy": float(K_np[1, 2]),
                "fov_x_deg_est": fov_x,
                "fov_y_deg_est": fov_y,
            }
        )

    # ---- undistortPoints mapping checks ----
    if sample_points_px is None:
        pts = np.array(
            [[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [W / 2.0, H / 2.0]],
            dtype=np.float64,
        )
    else:
        pts = np.asarray(sample_points_px, dtype=np.float64)
        if pts.ndim == 3 and pts.shape[1:] == (1, 2):
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("sample_points_px must have shape (N,2) or (N,1,2).")

    pts_cv = pts.reshape(-1, 1, 2)

    und_norm = cv2.undistortPoints(pts_cv, K_np, dist_np)  # normalized camera coords
    new_K, roi = cv2.getOptimalNewCameraMatrix(K_np, dist_np, (W, H), alpha=float(alpha), newImgSize=(W, H))
    und_px = cv2.undistortPoints(pts_cv, K_np, dist_np, P=new_K)  # pixel coords in new_K image

    und_px_flat = und_px.reshape(-1, 2)
    if not np.isfinite(und_px_flat).all():
        warnings.append("undistortPoints produced non-finite pixel coordinates (NaN/Inf).")
    if np.max(np.abs(und_px_flat)) > float(explode_thresh_px):
        warnings.append(
            f"undistortPoints pixel coords exceed explode_thresh_px={explode_thresh_px}; "
            f"likely wrong camera json / wrong scale / wrong coefficient order."
        )

    # Pixel displacement magnitude (orig -> undistorted in pixel space)
    disp = und_px_flat - pts.reshape(-1, 2)
    disp_norm = np.linalg.norm(disp, axis=1)
    metrics.update(
        {
            "dist_len": int(dist_np.size),
            "dist_abs_max": float(np.max(np.abs(dist_np))) if dist_np.size else 0.0,
            "undistort_disp_px_max": float(np.max(disp_norm)) if disp_norm.size else 0.0,
            "undistort_disp_px_mean": float(np.mean(disp_norm)) if disp_norm.size else 0.0,
            "roi": [int(x) for x in roi] if roi is not None else None,
        }
    )

    return {
        "K": _to_jsonable(K_np),
        "dist": _to_jsonable(dist_np),
        "metrics": metrics,
        "undistort_points": {
            "points_px_in": _to_jsonable(pts),
            "points_norm_out": _to_jsonable(und_norm.reshape(-1, 2)),
            "points_px_out": _to_jsonable(und_px.reshape(-1, 2)),
            "new_K": _to_jsonable(new_K),
        },
        "warnings": warnings,
    }


if __name__ == "__main__":
    camera_path = "/home/zonghuan/tudelft/projects/datasets/modification/conflab"
    undistort_videos_folder(
        input_folder=os.path.join(camera_path, "segments"),
        output_folder=os.path.join(camera_path, "segments_undistorted"),
        camera_params_folder="./experiments/intrinsics",
        scale=0.5,
        balance=0,
        overwrite=True,
        codec="mp4v",
    )

    # input_folder = os.path.join(camera_path, "video_segments_test")
    # cam_id = 2
    # cam_dir = os.path.join(input_folder, f"cam{cam_id}")
    # vids = sorted(glob.glob(os.path.join(cam_dir, "*.mp4")))
    # if not vids:
    #     raise FileNotFoundError(f"No .mp4 found under {cam_dir}")

    # cap = cv2.VideoCapture(vids[0])
    # ok, frame = cap.read()
    # cap.release()
    # if not ok or frame is None:
    #     raise RuntimeError(f"Failed to read first frame from {vids[0]}")

    # H, W = frame.shape[:2]

    # res = sanity_check_camera_intrinsics_and_distortion(
    #     intrinsic_json_path=os.path.join(camera_path, "camera_params", f"intrinsic_{cam_id}.json"),
    #     image_size=(W, H),
    #     scale=0.5,   # use the same scale you use for undistortion
    #     alpha=0.0,
    # )

    # print(json.dumps(res["metrics"], indent=2))
    # print("warnings:", res["warnings"])
    # print("points_px_in:", res["undistort_points"]["points_px_in"])
    # print("points_px_out:", res["undistort_points"]["points_px_out"])