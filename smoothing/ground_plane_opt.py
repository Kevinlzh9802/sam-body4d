from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Extrinsics:
    """
    OpenCV solvePnP convention:
      X_cam = R * X_world + t
    R: (3,3)
    t: (3,)
    """

    R: torch.Tensor
    t: torch.Tensor

    def cam_to_world(self, X_cam: torch.Tensor) -> torch.Tensor:
        """
        X_world = R^T (X_cam - t)
        X_cam: (...,3)
        """
        # Row vectors: R^T x == x @ R
        return (X_cam - self.t.view(1, 3)) @ self.R


def _find_key(data: Dict, keys: Tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float32).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)

    k = r / theta
    kx, ky, kz = [float(v) for v in k]
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    ).astype(np.float32)


def load_extrinsics_json(path: str, device: torch.device) -> Extrinsics:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rotation_data = _find_key(data, ("rotation", "R", "rotation_matrix"))
    if rotation_data is not None:
        R_np = np.asarray(rotation_data, dtype=np.float32).reshape(3, 3)
    else:
        rvec_data = _find_key(data, ("rvec", "rvec_wc", "rotation_vector"))
        if rvec_data is None:
            raise KeyError(
                f"Extrinsics JSON {path!r} must contain either 'rotation'/'R' "
                "or 'rvec'/'rotation_vector'."
            )
        R_np = _rodrigues_to_matrix(np.asarray(rvec_data, dtype=np.float32))

    t_data = _find_key(data, ("translation", "t", "tvec", "tvec_wc", "translation_vector"))
    camera_center_data = _find_key(data, ("camera_center_world_m", "camera_center_world"))
    if t_data is None and camera_center_data is not None:
        camera_center = np.asarray(camera_center_data, dtype=np.float32).reshape(3)
        t_np = -(R_np @ camera_center)
    elif t_data is None:
        raise KeyError(
            f"Extrinsics JSON {path!r} must contain 'translation', 't', 'tvec', "
            "or 'camera_center_world_m'."
        )
    else:
        t_np = np.asarray(t_data, dtype=np.float32).reshape(3)

    extrinsic_type = str(data.get("extrinsic_type", "world_to_camera")).lower()
    if extrinsic_type in {"world_to_camera", "world2camera", "w2c", "opencv"}:
        pass
    elif extrinsic_type in {"camera_to_world", "camera2world", "c2w"}:
        R_cw = R_np
        t_cw = t_np
        R_np = R_cw.T
        t_np = -(R_np @ t_cw)
    else:
        raise ValueError(
            f"Unsupported extrinsic_type={extrinsic_type!r} in {path!r}. "
            "Expected 'world_to_camera' or 'camera_to_world'."
        )

    R = torch.tensor(R_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    return Extrinsics(R=R, t=t)


@dataclass
class GroundOptConfig:
    """Ground step of Stage 3 (optimize_ground_ray_scale)."""
    iters: int = 200
    lambda_plane: float = 5.0   # lowest foot point on the floor z=0
    lambda_prior: float = 0.2   # per-frame shift along the ray stays small
    lambda_vel: float = 1.0     # ... and smooth (1st order)
    lambda_accel: float = 0.5   # ... and smooth (2nd order)
    world_scale: float = 100.0  # world units per metre (100 = cm, as the extrinsics)
    huber_delta: float = 10.0   # Huber plane loss (world units), so frames with no foot down weigh less
    ray_lr: float = 0.01        # Adam lr for log(scale) and the per-frame ray factor


FOOT_IDXS = (13, 14, 15, 16, 17, 18, 19, 20)  # ankles + toes + heels in mhr70


def optimize_ground_ray_scale(
    *,
    extr: Extrinsics,
    X3d: torch.Tensor,            # (P,70,3) local keypoints of one person's frames (before translation)
    t0: torch.Tensor,             # (P,3) pred_cam_t of those frames
    frame_numbers: torch.Tensor,  # (P,) their frame numbers (smoothness only across consecutive ones)
    cfg: GroundOptConfig,
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    """
    Ground constraint that keeps the person where they are in the image.

    Camera-space body: x = s * (X_local + (1 + k_t) * t0_t)
      s    one scale per person about the camera centre. Scaling a camera-space
           mesh about the camera centre leaves its projection unchanged: this is
           the depth/size ambiguity a single view cannot resolve, and the floor
           resolves it.
      k_t  small per-frame shift along the viewing ray (changes the apparent
           size by k_t, not the position), for per-frame floor contact.
    Losses: lowest foot point on z=0 (Huber), prior + velocity + acceleration
    on the ray shift, the latter two across consecutive frames only.

    Returns pred_cam_t (P,3) = s (1 + k) t0, s (the caller must multiply the
    person's local vertices/keypoints by it) and metrics.
    """
    import torch.nn.functional as F

    device = X3d.device
    ws = float(cfg.world_scale)
    tp = t0.detach().to(device=device, dtype=torch.float32)  # (P,3)
    if len(tp) == 0:
        return tp.clone(), 1.0, {"skipped": True}

    feet = X3d[:, list(FOOT_IDXS), :].float()        # (P,8,3)
    fn = frame_numbers.to(device)
    pair = (fn[1:] - fn[:-1]) == 1                   # consecutive frames
    trip = pair[1:] & pair[:-1]
    # World z of a camera point x: (x - t) @ R[:, 2]
    up = extr.R[:, 2].float()
    z_off = float(extr.t.float().view(3) @ up)
    ray_len = tp.norm(dim=1) * ws                    # (P,) camera-to-root distance, world units

    def feet_z(log_s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        cam = log_s.exp() * (feet + (1.0 + k)[:, None, None] * tp[:, None, :])
        return (cam * ws) @ up - z_off               # (P,8)

    # Start from the median per-frame scale that puts the lowest foot on the floor
    with torch.no_grad():
        a = ((feet + tp[:, None, :]) * ws @ up).min(dim=1).values  # lowest foot z at s=1 is a - z_off
        s_t = z_off / a
        s_t = s_t[torch.isfinite(s_t) & (s_t > 0)]
        s0 = float(s_t.median().clamp(0.5, 2.0)) if len(s_t) else 1.0

    log_s = torch.tensor(float(np.log(s0)), device=device, requires_grad=True)
    k = torch.zeros(len(tp), device=device, requires_grad=True)
    opt = torch.optim.Adam([log_s, k], lr=float(cfg.ray_lr))
    zero = torch.zeros((), device=device)

    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        min_z = feet_z(log_s, k).min(dim=1).values
        l_plane = F.huber_loss(min_z, torch.zeros_like(min_z), delta=float(cfg.huber_delta))
        d = k * log_s.exp() * ray_len                # shift along the ray, world units
        l_prior = (d * d).mean()
        l_vel = ((d[1:] - d[:-1])[pair] ** 2).mean() if bool(pair.any()) else zero
        l_accel = ((d[2:] - 2.0 * d[1:-1] + d[:-2])[trip] ** 2).mean() if bool(trip.any()) else zero
        loss = (
            float(cfg.lambda_plane) * l_plane
            + float(cfg.lambda_prior) * l_prior
            + float(cfg.lambda_vel) * l_vel
            + float(cfg.lambda_accel) * l_accel
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        s = float(log_s.exp())
        t_out = s * (1.0 + k)[:, None] * tp
        min_z = feet_z(log_s, k).min(dim=1).values
        d = k * s * ray_len
        metrics = {
            "mode": "ray",
            "scale": s,
            "scale_init": s0,
            "min_foot_z_mean": float(min_z.mean()),
            "min_foot_z_std": float(min_z.std()) if len(min_z) > 1 else 0.0,
            "ray_shift_rms": float(d.pow(2).mean().sqrt()),
            "loss_plane": float(l_plane),
            "loss_prior": float(l_prior),
            "loss_vel": float(l_vel),
            "loss_accel": float(l_accel),
            "total_frames": int(len(tp)),
        }
    return t_out, s, metrics
