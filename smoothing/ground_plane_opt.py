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
        Rt = self.R.transpose(0, 1)
        return (X_cam - self.t.view(1, 3)) @ Rt


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
    iters: int = 200
    lr: float = 0.05
    lambda_prior: float = 0.2
    lambda_vel: float = 1.0
    lambda_accel: float = 2.0  # acceleration penalty (2nd order smoothness, reduces jitter)
    lambda_plane: float = 5.0
    lambda_slide: float = 1.0
    # Contact detection thresholds (used for slide loss, and plane loss if always_grounded=False)
    contact_z_thresh: float = 3.0  # in world units (cm if world_scale=100); frames where min_foot_z < this are "contact"
    contact_v_xy_thresh: float = 5.0  # in world units (cm if world_scale=100); low XY velocity threshold for contact
    world_scale: float = 100.0  # scale factor to convert SMPL-X meters to world units (100 for cm)
    # If True (default), always push the lowest foot to ground (assumes humans are always standing)
    # If False, only apply ground constraint when contact is detected
    always_grounded: bool = True


FOOT_IDXS = (13, 14, 15, 16, 17, 18, 19, 20)  # ankles + toes + heels in mhr70


def optimize_ground_plane_translation(
    *,
    extr: Extrinsics,
    X3d: torch.Tensor,      # (T,70,3) local keypoints (before translation)
    t0: torch.Tensor,       # (T,3) initial pred_cam_t
    cfg: GroundOptConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Optimize translation time-series t(t) with a world-ground constraint:
      - encourage foot points to lie on plane z=0 during contact
      - discourage sliding in world XY during contact

    Note: This only optimizes translation, not pose.
    """
    device = X3d.device
    T = X3d.shape[0]
    t0 = t0.to(device=device, dtype=torch.float32)

    dt = torch.zeros_like(t0, requires_grad=True)
    opt = torch.optim.Adam([dt], lr=float(cfg.lr))
    world_scale = float(cfg.world_scale)

    def world_feet(t: torch.Tensor) -> torch.Tensor:
        # (T,8,3) in world coordinates
        feet_cam = X3d[:, list(FOOT_IDXS), :] + t[:, None, :]
        # Scale from SMPL-X meters to world units (e.g., cm) before world transformation
        feet_cam_scaled = feet_cam * world_scale
        feet_world = extr.cam_to_world(feet_cam_scaled.reshape(-1, 3)).view(T, len(FOOT_IDXS), 3)
        return feet_world

    def contact_mask(feet_world: torch.Tensor) -> torch.Tensor:
        # heuristics: near ground and low xy velocity for minimum-foot point
        z = feet_world[..., 2]  # (T,8)
        min_z, min_idx = z.min(dim=1)  # (T,)
        xy = feet_world[torch.arange(T, device=device), min_idx, :2]  # (T,2)
        if T > 1:
            v = torch.cat([torch.zeros((1, 2), device=device), xy[1:] - xy[:-1]], dim=0)
            vmag = torch.norm(v, dim=1)
        else:
            vmag = torch.zeros((T,), device=device)
        c = (min_z.abs() < float(cfg.contact_z_thresh)) & (vmag < float(cfg.contact_v_xy_thresh))
        return c.float()  # (T,)

    always_grounded = bool(cfg.always_grounded)
    
    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt

        feet_w = world_feet(t)
        
        # Min foot z per frame
        min_z = feet_w[..., 2].min(dim=1).values  # (T,)
        
        if always_grounded:
            # Always push the lowest foot toward ground (z=0) for ALL frames
            # This assumes humans are always standing with at least one foot on ground
            l_plane = (min_z ** 2).mean()
            
            # For slide loss, use frames where feet are actually close to ground
            c = contact_mask(feet_w)
        else:
            # Only apply ground constraint when contact is detected (legacy mode)
            c = contact_mask(feet_w)  # (T,)
            l_plane = (c * (min_z ** 2)).sum() / (c.sum() + 1e-6)

        # Slide loss: keep the contact foot xy stable (only when near ground)
        l_slide = torch.zeros((), device=device)
        if T > 1:
            # use min-foot point per frame
            z = feet_w[..., 2]
            min_idx = z.min(dim=1).indices
            xy = feet_w[torch.arange(T, device=device), min_idx, :2]  # (T,2)
            v = xy[1:] - xy[:-1]
            c_pair = (c[1:] * c[:-1])
            l_slide = (c_pair * (v * v).sum(dim=1)).sum() / (c_pair.sum() + 1e-6)

        # Smoothness + prior — scale to world units so that lambdas are
        # comparable to l_plane / l_slide (which are already in world-unit²).
        # Without this, prior/vel losses are in meters² while plane/slide are
        # in cm², making the regularisation ~world_scale² ≈ 10 000× too weak.
        l_prior = (((t - t0) * world_scale) ** 2).sum(dim=-1).mean()
        if T > 1:
            v = (t[1:] - t[:-1]) * world_scale
            l_vel = (v * v).sum(dim=-1).mean()
        else:
            l_vel = t.sum() * 0.0
        
        # Acceleration penalty (2nd order smoothness — reduces jitter)
        l_accel = t.sum() * 0.0
        if T >= 3 and float(cfg.lambda_accel) > 0.0:
            a = (t[2:] - 2.0 * t[1:-1] + t[:-2]) * world_scale
            l_accel = (a * a).sum(dim=-1).mean()

        loss = (
            float(cfg.lambda_plane) * l_plane
            + float(cfg.lambda_slide) * l_slide
            + float(cfg.lambda_prior) * l_prior
            + float(cfg.lambda_vel) * l_vel
            + float(cfg.lambda_accel) * l_accel
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_opt = (t0 + dt).detach()
        # Compute final feet positions for metrics
        feet_w_final = world_feet(t_opt)
        min_z_final = feet_w_final[..., 2].min(dim=1).values
        c_final = contact_mask(feet_w_final)
        metrics = {
            "loss_plane": float(l_plane.detach().cpu().item()),
            "loss_slide": float(l_slide.detach().cpu().item()),
            "loss_prior": float(l_prior.detach().cpu().item()),
            "loss_vel": float(l_vel.detach().cpu().item()),
            "loss_accel": float(l_accel.detach().cpu().item()),
            "min_foot_z_mean": float(min_z_final.mean().item()),
            "min_foot_z_std": float(min_z_final.std().item()),
            "contact_frames": int(c_final.sum().item()),
            "total_frames": T,
            "always_grounded": always_grounded,
        }
    return t_opt, metrics

