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


def load_extrinsics_json(path: str, device: torch.device) -> Extrinsics:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    R = torch.tensor(np.asarray(data["rotation"], dtype=np.float32), device=device)
    t = torch.tensor(np.asarray(data["translation"], dtype=np.float32).reshape(3), device=device)
    return Extrinsics(R=R, t=t)


@dataclass
class GroundOptConfig:
    iters: int = 200
    lr: float = 0.05
    lambda_prior: float = 0.2
    lambda_vel: float = 1.0
    lambda_plane: float = 5.0
    lambda_slide: float = 1.0
    contact_z_thresh: float = 0.03  # in *world units* (depends on your extrinsic unit)
    contact_v_xy_thresh: float = 0.05


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

    def world_feet(t: torch.Tensor) -> torch.Tensor:
        # (T,8,3) in world coordinates
        feet_cam = X3d[:, list(FOOT_IDXS), :] + t[:, None, :]
        feet_world = extr.cam_to_world(feet_cam.reshape(-1, 3)).view(T, len(FOOT_IDXS), 3)
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

    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt

        feet_w = world_feet(t)
        c = contact_mask(feet_w)  # (T,)

        # Plane loss: drive min foot z to 0 on contact frames
        min_z = feet_w[..., 2].min(dim=1).values  # (T,)
        l_plane = (c * (min_z ** 2)).sum() / (c.sum() + 1e-6)

        # Slide loss: keep the contact foot xy stable
        l_slide = torch.zeros((), device=device)
        if T > 1:
            # use min-foot point per frame
            z = feet_w[..., 2]
            min_idx = z.min(dim=1).indices
            xy = feet_w[torch.arange(T, device=device), min_idx, :2]  # (T,2)
            v = xy[1:] - xy[:-1]
            c_pair = (c[1:] * c[:-1])
            l_slide = (c_pair * (v * v).sum(dim=1)).sum() / (c_pair.sum() + 1e-6)

        # Smoothness + prior
        l_prior = ((t - t0) ** 2).sum(dim=-1).mean()
        if T > 1:
            v = t[1:] - t[:-1]
            l_vel = (v * v).sum(dim=-1).mean()
        else:
            l_vel = t.sum() * 0.0

        loss = (
            float(cfg.lambda_plane) * l_plane
            + float(cfg.lambda_slide) * l_slide
            + float(cfg.lambda_prior) * l_prior
            + float(cfg.lambda_vel) * l_vel
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_opt = (t0 + dt).detach()
        metrics = {
            "loss_plane": float(l_plane.detach().cpu().item()),
            "loss_slide": float(l_slide.detach().cpu().item()),
            "loss_prior": float(l_prior.detach().cpu().item()),
            "loss_vel": float(l_vel.detach().cpu().item()),
        }
    return t_opt, metrics

