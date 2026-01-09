from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .obs_kps import ObsKps


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    d = float(delta)
    return torch.where(x <= d, 0.5 * x * x, d * (x - 0.5 * d))


def project_points(K: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    K: (3,3)
    X: (...,3) camera coords
    returns: (...,2) pixels
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z = X[..., 2].clamp(min=1e-6)
    u = fx * (X[..., 0] / z) + cx
    v = fy * (X[..., 1] / z) + cy
    return torch.stack([u, v], dim=-1)


@dataclass
class ReprojOptConfig:
    iters: int = 200
    lr: float = 0.05
    huber_delta_px: float = 10.0
    lambda_prior: float = 0.1
    lambda_vel: float = 1.0
    lambda_accel: float = 0.5
    obs_scale_candidates: Sequence[float] = (1.0, 0.5, 2.0)


def _reproj_loss(
    K: torch.Tensor,
    X3d: torch.Tensor,  # (T,70,3)
    t: torch.Tensor,    # (T,3)
    obs_by_t: List[Optional[ObsKps]],
    obs_scale: float,
    huber_delta_px: float,
) -> torch.Tensor:
    device = X3d.device
    T = X3d.shape[0]
    loss = torch.zeros((), device=device)
    count = 0
    for ti in range(T):
        obs = obs_by_t[ti]
        if obs is None:
            continue
        pts = X3d[ti][obs.kp_idx] + t[ti].view(1, 3)
        uv = project_points(K, pts)
        resid = uv - (obs.xy * float(obs_scale))
        r = torch.sqrt((resid * resid).sum(dim=-1) + 1e-8)
        loss = loss + huber(r, huber_delta_px).mean()
        count += 1
    return loss / float(max(count, 1))


def _smoothness_loss(t: torch.Tensor, lambda_accel: float) -> torch.Tensor:
    # velocity + optional acceleration
    T = t.shape[0]
    if T <= 1:
        return t.sum() * 0.0
    v = t[1:] - t[:-1]
    l = (v * v).sum(dim=-1).mean()
    if lambda_accel > 0.0 and T >= 3:
        a = t[2:] - 2.0 * t[1:-1] + t[:-2]
        l = l + float(lambda_accel) * (a * a).sum(dim=-1).mean()
    return l


def pick_best_obs_scale(
    K: torch.Tensor,
    X3d: torch.Tensor,
    t0: torch.Tensor,
    obs_by_t: List[Optional[ObsKps]],
    cfg: ReprojOptConfig,
) -> float:
    best_s = float(cfg.obs_scale_candidates[0]) if cfg.obs_scale_candidates else 1.0
    best = float("inf")
    with torch.no_grad():
        for s in cfg.obs_scale_candidates:
            l = _reproj_loss(
                K=K, X3d=X3d, t=t0, obs_by_t=obs_by_t,
                obs_scale=float(s), huber_delta_px=float(cfg.huber_delta_px),
            )
            lv = float(l.detach().cpu().item())
            if lv < best:
                best = lv
                best_s = float(s)
    return best_s


def optimize_pred_cam_t(
    K: torch.Tensor,
    X3d: torch.Tensor,           # (T,70,3)
    t0: torch.Tensor,            # (T,3)
    obs_by_t: List[Optional[ObsKps]],
    cfg: ReprojOptConfig,
    obs_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Optimize t(t) (camera translation time-series) given 3D keypoints and observed 2D keypoints.
    Returns:
      t_opt (T,3), metrics
    """
    device = X3d.device
    t0 = t0.to(device=device, dtype=torch.float32)
    if obs_scale is None:
        obs_scale = pick_best_obs_scale(K, X3d, t0, obs_by_t, cfg)

    dt = torch.zeros_like(t0, requires_grad=True)
    opt = torch.optim.Adam([dt], lr=float(cfg.lr))

    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt
        l_reproj = _reproj_loss(K, X3d, t, obs_by_t, float(obs_scale), float(cfg.huber_delta_px))
        l_prior = ((t - t0) ** 2).sum(dim=-1).mean()
        l_smooth = _smoothness_loss(t, float(cfg.lambda_accel))
        loss = l_reproj + float(cfg.lambda_prior) * l_prior + float(cfg.lambda_vel) * l_smooth
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_opt = (t0 + dt).detach()
        metrics = {
            "obs_scale": float(obs_scale),
            "loss_reproj": float(_reproj_loss(K, X3d, t_opt, obs_by_t, float(obs_scale), float(cfg.huber_delta_px)).cpu().item()),
            "loss_prior": float((((t_opt - t0) ** 2).sum(dim=-1).mean()).cpu().item()),
            "loss_vel": float((((t_opt[1:] - t_opt[:-1]) ** 2).sum(dim=-1).mean()).cpu().item()) if t_opt.shape[0] > 1 else 0.0,
        }
    return t_opt, metrics

