"""
Mask-based reprojection optimization.

Instead of using external bbox/keypoints, this optimizes pred_cam_t to maximize
the overlap between the projected mesh silhouette and the ground truth mask from Stage 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class MaskReprojOptConfig:
    iters: int = 200
    lr: float = 0.01
    lambda_vertex_in_mask: float = 1.0  # projected vertices should be inside mask
    lambda_mask_coverage: float = 0.5   # mask pixels should be covered by projected mesh
    lambda_prior: float = 0.05          # don't deviate too much from initial
    lambda_vel: float = 0.5             # temporal smoothness (1st order: penalise velocity)
    lambda_accel: float = 2.0           # temporal smoothness (2nd order: penalise acceleration / jitter)
    num_sample_vertices: int = 500      # sample vertices for efficiency (0 = use all)
    num_sample_mask_points: int = 200   # mask points to sample for coverage loss (0 = disable)
    mask_scale: float = 1.0             # scale factor if masks are at different resolution


def load_masks_for_sequence(
    mask_dir: str,
    frame_names: List[str],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Load mask PNGs for a sequence.
    
    Args:
        mask_dir: Directory containing mask PNGs (frame_name.png)
        frame_names: List of frame names (e.g., ["00000000", "00000001", ...])
        device: Torch device
    
    Returns:
        Dict mapping frame_index -> mask tensor (H, W) with pixel values = obj_id
    """
    masks = {}
    for ti, fname in enumerate(frame_names):
        mask_path = os.path.join(mask_dir, f"{fname}.png")
        if not os.path.exists(mask_path):
            continue
        
        # Load as palette image to get obj_ids. Palette indices are 0-255, so uint8
        # is exact and a quarter of int32 (all frames are kept on the device).
        mask_img = Image.open(mask_path).convert('P')
        mask_np = np.array(mask_img, dtype=np.uint8)
        masks[ti] = torch.from_numpy(mask_np).to(device)
    
    return masks


def project_vertices_to_2d(
    vertices: torch.Tensor,  # (V, 3) in camera space
    K: torch.Tensor,         # (3, 3) intrinsic matrix
) -> torch.Tensor:
    """
    Project 3D vertices to 2D pixel coordinates.
    
    Returns:
        (V, 2) tensor of pixel coordinates (x, y)
    """
    # Project: x = K @ X
    v_proj = (K @ vertices.T).T  # (V, 3)
    # Perspective division
    v_2d = v_proj[:, :2] / (v_proj[:, 2:3] + 1e-8)  # (V, 2)
    return v_2d


def _build_soft_mask(
    mask: torch.Tensor,
    target_id: int,
    blur_radius: int = 5,
) -> torch.Tensor:
    """
    Build a soft mask from an integer mask.
    The binary mask is blurred slightly so that bilinear sampling at edge
    pixels produces non-zero gradients w.r.t. the sampling coordinates.
    """
    binary = (mask == target_id).float()
    if blur_radius <= 0 or not binary.any():
        return binary
    ks = 2 * blur_radius + 1
    kernel = torch.ones(1, 1, ks, ks, device=mask.device) / (ks * ks)
    soft = F.conv2d(binary.unsqueeze(0).unsqueeze(0), kernel, padding=blur_radius)
    return soft.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _sample_soft_mask(
    soft_mask: torch.Tensor,
    points_2d: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiably sample a soft mask at 2D points using grid_sample.

    Gradients propagate through *points_2d* into the upstream projection
    and camera translation, so the optimizer can drive vertices toward the
    mask interior.

    Args:
        soft_mask: (H, W) float in [0, 1]
        points_2d: (N, 2) pixel coordinates (x, y)
    Returns:
        (N,) sampled values in [0, 1] with gradients w.r.t. points_2d
    """
    H, W = soft_mask.shape
    grid_x = 2.0 * points_2d[:, 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * points_2d[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    soft_4d = soft_mask.unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        soft_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return sampled.reshape(-1)


def _build_soft_mask_counts(
    masks: List[torch.Tensor],
    target_ids: List[int],
    blur_radius: int = 5,
    chunk: int = 64,
) -> torch.Tensor:
    """
    Batched _build_soft_mask, stored compactly: (F, H, W) uint8 counts of
    target pixels in each (2r+1)^2 box. Divided by (2r+1)^2 this is exactly
    the soft mask, at a quarter of float32's memory.
    """
    ks = 2 * blur_radius + 1
    assert ks * ks <= 255, "blur_radius too large for uint8 counts"
    H, W = masks[0].shape
    device = masks[0].device
    out = torch.empty(len(masks), H, W, dtype=torch.uint8, device=device)
    kernel = torch.ones(1, 1, ks, ks, device=device)
    for s in range(0, len(masks), chunk):
        ids = torch.tensor(target_ids[s:s + chunk], device=device).view(-1, 1, 1)
        binary = (torch.stack(masks[s:s + chunk]) == ids).float().unsqueeze(1)
        counts = F.conv2d(binary, kernel, padding=blur_radius) if blur_radius > 0 else binary
        out[s:s + chunk] = counts.squeeze(1).round().to(torch.uint8)
    return out


def _sample_soft_masks_batched(
    counts: torch.Tensor,     # (F, H, W) uint8 from _build_soft_mask_counts
    divisor: float,           # (2r+1)^2
    points_2d: torch.Tensor,  # (F, P, 2) pixel coordinates (x, y)
) -> torch.Tensor:
    """
    Bilinear sampling of frame f's soft mask at points_2d[f], the same as
    _sample_soft_mask (grid_sample, align_corners=True, zero padding).
    Gathers only the 4 neighbours of each point, so no float copy of the
    masks is made; gradients flow through the bilinear weights.
    Returns (F, P).
    """
    n_f, H, W = counts.shape
    flat = counts.reshape(-1)
    x, y = points_2d[..., 0], points_2d[..., 1]
    x0, y0 = x.detach().floor(), y.detach().floor()
    wx1, wy1 = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    base = (torch.arange(n_f, device=counts.device) * (H * W)).view(-1, 1)
    out = torch.zeros_like(x)
    for dx, dy, w in ((0, 0, (1 - wx1) * (1 - wy1)), (1, 0, wx1 * (1 - wy1)),
                      (0, 1, (1 - wx1) * wy1), (1, 1, wx1 * wy1)):
        xi, yi = x0 + dx, y0 + dy
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = base + yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)
        out = out + w * flat[idx].to(x.dtype) * inside
    return out / divisor


def sample_mask_at_points(
    mask: torch.Tensor,
    points_2d: torch.Tensor,
    target_id: int,
) -> torch.Tensor:
    """Non-differentiable hard mask check (kept for logging/metrics)."""
    H, W = mask.shape
    x = points_2d[:, 0].detach().clamp(0, W - 1).long()
    y = points_2d[:, 1].detach().clamp(0, H - 1).long()
    return (mask[y, x] == target_id)


def _sample_points_from_mask(
    mask: torch.Tensor,
    target_id: int,
    num_points: int,
) -> Optional[torch.Tensor]:
    """
    Uniformly sample 2D pixel coordinates from the mask region for a given ID.

    Returns (num_points, 2) float tensor of (x, y) pixel coords, or None if the
    mask region is empty.
    """
    ys, xs = (mask == target_id).nonzero(as_tuple=True)
    if len(xs) == 0:
        return None
    if num_points >= len(xs):
        idx = torch.arange(len(xs), device=mask.device)
    else:
        idx = torch.randperm(len(xs), device=mask.device)[:num_points]
    return torch.stack([xs[idx].float(), ys[idx].float()], dim=-1)


def optimize_mask_reprojection(
    *,
    K: torch.Tensor,           # (3, 3) intrinsic matrix
    vertices_local: torch.Tensor,  # (T, V, 3) mesh vertices in local space
    t0: torch.Tensor,          # (T, 3) initial pred_cam_t
    masks: Dict[int, torch.Tensor],  # frame_idx -> (H, W) mask
    obj_id: int,               # actual person ID (for logging)
    mask_ids: Optional[Dict[int, int]] = None,  # frame_idx -> consecutive mask pixel ID (if None, use obj_id)
    cfg: MaskReprojOptConfig = MaskReprojOptConfig(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Optimize pred_cam_t to maximize overlap between projected mesh and mask.
    
    Args:
        K: Camera intrinsic matrix
        vertices_local: Mesh vertices before translation
        t0: Initial camera translation per frame
        masks: Ground truth masks indexed by frame
        obj_id: The actual person ID (for logging / fallback)
        mask_ids: Per-frame mapping from frame_idx to the consecutive mask pixel ID
                  for this person. If None, obj_id is used as mask pixel ID for all frames.
        cfg: Optimization config
    
    Returns:
        Optimized translation (T, 3) and metrics dict
    """
    device = vertices_local.device
    T, V, _ = vertices_local.shape
    
    # Sample vertices for efficiency
    if cfg.num_sample_vertices > 0 and cfg.num_sample_vertices < V:
        sample_idx = torch.randperm(V, device=device)[:cfg.num_sample_vertices]
        verts_sampled = vertices_local[:, sample_idx, :]
    else:
        verts_sampled = vertices_local
    
    # Optimization variable
    dt = torch.zeros_like(t0, requires_grad=True)
    opt = torch.optim.Adam([dt], lr=float(cfg.lr))
    
    # Frame indices that have masks AND a known mask pixel ID for this person
    valid_frames = sorted([ti for ti in masks.keys() if ti < T])
    if mask_ids is not None:
        valid_frames = [ti for ti in valid_frames if ti in mask_ids]
    if not valid_frames:
        print(f"[WARN] No valid masks found for obj_id={obj_id}, skipping optimization")
        return t0, {"skipped": True}
    
    # All frames are handled at once: per iteration the loss is a few batched
    # ops over (frames, sampled vertices) instead of a Python loop over frames.
    target_ids = [mask_ids[ti] if mask_ids is not None else obj_id for ti in valid_frames]
    fidx = torch.tensor(valid_frames, device=device)
    verts_v = verts_sampled[fidx]  # (F, P, 3)

    # Soft masks (blurred, for differentiable sampling) as uint8 box counts
    blur_radius = 5
    soft_div = float((2 * blur_radius + 1) ** 2)
    soft_counts = _build_soft_mask_counts([masks[ti] for ti in valid_frames], target_ids, blur_radius)
    H, W = soft_counts.shape[1:]
    diag_sq = float(H * H + W * W)

    # Pre-sample 2D points from each mask region for the coverage loss.
    # These are fixed across iterations (re-sampling each iter is too noisy);
    # the loss only uses their centroid and bbox extent.
    use_coverage = float(cfg.lambda_mask_coverage) > 0.0 and cfg.num_sample_mask_points > 0
    has_mask_pts = torch.zeros(len(valid_frames), dtype=torch.bool, device=device)
    m_center = torch.zeros(len(valid_frames), 2, device=device)
    m_ext = torch.zeros(len(valid_frames), 2, device=device)
    if use_coverage:
        for i, ti in enumerate(valid_frames):
            pts = _sample_points_from_mask(masks[ti], target_ids[i], cfg.num_sample_mask_points)
            if pts is not None:
                has_mask_pts[i] = True
                m_center[i] = pts.mean(dim=0)
                m_ext[i] = pts.max(dim=0).values - pts.min(dim=0).values

    # Approximate pixel-scale factor so regularisation terms are comparable to the
    # pixel-space data loss. Without this, vel/accel in SMPL-X metres are ~10^6
    # smaller and effectively ignored.
    focal = float(max(K[0, 0], K[1, 1]).item())
    z_mean = float(t0[:, 2].mean().clamp(min=0.5).item())
    pix_scale = focal / z_mean

    best_loss = float("inf")
    best_dt = dt.detach().clone()

    for it in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt

        # Project the sampled vertices of every valid frame: (F, P, ...)
        v_cam = verts_v + t[fidx].unsqueeze(1)
        in_front = v_cam[..., 2] > 0.1
        v_proj = v_cam @ K.T
        z = torch.where(in_front, v_proj[..., 2], torch.ones_like(v_proj[..., 2]))
        v_2d = v_proj[..., :2] / (z.unsqueeze(-1) + 1e-8)
        if cfg.mask_scale != 1.0:
            v_2d = v_2d * cfg.mask_scale
        valid = in_front & (v_2d[..., 0] >= 0) & (v_2d[..., 0] < W) & \
                (v_2d[..., 1] >= 0) & (v_2d[..., 1] < H)
        v_2d = torch.where(valid.unsqueeze(-1), v_2d, torch.zeros_like(v_2d))
        n_valid = valid.sum(dim=1)
        frame_ok = n_valid > 0
        n_frames = int(frame_ok.sum().item())
        if n_frames == 0:
            continue
        validf = valid.to(v_2d.dtype)
        n_valid_f = n_valid.clamp(min=1).to(v_2d.dtype)

        sampled_vals = _sample_soft_masks_batched(soft_counts, soft_div, v_2d)
        mean_inside = (sampled_vals * validf).sum(dim=1) / n_valid_f
        loss_vertex_in_mask = (1.0 - mean_inside)[frame_ok].sum() / n_frames

        # Coverage: projected mesh should match the mask in position and
        # scale.  Uses centroid + bbox-extent matching instead of the full
        # pairwise distance matrix (torch.cdist) which blows up the
        # autograd graph when accumulated over hundreds of frames.
        loss_coverage = torch.tensor(0.0, device=device)
        cov_ok = frame_ok & has_mask_pts
        n_coverage = int(cov_ok.sum().item())
        if n_coverage > 0:
            v_center = (v_2d * validf.unsqueeze(-1)).sum(dim=1) / n_valid_f.unsqueeze(-1)
            big = torch.finfo(v_2d.dtype).max
            vmask = valid.unsqueeze(-1)
            v_max = torch.where(vmask, v_2d, torch.full_like(v_2d, -big)).max(dim=1).values
            v_min = torch.where(vmask, v_2d, torch.full_like(v_2d, big)).min(dim=1).values
            # Frames without valid points have +-big here; zero them so their
            # (masked-out) loss terms can't produce inf/NaN gradients
            fok = frame_ok.unsqueeze(-1)
            v_max = torch.where(fok, v_max, torch.zeros_like(v_max))
            v_min = torch.where(fok, v_min, torch.zeros_like(v_min))
            cov_centroid = ((v_center - m_center) ** 2).sum(dim=-1) / diag_sq
            cov_extent = ((v_max - v_min - m_ext) ** 2).sum(dim=-1) / diag_sq
            loss_coverage = (cov_centroid + cov_extent)[cov_ok].sum() / n_coverage

        # Scale regularisation to pixel-equivalent units
        loss_prior = ((t - t0) ** 2).sum(dim=-1).mean() * (pix_scale ** 2)

        loss_vel = torch.tensor(0.0, device=device)
        if T > 1:
            v = t[1:] - t[:-1]
            loss_vel = (v * v).sum(dim=-1).mean() * (pix_scale ** 2)

        loss_accel = torch.tensor(0.0, device=device)
        if T >= 3 and float(cfg.lambda_accel) > 0.0:
            a = t[2:] - 2.0 * t[1:-1] + t[:-2]
            loss_accel = (a * a).sum(dim=-1).mean() * (pix_scale ** 2)

        loss = (
            float(cfg.lambda_vertex_in_mask) * loss_vertex_in_mask
            + float(cfg.lambda_mask_coverage) * loss_coverage
            + float(cfg.lambda_prior) * loss_prior
            + float(cfg.lambda_vel) * loss_vel
            + float(cfg.lambda_accel) * loss_accel
        )

        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_dt = dt.detach().clone()

    # Compute final hard metrics for logging
    with torch.no_grad():
        t_opt = (t0 + best_dt).detach()
        hard_inside = 0.0
        hard_n = 0
        for ti in valid_frames:
            target_id = mask_ids[ti] if mask_ids is not None else obj_id
            v_cam = verts_sampled[ti] + t_opt[ti:ti+1]
            in_front = v_cam[:, 2] > 0.1
            if not in_front.any():
                continue
            v_2d = project_vertices_to_2d(v_cam[in_front], K)
            if cfg.mask_scale != 1.0:
                v_2d = v_2d * cfg.mask_scale
            H, W = masks[ti].shape
            in_bounds = (v_2d[:, 0] >= 0) & (v_2d[:, 0] < W) & \
                        (v_2d[:, 1] >= 0) & (v_2d[:, 1] < H)
            if not in_bounds.any():
                continue
            inside = sample_mask_at_points(masks[ti], v_2d[in_bounds], target_id)
            hard_inside += inside.float().mean().item()
            hard_n += 1

        metrics = {
            "loss_vertex_in_mask_soft": float(loss_vertex_in_mask.item()) if n_frames > 0 else 0.0,
            "loss_coverage": float(loss_coverage.item()) if n_coverage > 0 else 0.0,
            "hard_inside_ratio": hard_inside / max(hard_n, 1),
            "loss_prior": float(loss_prior.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_accel": float(loss_accel.item()),
            "valid_frames": n_frames,
            "best_loss": best_loss,
        }

    return t_opt, metrics
