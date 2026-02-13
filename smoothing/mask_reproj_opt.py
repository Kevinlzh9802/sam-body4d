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
    iters: int = 150
    lr: float = 0.02
    lambda_vertex_in_mask: float = 1.0  # projected vertices should be inside mask
    lambda_mask_coverage: float = 0.5   # mask should be covered by projection
    lambda_prior: float = 0.1           # don't deviate too much from initial
    lambda_vel: float = 0.5             # temporal smoothness
    num_sample_vertices: int = 500      # sample vertices for efficiency (0 = use all)
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
        
        # Load as palette image to get obj_ids
        mask_img = Image.open(mask_path).convert('P')
        mask_np = np.array(mask_img, dtype=np.int32)
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


def sample_mask_at_points(
    mask: torch.Tensor,      # (H, W) integer mask
    points_2d: torch.Tensor, # (N, 2) pixel coordinates
    target_id: int,
) -> torch.Tensor:
    """
    Check which 2D points fall inside the mask for a given object ID.
    Uses nearest-neighbor sampling (no interpolation for integer masks).
    
    Returns:
        (N,) boolean tensor: True if point is inside mask for target_id
    """
    H, W = mask.shape
    device = mask.device
    
    # Clamp to valid pixel range
    x = points_2d[:, 0].clamp(0, W - 1).long()
    y = points_2d[:, 1].clamp(0, H - 1).long()
    
    # Sample mask values
    sampled_ids = mask[y, x]
    
    # Check if inside target mask
    inside = (sampled_ids == target_id)
    return inside


def compute_mask_coverage(
    mask: torch.Tensor,      # (H, W) integer mask
    points_2d: torch.Tensor, # (N, 2) pixel coordinates
    target_id: int,
    grid_size: int = 32,
) -> torch.Tensor:
    """
    Estimate how well the projected points cover the mask.
    
    Divides the mask bounding box into a grid and checks coverage.
    
    Returns:
        Scalar coverage ratio (0 to 1)
    """
    H, W = mask.shape
    device = mask.device
    
    # Find mask bounding box
    mask_binary = (mask == target_id)
    if not mask_binary.any():
        return torch.tensor(0.0, device=device)
    
    ys, xs = torch.where(mask_binary)
    x_min, x_max = xs.min().item(), xs.max().item()
    y_min, y_max = ys.min().item(), ys.max().item()
    
    if x_max <= x_min or y_max <= y_min:
        return torch.tensor(0.0, device=device)
    
    # Create grid cells
    cell_w = (x_max - x_min) / grid_size
    cell_h = (y_max - y_min) / grid_size
    
    # Assign points to grid cells
    px = ((points_2d[:, 0] - x_min) / cell_w).clamp(0, grid_size - 1).long()
    py = ((points_2d[:, 1] - y_min) / cell_h).clamp(0, grid_size - 1).long()
    
    # Count covered cells
    covered = torch.zeros((grid_size, grid_size), device=device, dtype=torch.bool)
    valid = (points_2d[:, 0] >= x_min) & (points_2d[:, 0] <= x_max) & \
            (points_2d[:, 1] >= y_min) & (points_2d[:, 1] <= y_max)
    
    if valid.any():
        covered[py[valid], px[valid]] = True
    
    # Coverage ratio
    coverage = covered.float().mean()
    return coverage


def optimize_mask_reprojection(
    *,
    K: torch.Tensor,           # (3, 3) intrinsic matrix
    vertices_local: torch.Tensor,  # (T, V, 3) mesh vertices in local space
    t0: torch.Tensor,          # (T, 3) initial pred_cam_t
    masks: Dict[int, torch.Tensor],  # frame_idx -> (H, W) mask
    obj_id: int,               # object ID to match
    cfg: MaskReprojOptConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Optimize pred_cam_t to maximize overlap between projected mesh and mask.
    
    Args:
        K: Camera intrinsic matrix
        vertices_local: Mesh vertices before translation
        t0: Initial camera translation per frame
        masks: Ground truth masks indexed by frame
        obj_id: The object ID to match in the masks
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
    
    # Frame indices that have masks
    valid_frames = sorted([ti for ti in masks.keys() if ti < T])
    if not valid_frames:
        print(f"[WARN] No valid masks found for obj_id={obj_id}, skipping optimization")
        return t0, {"skipped": True}
    
    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        t = t0 + dt
        
        loss_vertex_in_mask = torch.tensor(0.0, device=device)
        loss_coverage = torch.tensor(0.0, device=device)
        n_frames = 0
        
        for ti in valid_frames:
            mask = masks[ti]
            H, W = mask.shape
            
            # Get vertices in camera space
            v_cam = verts_sampled[ti] + t[ti:ti+1]  # (V_sample, 3)
            
            # Filter vertices in front of camera
            in_front = v_cam[:, 2] > 0.1
            if not in_front.any():
                continue
            v_cam_front = v_cam[in_front]
            
            # Project to 2D
            v_2d = project_vertices_to_2d(v_cam_front, K)
            
            # Scale if mask is at different resolution
            if cfg.mask_scale != 1.0:
                v_2d = v_2d * cfg.mask_scale
            
            # Filter to image bounds
            in_bounds = (v_2d[:, 0] >= 0) & (v_2d[:, 0] < W) & \
                        (v_2d[:, 1] >= 0) & (v_2d[:, 1] < H)
            if not in_bounds.any():
                continue
            v_2d_valid = v_2d[in_bounds]
            
            # Loss 1: Vertices should project inside the mask
            inside = sample_mask_at_points(mask, v_2d_valid, obj_id)
            # Use soft loss: proportion inside (higher is better, so minimize 1 - ratio)
            inside_ratio = inside.float().mean()
            loss_vertex_in_mask = loss_vertex_in_mask + (1.0 - inside_ratio)
            
            # Loss 2: Mask should be covered by projection
            coverage = compute_mask_coverage(mask, v_2d_valid, obj_id)
            loss_coverage = loss_coverage + (1.0 - coverage)
            
            n_frames += 1
        
        if n_frames == 0:
            continue
        
        loss_vertex_in_mask = loss_vertex_in_mask / n_frames
        loss_coverage = loss_coverage / n_frames
        
        # Prior: don't deviate too much from initial
        loss_prior = ((t - t0) ** 2).sum(dim=-1).mean()
        
        # Temporal smoothness
        loss_vel = torch.tensor(0.0, device=device)
        if T > 1:
            v = t[1:] - t[:-1]
            loss_vel = (v * v).sum(dim=-1).mean()
        
        loss = (
            float(cfg.lambda_vertex_in_mask) * loss_vertex_in_mask
            + float(cfg.lambda_mask_coverage) * loss_coverage
            + float(cfg.lambda_prior) * loss_prior
            + float(cfg.lambda_vel) * loss_vel
        )
        
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        t_opt = (t0 + dt).detach()
        metrics = {
            "loss_vertex_in_mask": float(loss_vertex_in_mask.item()) if n_frames > 0 else 0.0,
            "loss_coverage": float(loss_coverage.item()) if n_frames > 0 else 0.0,
            "loss_prior": float(loss_prior.item()),
            "loss_vel": float(loss_vel.item()),
            "valid_frames": n_frames,
        }
    
    return t_opt, metrics
