"""Plot 2D projections (bird's eye view) of keypoints on the ground plane."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

try:
    from .ground_plane_opt import Extrinsics, FOOT_IDXS
except ImportError:
    # Allow running as standalone script
    from ground_plane_opt import Extrinsics, FOOT_IDXS


# Keypoint indices for body parts (mhr70 format)
HEAD_IDXS = (0,)  # nose
PELVIS_IDXS = (9, 10)  # left-hip, right-hip (midpoint = pelvis center)
# FOOT_IDXS already imported: (13, 14, 15, 16, 17, 18, 19, 20)


def compute_keypoint_projections_world(
    *,
    keypoints3d_local: torch.Tensor,  # (T*N, K, 3) local keypoints
    pred_cam_t: torch.Tensor,          # (T*N, 3) camera translation
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    world_scale: float = 100.0,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute world-space keypoint positions for head, pelvis, and feet.
    
    Args:
        world_scale: Scale factor to convert SMPL-X meters to extrinsic units (default 100.0 for cm).
    
    Returns:
        Dict mapping obj_id -> {
            'frames': np.ndarray of frame indices,
            'head_xy': (num_frames, 2) array of head XY positions,
            'pelvis_xy': (num_frames, 2) array of pelvis XY positions,
            'feet_xy': (num_frames, 2) array of feet center XY positions,
        }
    """
    device = keypoints3d_local.device
    K = keypoints3d_local.shape[1]
    
    # Clamp indices to available keypoints
    head_idxs = [i for i in HEAD_IDXS if i < K]
    pelvis_idxs = [i for i in PELVIS_IDXS if i < K]
    foot_idxs = [i for i in FOOT_IDXS if i < K]
    
    results: Dict[int, Dict[str, List]] = {
        oid: {'frames': [], 'head_xy': [], 'pelvis_xy': [], 'feet_xy': []}
        for oid in obj_ids_all
    }
    
    for ti in range(T):
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            
            bi = ti * N + si
            kps_local = keypoints3d_local[bi]  # (K, 3)
            camt = pred_cam_t[bi]  # (3,)
            
            # Transform to camera space and scale to extrinsic units
            kps_cam = kps_local + camt.view(1, 3)
            kps_cam_scaled = kps_cam * world_scale
            kps_world = extr.cam_to_world(kps_cam_scaled)  # (K, 3)
            
            # Extract XY (ground plane projection) for each body part
            if head_idxs:
                head_world = kps_world[head_idxs].mean(dim=0)
                results[oid]['head_xy'].append(head_world[:2].cpu().numpy())
            
            if pelvis_idxs:
                pelvis_world = kps_world[pelvis_idxs].mean(dim=0)
                results[oid]['pelvis_xy'].append(pelvis_world[:2].cpu().numpy())
            
            if foot_idxs:
                feet_world = kps_world[foot_idxs].mean(dim=0)
                results[oid]['feet_xy'].append(feet_world[:2].cpu().numpy())
            
            results[oid]['frames'].append(ti)
    
    # Convert to numpy arrays
    output = {}
    for oid, data in results.items():
        if not data['frames']:
            continue
        output[oid] = {
            'frames': np.array(data['frames']),
            'head_xy': np.array(data['head_xy']) if data['head_xy'] else np.zeros((0, 2)),
            'pelvis_xy': np.array(data['pelvis_xy']) if data['pelvis_xy'] else np.zeros((0, 2)),
            'feet_xy': np.array(data['feet_xy']) if data['feet_xy'] else np.zeros((0, 2)),
        }
    
    return output


def plot_2d_projection_single_frame(
    projections: Dict[int, Dict[str, np.ndarray]],
    frame_idx: int,
    output_path: str,
    title: Optional[str] = None,
) -> bool:
    """
    Plot 2D projection of all people at a specific frame.
    
    Returns True if plot was created, False if no data for this frame.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color map for different people
    cmap = plt.cm.get_cmap('tab20')
    num_persons = len(projections)
    
    has_data = False
    legend_handles = []
    
    for idx, (oid, data) in enumerate(sorted(projections.items())):
        frames = data['frames']
        # Find index in this person's data for the requested frame
        frame_mask = frames == frame_idx
        if not frame_mask.any():
            continue
        
        has_data = True
        fi = np.where(frame_mask)[0][0]
        color = cmap(idx / max(num_persons, 1))
        
        # Plot head (triangle marker)
        if len(data['head_xy']) > fi:
            hx, hy = data['head_xy'][fi]
            ax.scatter(hx, hy, marker='^', s=150, color=color, edgecolors='black', linewidths=0.5, zorder=3)
        
        # Plot pelvis (circle marker)
        if len(data['pelvis_xy']) > fi:
            px, py = data['pelvis_xy'][fi]
            ax.scatter(px, py, marker='o', s=100, color=color, edgecolors='black', linewidths=0.5, zorder=3)
        
        # Plot feet (square marker)
        if len(data['feet_xy']) > fi:
            fx, fy = data['feet_xy'][fi]
            ax.scatter(fx, fy, marker='s', s=80, color=color, edgecolors='black', linewidths=0.5, zorder=3)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=color, label=f'ID {oid}'))
    
    if not has_data:
        plt.close(fig)
        return False
    
    # Add marker legend
    marker_handles = [
        plt.Line2D([0], [0], marker='^', color='gray', linestyle='', markersize=10, label='Head'),
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=10, label='Pelvis'),
        plt.Line2D([0], [0], marker='s', color='gray', linestyle='', markersize=10, label='Feet'),
    ]
    
    ax.set_xlabel('X (world units)')
    ax.set_ylabel('Y (world units)')
    ax.set_title(title or f'2D Projection (Frame {frame_idx})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Two legends: one for people, one for markers
    legend1 = ax.legend(handles=legend_handles, loc='upper left', fontsize='small', title='People')
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc='upper right', fontsize='small', title='Body Parts')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_2d_projections_sequence(
    projections: Dict[int, Dict[str, np.ndarray]],
    output_dir: str,
    frame_interval: int = 100,
    title_prefix: str = "2D Projection",
) -> int:
    """
    Plot 2D projections at regular frame intervals.
    
    Args:
        projections: Output from compute_keypoint_projections_world
        output_dir: Directory to save plots
        frame_interval: Plot every N frames (default: 100)
        title_prefix: Prefix for plot titles
    
    Returns:
        Number of plots created
    """
    if not projections:
        print("[WARN] No projection data to plot")
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find frame range across all people
    all_frames = np.concatenate([d['frames'] for d in projections.values()])
    min_frame = int(all_frames.min())
    max_frame = int(all_frames.max())
    
    # Generate frame indices at intervals
    frame_indices = list(range(min_frame, max_frame + 1, frame_interval))
    if max_frame not in frame_indices:
        frame_indices.append(max_frame)
    
    num_plots = 0
    for fi in frame_indices:
        output_path = os.path.join(output_dir, f"projection_frame_{fi:06d}.png")
        if plot_2d_projection_single_frame(
            projections=projections,
            frame_idx=fi,
            output_path=output_path,
            title=f"{title_prefix} (Frame {fi})",
        ):
            num_plots += 1
    
    print(f"[INFO] Created {num_plots} 2D projection plots in: {output_dir}")
    return num_plots


def plot_2d_projections_from_stage3(
    *,
    keypoints3d_local: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    output_dir: str,
    frame_interval: int = 100,
    world_scale: float = 100.0,
) -> None:
    """
    Convenience function to compute and plot 2D projections in one call.
    Called from smooth_mhr_and_export_meshes.py after stage 3 processing.
    
    Args:
        world_scale: Scale factor to convert SMPL-X meters to extrinsic units (default 100.0 for cm).
    """
    projections = compute_keypoint_projections_world(
        keypoints3d_local=keypoints3d_local,
        pred_cam_t=pred_cam_t,
        extr=extr,
        T=T,
        N=N,
        obj_ids_all=obj_ids_all,
        frame_obj_ids_slots=frame_obj_ids_slots,
        world_scale=world_scale,
    )
    
    plot_2d_projections_sequence(
        projections=projections,
        output_dir=output_dir,
        frame_interval=frame_interval,
        title_prefix="2D Projection (World Space)",
    )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def compute_projections_from_meshes(
    mesh_dir: str,
    frame_interval: int = 100,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute 2D projections from mesh PLY files.
    Uses specific vertex positions as approximations for body parts.
    
    Only loads PLY files at the specified frame interval for efficiency.
    
    Note: This is a rough approximation since we don't have keypoints,
    only mesh vertices. Uses centroid and extremal vertices.
    """
    import trimesh
    
    if not os.path.isdir(mesh_dir):
        raise ValueError(f"Mesh directory not found: {mesh_dir}")
    
    results: Dict[int, Dict[str, List]] = {}
    
    # Find all person subdirectories
    person_dirs = []
    for name in os.listdir(mesh_dir):
        subdir = os.path.join(mesh_dir, name)
        if os.path.isdir(subdir):
            try:
                obj_id = int(name)
                person_dirs.append((obj_id, subdir))
            except ValueError:
                continue
    
    if not person_dirs:
        print(f"[WARN] No person subdirectories found in {mesh_dir}")
        return {}
    
    print(f"[INFO] Found {len(person_dirs)} persons in mesh folder")
    
    for obj_id, person_dir in sorted(person_dirs):
        all_ply_files = sorted(
            [f for f in os.listdir(person_dir) if f.lower().endswith(".ply")],
            key=_natural_sort_key
        )
        
        if not all_ply_files:
            continue
        
        # Filter to only load files at frame_interval
        # First, extract frame indices from all filenames to determine which to load
        ply_files_to_load = []
        for ply_name in all_ply_files:
            frame_match = re.search(r"(\d+)", ply_name)
            if frame_match:
                frame_idx = int(frame_match.group(1))
                if frame_idx % frame_interval == 0:
                    ply_files_to_load.append((ply_name, frame_idx))
        
        # Also include the last frame if not already included
        if all_ply_files:
            last_ply = all_ply_files[-1]
            last_match = re.search(r"(\d+)", last_ply)
            if last_match:
                last_frame = int(last_match.group(1))
                if last_frame % frame_interval != 0:
                    ply_files_to_load.append((last_ply, last_frame))
        
        if not ply_files_to_load:
            continue
        
        print(f"  ID {obj_id}: loading {len(ply_files_to_load)} of {len(all_ply_files)} files")
        results[obj_id] = {'frames': [], 'head_xy': [], 'pelvis_xy': [], 'feet_xy': []}
        
        for ply_name, frame_idx in ply_files_to_load:
            ply_path = os.path.join(person_dir, ply_name)
            try:
                mesh = trimesh.load(ply_path, process=False)
                if hasattr(mesh, "geometry") and not hasattr(mesh, "vertices"):
                    geoms = list(mesh.geometry.values())
                    if geoms:
                        mesh = geoms[0]
                
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                
                # Approximate body parts from vertices:
                # - Head: highest Z vertices
                # - Pelvis: centroid
                # - Feet: lowest Z vertices
                z_coords = verts[:, 2]
                
                # Head: average of top 50 vertices by Z
                top_indices = np.argsort(z_coords)[-50:]
                head_xy = verts[top_indices, :2].mean(axis=0)
                
                # Pelvis: centroid XY
                pelvis_xy = verts[:, :2].mean(axis=0)
                
                # Feet: average of bottom 50 vertices by Z
                bottom_indices = np.argsort(z_coords)[:50]
                feet_xy = verts[bottom_indices, :2].mean(axis=0)
                
                results[obj_id]['frames'].append(frame_idx)
                results[obj_id]['head_xy'].append(head_xy)
                results[obj_id]['pelvis_xy'].append(pelvis_xy)
                results[obj_id]['feet_xy'].append(feet_xy)
                
            except Exception as e:
                print(f"[WARN] Failed to load {ply_path}: {e}")
                continue
    
    # Convert to numpy arrays
    output = {}
    for oid, data in results.items():
        if not data['frames']:
            continue
        output[oid] = {
            'frames': np.array(data['frames']),
            'head_xy': np.array(data['head_xy']),
            'pelvis_xy': np.array(data['pelvis_xy']),
            'feet_xy': np.array(data['feet_xy']),
        }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D projections (bird's eye view) of body keypoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from mesh folder
  python -m smoothing.projection_2d_plot --mesh-dir /path/to/meshes_4d_individual
  
  # Specify output directory and frame interval
  python -m smoothing.projection_2d_plot --mesh-dir ./meshes_4d_individual -o ./projections --interval 50
        """
    )
    parser.add_argument(
        "--mesh-dir", "-m", required=True,
        help="Path to meshes_4d_individual folder"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: projection_2d in parent of mesh-dir)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=100,
        help="Plot every N frames (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Compute projections from meshes (only load files at the interval)
    projections = compute_projections_from_meshes(
        mesh_dir=args.mesh_dir,
        frame_interval=args.interval,
    )
    
    if not projections:
        print("[ERROR] No data to plot")
        return 1
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        parent_dir = os.path.dirname(os.path.abspath(args.mesh_dir))
        output_dir = os.path.join(parent_dir, "projection_2d")
    
    # Plot
    plot_2d_projections_sequence(
        projections=projections,
        output_dir=output_dir,
        frame_interval=args.interval,
        title_prefix="2D Projection (World Space)",
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
