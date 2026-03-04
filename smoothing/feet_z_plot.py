"""Plot average z-coordinate of feet in world coordinates across frames."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

try:
    from .ground_plane_opt import Extrinsics, FOOT_IDXS
except ImportError:
    # Allow running as standalone script
    from ground_plane_opt import Extrinsics, FOOT_IDXS

# MHR70 keypoint indices (NOT SMPL ordering — see metadata/mhr70.py)
NOSE_IDX = 0
HIP_IDXS = (9, 10)        # left_hip, right_hip (proxy for pelvis)
NECK_IDX = 69
# FOOT_IDXS = (13..20) already defined in ground_plane_opt


def compute_feet_z_world(
    *,
    keypoints3d_local: torch.Tensor,  # (T*N, K, 3) local keypoints
    pred_cam_t: torch.Tensor,          # (T*N, 3) camera translation
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    world_scale: float = 100.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute average feet z-coordinate in world space for each person across frames.
    
    Args:
        world_scale: Scale factor to convert SMPL-X meters to extrinsic units (default 100.0 for cm).
    
    Returns:
        Dict mapping obj_id -> (frame_indices, avg_feet_z_values)
        Only frames where the person is present are included.
    """
    device = keypoints3d_local.device
    K = keypoints3d_local.shape[1]
    
    # Get foot indices (clamp to available keypoints)
    foot_idxs = [i for i in FOOT_IDXS if i < K]
    if not foot_idxs:
        print("[WARN] No foot keypoints available in the keypoint set")
        return {}
    
    results: Dict[int, Tuple[List[int], List[float]]] = {oid: ([], []) for oid in obj_ids_all}
    
    for ti in range(T):
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            
            bi = ti * N + si
            # Get foot keypoints in local (model) space
            feet_local = keypoints3d_local[bi, foot_idxs, :]  # (num_feet, 3)
            camt = pred_cam_t[bi]  # (3,)
            
            # Transform to camera space
            feet_cam = feet_local + camt.view(1, 3)
            
            # Scale to extrinsic units (e.g., meters -> centimeters)
            feet_cam_scaled = feet_cam * world_scale
            
            # Transform to world space
            feet_world = extr.cam_to_world(feet_cam_scaled)  # (num_feet, 3)
            
            # Average z coordinate of all foot keypoints
            avg_z = feet_world[:, 2].mean().item()
            
            results[oid][0].append(ti)
            results[oid][1].append(avg_z)
    
    # Convert to numpy arrays
    return {
        oid: (np.array(frames), np.array(zvals))
        for oid, (frames, zvals) in results.items()
        if frames  # Only include persons with at least one frame
    }


def plot_feet_z_world(
    feet_z_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Average Feet Z-Coordinate (World Space)",
) -> None:
    """
    Plot average feet z-coordinate for each person across frames.
    
    Args:
        feet_z_data: Dict from compute_feet_z_world
        output_path: Path to save the plot image
        title: Plot title
    """
    if not feet_z_data:
        print("[WARN] No feet z data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use a colormap to distinguish different persons
    cmap = plt.cm.get_cmap('tab20')
    num_persons = len(feet_z_data)
    
    for idx, (oid, (frames, zvals)) in enumerate(sorted(feet_z_data.items())):
        color = cmap(idx / max(num_persons, 1))
        ax.plot(frames, zvals, label=f'ID {oid}', color=color, alpha=0.8, linewidth=1)
    
    # Add horizontal line at z=0 (ground plane)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Ground (z=0)')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Average Feet Z (world units)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotation
    all_z = np.concatenate([zvals for _, (_, zvals) in feet_z_data.items()])
    stats_text = f'Overall: min={all_z.min():.3f}, max={all_z.max():.3f}, mean={all_z.mean():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Feet z-coordinate plot saved to: {output_path}")


def plot_feet_z_from_stage3(
    *,
    keypoints3d_local: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    output_path: str,
    title: Optional[str] = None,
    world_scale: float = 100.0,
) -> None:
    """
    Convenience function to compute and plot feet z-coordinates in one call.
    Called from smooth_mhr_and_export_meshes.py after stage 3 processing.
    
    Args:
        world_scale: Scale factor to convert SMPL-X meters to extrinsic units (default 100.0 for cm).
    """
    feet_z_data = compute_feet_z_world(
        keypoints3d_local=keypoints3d_local,
        pred_cam_t=pred_cam_t,
        extr=extr,
        T=T,
        N=N,
        obj_ids_all=obj_ids_all,
        frame_obj_ids_slots=frame_obj_ids_slots,
        world_scale=world_scale,
    )
    
    plot_feet_z_world(
        feet_z_data=feet_z_data,
        output_path=output_path,
        title=title or "Average Feet Z-Coordinate (World Space)",
    )


def _compute_landmark_z_world(
    *,
    keypoints3d_local: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    world_scale: float,
    joint_indices: List[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Compute mean world-z of *joint_indices* per person per frame."""
    device = keypoints3d_local.device
    K = keypoints3d_local.shape[1]
    idxs = [i for i in joint_indices if i < K]
    if not idxs:
        return {}
    results: Dict[int, Tuple[List[int], List[float]]] = {oid: ([], []) for oid in obj_ids_all}
    for ti in range(T):
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            pts = keypoints3d_local[bi, idxs, :] + pred_cam_t[bi].view(1, 3)
            pts_world = extr.cam_to_world(pts * world_scale)
            results[oid][0].append(ti)
            results[oid][1].append(float(pts_world[:, 2].mean().item()))
    return {oid: (np.array(f), np.array(z)) for oid, (f, z) in results.items() if f}


def plot_body_landmarks_z(
    *,
    keypoints3d_local: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr: Extrinsics,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    world_scale: float = 100.0,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot nose, hips, neck, and feet world-z on a single multi-subplot figure."""
    groups = {
        "Nose (joint 0)": [NOSE_IDX],
        "Neck (joint 69)": [NECK_IDX],
        "Hips — pelvis proxy (joints 9,10)": list(HIP_IDXS),
        "Feet — ankles/toes/heels (joints 13-20)": list(FOOT_IDXS),
    }
    data_per_group: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for label, idxs in groups.items():
        data_per_group[label] = _compute_landmark_z_world(
            keypoints3d_local=keypoints3d_local, pred_cam_t=pred_cam_t,
            extr=extr, T=T, N=N, obj_ids_all=obj_ids_all,
            frame_obj_ids_slots=frame_obj_ids_slots,
            world_scale=world_scale, joint_indices=idxs,
        )

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("tab20")
    n_persons = max(len(obj_ids_all), 1)
    wu = "cm" if abs(world_scale - 100.0) < 1 else "world units"

    for ax, (label, person_data) in zip(axes, data_per_group.items()):
        for idx, (oid, (frames, zvals)) in enumerate(sorted(person_data.items())):
            color = cmap(idx / n_persons)
            ax.plot(frames, zvals, color=color, alpha=0.8, linewidth=1,
                    label=f"ID {oid} (mean {zvals.mean():.1f})")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Ground (z=0)")
        ax.set_ylabel(f"z ({wu})")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(title or "Body Landmark Z-Coordinates (World Space)", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Body landmark z-plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Standalone CLI: compute feet z from mesh folder
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally (e.g., frame_2 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def compute_feet_z_from_meshes(
    mesh_dir: str,
    num_lowest_verts: int = 50,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute average feet z-coordinate from mesh PLY files.
    
    Uses the N lowest vertices (by z-coordinate) as an approximation for feet.
    
    Args:
        mesh_dir: Path to meshes_4d_individual folder containing per-person subdirs
        num_lowest_verts: Number of lowest vertices to average for feet estimation
    
    Returns:
        Dict mapping obj_id -> (frame_indices, avg_feet_z_values)
    """
    if not os.path.isdir(mesh_dir):
        raise ValueError(f"Mesh directory not found: {mesh_dir}")
    
    results: Dict[int, Tuple[List[int], List[float]]] = {}
    
    # Find all person subdirectories (named by obj_id)
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
        ply_files = sorted(
            [f for f in os.listdir(person_dir) if f.lower().endswith(".ply")],
            key=_natural_sort_key
        )
        
        if not ply_files:
            print(f"[WARN] No PLY files for ID {obj_id}")
            continue
        
        frames: List[int] = []
        z_vals: List[float] = []
        
        for ply_name in ply_files:
            ply_path = os.path.join(person_dir, ply_name)
            try:
                mesh = trimesh.load(ply_path, process=False)
                if hasattr(mesh, "geometry") and not hasattr(mesh, "vertices"):
                    geoms = list(mesh.geometry.values())
                    if geoms:
                        mesh = geoms[0]
                
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                
                # Extract frame index from filename (e.g., "00000123.ply" -> 123)
                frame_match = re.search(r"(\d+)", ply_name)
                if frame_match:
                    frame_idx = int(frame_match.group(1))
                else:
                    frame_idx = len(frames)
                
                # Use N lowest vertices by z-coordinate as feet approximation
                z_coords = verts[:, 2]
                lowest_indices = np.argsort(z_coords)[:num_lowest_verts]
                avg_feet_z = z_coords[lowest_indices].mean()
                
                frames.append(frame_idx)
                z_vals.append(float(avg_feet_z))
                
            except Exception as e:
                print(f"[WARN] Failed to load {ply_path}: {e}")
                continue
        
        if frames:
            results[obj_id] = (np.array(frames), np.array(z_vals))
            print(f"  ID {obj_id}: {len(frames)} frames, z range [{min(z_vals):.3f}, {max(z_vals):.3f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot average feet z-coordinate from mesh folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from meshes_4d_individual folder
  python -m smoothing.feet_z_plot --mesh-dir /path/to/exp/meshes_4d_individual
  
  # Specify output path and customize
  python -m smoothing.feet_z_plot --mesh-dir ./meshes_4d_individual -o feet_z.png --num-verts 100
        """
    )
    parser.add_argument(
        "--mesh-dir", "-m", required=True,
        help="Path to meshes_4d_individual folder (contains per-person subdirs with PLY files)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output image path (default: feet_z_world.png in parent of mesh-dir)"
    )
    parser.add_argument(
        "--num-verts", "-n", type=int, default=50,
        help="Number of lowest vertices to use for feet estimation (default: 50)"
    )
    parser.add_argument(
        "--title", "-t", default=None,
        help="Plot title (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Compute feet z from meshes
    feet_z_data = compute_feet_z_from_meshes(
        mesh_dir=args.mesh_dir,
        num_lowest_verts=args.num_verts,
    )
    
    if not feet_z_data:
        print("[ERROR] No data to plot")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        parent_dir = os.path.dirname(os.path.abspath(args.mesh_dir))
        output_path = os.path.join(parent_dir, "feet_z_world.png")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Generate title
    title = args.title or f"Average Feet Z-Coordinate (lowest {args.num_verts} vertices)"
    
    # Plot
    plot_feet_z_world(
        feet_z_data=feet_z_data,
        output_path=output_path,
        title=title,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
