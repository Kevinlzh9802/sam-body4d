"""
Plot per-person mesh height (max_z - min_z) in world coordinates across frames.

Height is the vertical extent of the full mesh (head to feet) in world space,
which should be roughly constant across time for a given person (~160-185 cm
for adults) since shape/scale are frozen after the first frame.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_mesh_heights_world(
    *,
    vertices: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr_R: np.ndarray,
    extr_t: np.ndarray,
    world_scale: float,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute mesh height (max_z - min_z in world coords) per person per frame.

    Args:
        vertices: (T*N, V, 3) local mesh vertices (after y,z flip).
        pred_cam_t: (T*N, 3) camera translation.
        extr_R: (3,3) rotation matrix (OpenCV: X_cam = R @ X_world + t).
        extr_t: (3,) translation vector.
        world_scale: metres-to-extrinsic-units factor (100 = cm).

    Returns:
        {obj_id: (frame_indices ndarray, heights ndarray)} in world units.
    """
    Rt = extr_R.T
    t = extr_t.reshape(1, 3)

    results: Dict[int, Tuple[List[int], List[float]]] = {}
    for si, oid in enumerate(obj_ids_all):
        frames_list: List[int] = []
        heights_list: List[float] = []
        for ti in range(T):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            v = vertices[bi].detach().float().cpu().numpy()
            camt = pred_cam_t[bi].detach().float().cpu().numpy()
            v_world = (v + camt) * world_scale
            v_world = (v_world - t) @ Rt
            h = float(v_world[:, 2].max() - v_world[:, 2].min())
            frames_list.append(ti)
            heights_list.append(h)
        if frames_list:
            results[oid] = (np.array(frames_list), np.array(heights_list))
    return results


def plot_mesh_heights(
    height_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Mesh Height (World Space)",
    world_units: str = "cm",
) -> None:
    """Save a per-person mesh-height-over-time plot."""
    if not height_data:
        print("[WARN] No mesh height data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.get_cmap("tab20")
    n_persons = len(height_data)

    for idx, (oid, (frames, heights)) in enumerate(sorted(height_data.items())):
        color = cmap(idx / max(n_persons, 1))
        mean_h = float(heights.mean())
        ax.plot(frames, heights, color=color, alpha=0.8, linewidth=1,
                label=f"ID {oid} (mean {mean_h:.1f} {world_units})")

    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Mesh height ({world_units})")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)

    all_h = np.concatenate([h for _, (_, h) in height_data.items()])
    stats = f"Overall: min={all_h.min():.1f}, max={all_h.max():.1f}, mean={all_h.mean():.1f} {world_units}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Mesh height plot saved to: {output_path}")


def plot_mesh_heights_from_stage3(
    *,
    vertices: torch.Tensor,
    pred_cam_t: torch.Tensor,
    extr,
    world_scale: float,
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Convenience wrapper matching the Extrinsics dataclass API."""
    extr_R = extr.R.detach().cpu().numpy()
    extr_t = extr.t.detach().cpu().numpy()
    data = compute_mesh_heights_world(
        vertices=vertices, pred_cam_t=pred_cam_t,
        extr_R=extr_R, extr_t=extr_t, world_scale=world_scale,
        T=T, N=N, obj_ids_all=obj_ids_all,
        frame_obj_ids_slots=frame_obj_ids_slots,
    )
    wu = "cm" if abs(world_scale - 100.0) < 1 else "world units"
    plot_mesh_heights(
        data, output_path,
        title=title or "Mesh Height (World Space)",
        world_units=wu,
    )
