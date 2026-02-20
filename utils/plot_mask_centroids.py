"""
Plot per-person mask centroid (x, y) over time from Stage 1 palette masks.

One figure with two subplots (x vs frame, y vs frame), one line per person,
using actual (real) IDs as labels. Saved as a single PNG.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mask_centroids(
    masks_list: List[str],
    to_actual_pid: Callable[[int, int], int],
) -> Dict[int, Dict[str, list]]:
    """
    Scan all mask PNGs and compute per-person centroid for each frame.

    Args:
        masks_list: sorted list of mask PNG paths (index = frame index).
        to_actual_pid: callable(consecutive_id, frame_idx) -> actual_pid.

    Returns:
        {actual_pid: {"frames": [int, ...], "cx": [float, ...], "cy": [float, ...]}}
    """
    data: Dict[int, Dict[str, list]] = defaultdict(lambda: {"frames": [], "cx": [], "cy": []})

    for frame_idx, mask_path in enumerate(tqdm(masks_list, desc="Mask centroids")):
        if os.path.getsize(mask_path) == 0:
            continue
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        unique_ids = np.unique(mask)
        for cid in unique_ids:
            if cid == 0:
                continue
            ys, xs = np.where(mask == cid)
            cx = float(xs.mean())
            cy = float(ys.mean())
            actual_pid = to_actual_pid(int(cid), frame_idx)
            data[actual_pid]["frames"].append(frame_idx)
            data[actual_pid]["cx"].append(cx)
            data[actual_pid]["cy"].append(cy)

    return dict(data)


def plot_mask_centroids(
    centroid_data: Dict[int, Dict[str, list]],
    output_path: str,
    title_prefix: str = "Stage 1 Mask Centroids",
) -> None:
    """Plot centroid x and y vs frame for all persons."""
    if not centroid_data:
        return

    pids = sorted(centroid_data.keys())
    try:
        cmap = matplotlib.colormaps["tab10" if len(pids) <= 10 else "tab20"]
    except Exception:
        cmap = plt.cm.tab10

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for i, pid in enumerate(pids):
        color = cmap(i / max(len(pids) - 1, 1))
        d = centroid_data[pid]
        ax_x.plot(d["frames"], d["cx"], color=color, label=str(pid), linewidth=0.8, alpha=0.85)
        ax_y.plot(d["frames"], d["cy"], color=color, label=str(pid), linewidth=0.8, alpha=0.85)

    ax_x.set_ylabel("centroid x (px)")
    ax_x.set_title(f"{title_prefix} — X coordinate")
    ax_x.legend(fontsize=6, ncol=max(1, len(pids) // 4), loc="upper right")
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel("frame index")
    ax_y.set_ylabel("centroid y (px)")
    ax_y.set_title(f"{title_prefix} — Y coordinate")
    ax_y.legend(fontsize=6, ncol=max(1, len(pids) // 4), loc="upper right")
    ax_y.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved mask centroid plot: {output_path}")
