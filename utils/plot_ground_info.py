"""
Plot 2D ground-plane positions and orientations (arrows) from ground_plane_info.

One image every N frames (default 200), saved under output_dir/ground_plane_plots/.
Different colors per person, different markers per body part (head/shoulder/hip/foot),
person id as text alongside.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Marker and short name per body part for legend/clarity
HEAD_MARKER = "o"   # circle
SHOULDER_MARKER = "s"  # square
HIP_MARKER = "D"    # diamond
FOOT_MARKER = "*"   # star

ARROW_SCALE_FRAC = 0.08  # arrow length as fraction of axis range (or min if range is 0)
ARROW_MIN_CM = 15.0
TEXT_OFFSET_CM = 8.0


def _arrow_scale(ax_xy_min: float, ax_xy_max: float) -> float:
    r = ax_xy_max - ax_xy_min
    if r <= 0:
        return ARROW_MIN_CM
    return max(ARROW_MIN_CM, r * ARROW_SCALE_FRAC)


def plot_ground_info_frame(
    frame_rows: List[Dict[str, Any]],
    frame_name: str,
    obj_id_to_color: Dict[Any, tuple],
    arrow_scale_cm: float,
    out_path: str,
) -> None:
    """Draw one frame: all persons with positions (markers), orientations (arrows), and id text."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")

    all_x: List[float] = []
    all_y: List[float] = []

    for r in frame_rows:
        oid = r["obj_id"]
        color = obj_id_to_color[oid]
        # Head: circle + orientation arrow
        hx, hy = r["head_xy"][0], r["head_xy"][1]
        ax.scatter(hx, hy, marker=HEAD_MARKER, c=[color], s=120, edgecolors="k", linewidths=0.5, zorder=5)
        dx = np.cos(r["head_orient_rad"]) * arrow_scale_cm
        dy = np.sin(r["head_orient_rad"]) * arrow_scale_cm
        ax.quiver(hx, hy, dx, dy, color=color, scale=1, scale_units="xy", angles="xy", width=0.003, headwidth=4, headlength=5, zorder=4)
        all_x.extend([hx, hx + dx]); all_y.extend([hy, hy + dy])

        # Shoulder: squares at L/R + orientation arrow from center
        sl, sr = r["shoulder_left_xy"], r["shoulder_right_xy"]
        ax.scatter(sl[0], sl[1], marker=SHOULDER_MARKER, c=[color], s=70, edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(sr[0], sr[1], marker=SHOULDER_MARKER, c=[color], s=70, edgecolors="k", linewidths=0.5, zorder=5)
        scx = (sl[0] + sr[0]) / 2
        scy = (sl[1] + sr[1]) / 2
        dx = np.cos(r["shoulder_orient_rad"]) * arrow_scale_cm
        dy = np.sin(r["shoulder_orient_rad"]) * arrow_scale_cm
        ax.quiver(scx, scy, dx, dy, color=color, scale=1, scale_units="xy", angles="xy", width=0.0025, headwidth=4, headlength=5, zorder=4)
        all_x.extend([sl[0], sr[0], scx + dx]); all_y.extend([sl[1], sr[1], scy + dy])

        # Hip: diamonds at L/R + orientation arrow from center
        hl, hr = r["hip_left_xy"], r["hip_right_xy"]
        ax.scatter(hl[0], hl[1], marker=HIP_MARKER, c=[color], s=70, edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(hr[0], hr[1], marker=HIP_MARKER, c=[color], s=70, edgecolors="k", linewidths=0.5, zorder=5)
        hcx = (hl[0] + hr[0]) / 2
        hcy = (hl[1] + hr[1]) / 2
        dx = np.cos(r["hip_orient_rad"]) * arrow_scale_cm
        dy = np.sin(r["hip_orient_rad"]) * arrow_scale_cm
        ax.quiver(hcx, hcy, dx, dy, color=color, scale=1, scale_units="xy", angles="xy", width=0.0025, headwidth=4, headlength=5, zorder=4)
        all_x.extend([hl[0], hr[0], hcx + dx]); all_y.extend([hl[1], hr[1], hcy + dy])

        # Foot: stars at L/R; single foot orientation (same for both), draw from each foot
        fl, fr = r["foot_left_xy"], r["foot_right_xy"]
        ax.scatter(fl[0], fl[1], marker=FOOT_MARKER, c=[color], s=150, edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(fr[0], fr[1], marker=FOOT_MARKER, c=[color], s=150, edgecolors="k", linewidths=0.5, zorder=5)
        for fx, fy in [fl, fr]:
            dx = np.cos(r["foot_orient_rad"]) * arrow_scale_cm
            dy = np.sin(r["foot_orient_rad"]) * arrow_scale_cm
            ax.quiver(fx, fy, dx, dy, color=color, scale=1, scale_units="xy", angles="xy", width=0.0025, headwidth=4, headlength=5, zorder=4)
        all_x.extend([fl[0], fr[0]]); all_y.extend([fl[1], fr[1]])

        # Person id text (offset from head so it doesn't overlap)
        ax.text(hx + TEXT_OFFSET_CM, hy + TEXT_OFFSET_CM, str(oid), color=color, fontsize=10, fontweight="bold", zorder=6)

    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 20.0)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"Ground plane — {frame_name}")
    ax.grid(True, alpha=0.3)
    # Legend for body-part markers (proxy artists)
    legend_elements = [
        Line2D([0], [0], marker=HEAD_MARKER, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=10, label="head"),
        Line2D([0], [0], marker=SHOULDER_MARKER, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="shoulder"),
        Line2D([0], [0], marker=HIP_MARKER, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label="hip"),
        Line2D([0], [0], marker=FOOT_MARKER, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=12, label="foot"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_plot_ground_info(
    rows: List[Dict[str, Any]],
    frame_names: List[str],
    output_dir: str,
    frame_interval: int = 200,
    plot_subdir: str = "ground_plane_plots",
) -> str:
    """
    Plot 2D ground-plane positions and orientations every frame_interval frames.

    rows: list of ground_plane_info dicts (one per frame per person).
    frame_names: ordered list of frame names (same order as in the sequence).
    output_dir: <digitnum> folder; plots go to output_dir/plot_subdir/.

    Returns the path to the plot subdir.
    """
    if not rows:
        return os.path.join(output_dir, plot_subdir)

    # Unique obj_ids for consistent colors
    obj_ids = sorted(set(r["obj_id"] for r in rows))
    try:
        cmap = matplotlib.colormaps["tab10" if len(obj_ids) <= 10 else "tab20"]
    except Exception:
        cmap = plt.cm.tab10
    obj_id_to_color = {}
    for i, oid in enumerate(obj_ids):
        t = i / max(len(obj_ids) - 1, 1)
        obj_id_to_color[oid] = cmap(t)

    # Arrow scale from full data range (rough)
    xs = [r["head_xy"][0] for r in rows] + [r["foot_left_xy"][0] for r in rows] + [r["foot_right_xy"][0] for r in rows]
    ys = [r["head_xy"][1] for r in rows] + [r["foot_left_xy"][1] for r in rows] + [r["foot_right_xy"][1] for r in rows]
    ax_min = min(min(xs), min(ys))
    ax_max = max(max(xs), max(ys))
    arrow_scale_cm = _arrow_scale(ax_min, ax_max)

    plot_dir = os.path.join(output_dir, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    n_saved = 0
    for frame_idx in range(0, len(frame_names), frame_interval):
        frame_name = frame_names[frame_idx]
        frame_rows = [r for r in rows if r["frame_name"] == frame_name]
        if not frame_rows:
            continue
        out_path = os.path.join(plot_dir, f"{frame_name}.png")
        plot_ground_info_frame(
            frame_rows,
            frame_name,
            obj_id_to_color,
            arrow_scale_cm,
            out_path,
        )
        n_saved += 1

    return plot_dir
