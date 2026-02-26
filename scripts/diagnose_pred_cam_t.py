#!/usr/bin/env python3
"""
Diagnostic tool: visualise pred_cam_t from raw_mhr.pt to identify jumps
and jittering, both across segment boundaries and within segments.

Produces:
  - pred_cam_t_xyz.png       Per-person X/Y/Z components over time, with
                              segment boundaries marked.
  - pred_cam_t_velocity.png  Frame-to-frame L2 velocity of pred_cam_t, with
                              segment boundaries marked. Spikes indicate jumps.
  - pred_cam_t_stats.json    Per-segment and overall statistics (mean, std,
                              max velocity, etc.) for each person.

Usage:
  python scripts/diagnose_pred_cam_t.py --raw /path/to/raw_mhr.pt
  python scripts/diagnose_pred_cam_t.py --raw /path/to/raw_mhr.pt --out /path/to/diag_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sam3d_body_pkg = os.path.join(REPO_DIR, "models", "sam_3d_body")
if sam3d_body_pkg not in sys.path:
    sys.path.insert(0, sam3d_body_pkg)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr
from smoothing.stage3_core import stack_frames_to_tensors
from utils.id_mapping import load_segment_id_mappings_from_meta


def _segment_boundaries(segment_id_mappings):
    """Return list of (frame_start, frame_end, segment_key) tuples."""
    if not segment_id_mappings:
        return []
    return [
        (int(s["frame_start"]), int(s["frame_end"]), s.get("segment_key", ""))
        for s in segment_id_mappings
    ]


def _find_segment_idx(frame, boundaries):
    for i, (fs, fe, _) in enumerate(boundaries):
        if fs <= frame <= fe:
            return i
    return -1


def plot_xyz(
    pred_cam_t_per_person, obj_ids_all, frame_names, boundaries, out_path,
):
    """Plot X, Y, Z components of pred_cam_t for each person."""
    n_persons = len(obj_ids_all)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    labels = ["X (horizontal)", "Y (vertical)", "Z (depth)"]
    cmap = plt.cm.get_cmap("tab20")

    frames_int = np.arange(len(frame_names))

    for ci, (ax, label) in enumerate(zip(axes, labels)):
        for pi, oid in enumerate(obj_ids_all):
            data = pred_cam_t_per_person[oid]
            fi = data["frame_indices"]
            vals = data["values"][:, ci]
            color = cmap(pi / max(n_persons, 1))
            ax.plot(fi, vals, color=color, alpha=0.6, linewidth=0.8, label=f"ID {oid}")

        for fs, fe, skey in boundaries:
            ax.axvline(x=fs, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    axes[0].set_title("pred_cam_t components (raw Stage 2)")
    axes[-1].set_xlabel("Frame")

    handles, lbls = axes[0].get_legend_handles_labels()
    if len(lbls) <= 20:
        axes[0].legend(handles, lbls, fontsize="x-small", ncol=4, loc="upper right")

    if boundaries:
        axes[0].text(
            0.01, 0.97, f"{len(boundaries)} segments (red = boundary)",
            transform=axes[0].transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


def plot_velocity(
    pred_cam_t_per_person, obj_ids_all, frame_names, boundaries, out_path,
):
    """Plot frame-to-frame velocity (L2 norm of delta pred_cam_t)."""
    n_persons = len(obj_ids_all)
    fig, ax = plt.subplots(figsize=(16, 5))
    cmap = plt.cm.get_cmap("tab20")

    for pi, oid in enumerate(obj_ids_all):
        data = pred_cam_t_per_person[oid]
        fi = data["frame_indices"]
        vals = data["values"]
        if len(fi) < 2:
            continue
        delta = np.diff(vals, axis=0)
        vel = np.linalg.norm(delta, axis=1)
        color = cmap(pi / max(n_persons, 1))
        ax.plot(fi[1:], vel, color=color, alpha=0.5, linewidth=0.7, label=f"ID {oid}")

    for fs, fe, skey in boundaries:
        ax.axvline(x=fs, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Frame")
    ax.set_ylabel("||Δ pred_cam_t|| (L2)")
    ax.set_title("pred_cam_t velocity (frame-to-frame L2 norm) — spikes indicate jumps")
    ax.grid(True, alpha=0.2)

    handles, lbls = ax.get_legend_handles_labels()
    if len(lbls) <= 20:
        ax.legend(handles, lbls, fontsize="x-small", ncol=4, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")


def compute_per_segment_stats(pred_cam_t_per_person, obj_ids_all, boundaries):
    """Compute per-person, per-segment and overall stats."""
    stats = {}
    for oid in obj_ids_all:
        data = pred_cam_t_per_person[oid]
        fi = data["frame_indices"]
        vals = data["values"]

        if len(fi) < 2:
            stats[str(oid)] = {"num_frames": int(len(fi)), "segments": {}}
            continue

        delta = np.diff(vals, axis=0)
        vel = np.linalg.norm(delta, axis=1)

        overall = {
            "num_frames": int(len(fi)),
            "mean_xyz": vals.mean(axis=0).tolist(),
            "std_xyz": vals.std(axis=0).tolist(),
            "vel_mean": float(vel.mean()),
            "vel_std": float(vel.std()),
            "vel_max": float(vel.max()),
            "vel_p95": float(np.percentile(vel, 95)),
            "vel_p99": float(np.percentile(vel, 99)),
        }

        seg_stats = {}
        if boundaries:
            for si, (fs, fe, skey) in enumerate(boundaries):
                mask = (fi >= fs) & (fi <= fe)
                seg_fi = fi[mask]
                seg_vals = vals[mask]
                if len(seg_fi) < 2:
                    seg_stats[skey or str(si)] = {"num_frames": int(len(seg_fi))}
                    continue
                seg_delta = np.diff(seg_vals, axis=0)
                seg_vel = np.linalg.norm(seg_delta, axis=1)
                seg_stats[skey or str(si)] = {
                    "num_frames": int(len(seg_fi)),
                    "mean_xyz": seg_vals.mean(axis=0).tolist(),
                    "std_xyz": seg_vals.std(axis=0).tolist(),
                    "vel_mean": float(seg_vel.mean()),
                    "vel_std": float(seg_vel.std()),
                    "vel_max": float(seg_vel.max()),
                    "vel_p95": float(np.percentile(seg_vel, 95)),
                }

            # Boundary jumps: compare last frame of segment i vs first frame of segment i+1
            boundary_jumps = []
            for si in range(len(boundaries) - 1):
                _, fe_cur, _ = boundaries[si]
                fs_nxt, _, _ = boundaries[si + 1]
                mask_cur = fi == fe_cur
                mask_nxt = fi == fs_nxt
                if mask_cur.any() and mask_nxt.any():
                    v_cur = vals[mask_cur][-1]
                    v_nxt = vals[mask_nxt][0]
                    jump = float(np.linalg.norm(v_nxt - v_cur))
                    boundary_jumps.append({
                        "boundary": f"{fe_cur}->{fs_nxt}",
                        "jump_l2": jump,
                        "delta_xyz": (v_nxt - v_cur).tolist(),
                    })
            overall["boundary_jumps"] = boundary_jumps

        overall["segments"] = seg_stats
        stats[str(oid)] = overall

    return stats


def run_diagnostics(
    *,
    mhr: Dict,
    frame_obj_ids_slots: List,
    frame_names: List[str],
    obj_ids_all: List[int],
    segment_id_mappings,
    out_dir: str,
    label: str = "raw",
):
    """
    Callable entry point for Stage 3 scripts.

    Accepts already-loaded tensors (avoids re-loading raw_mhr.pt).
    *label* is used in filenames to distinguish raw vs post-smoothing diagnostics.
    """
    os.makedirs(out_dir, exist_ok=True)
    boundaries = _segment_boundaries(segment_id_mappings) if segment_id_mappings else []

    T = len(frame_names)
    N = len(obj_ids_all)

    pred_cam_t_np = mhr["pred_cam_t"].detach().cpu().view(T, N, 3).numpy()

    pred_cam_t_per_person: Dict[int, Dict[str, np.ndarray]] = {}
    for si, oid in enumerate(obj_ids_all):
        fi_list, val_list = [], []
        for ti in range(T):
            if frame_obj_ids_slots[ti][si] == oid:
                fi_list.append(ti)
                val_list.append(pred_cam_t_np[ti, si])
        if fi_list:
            pred_cam_t_per_person[oid] = {
                "frame_indices": np.array(fi_list),
                "values": np.array(val_list),
            }

    plot_xyz(pred_cam_t_per_person, obj_ids_all, frame_names, boundaries,
             os.path.join(out_dir, f"pred_cam_t_xyz_{label}.png"))
    plot_velocity(pred_cam_t_per_person, obj_ids_all, frame_names, boundaries,
                  os.path.join(out_dir, f"pred_cam_t_velocity_{label}.png"))

    stats = compute_per_segment_stats(pred_cam_t_per_person, obj_ids_all, boundaries)
    stats_path = os.path.join(out_dir, f"pred_cam_t_stats_{label}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DIAG:{label}] pred_cam_t summary  (saved to {out_dir})")
    print(f"{'ID':>6} | {'Frames':>7} | {'vel_mean':>9} | {'vel_std':>9} | {'vel_max':>9} | {'vel_p99':>9} | {'bnd_jumps':>9}")
    print(f"{'-'*80}")
    for oid_str, s in sorted(stats.items(), key=lambda kv: int(kv[0])):
        nf = s.get("num_frames", 0)
        vm = s.get("vel_mean", 0)
        vs = s.get("vel_std", 0)
        vx = s.get("vel_max", 0)
        v99 = s.get("vel_p99", 0)
        bj = s.get("boundary_jumps", [])
        max_bj = max((j["jump_l2"] for j in bj), default=0) if bj else 0
        print(f"{oid_str:>6} | {nf:>7} | {vm:>9.5f} | {vs:>9.5f} | {vx:>9.5f} | {v99:>9.5f} | {max_bj:>9.5f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose pred_cam_t from raw_mhr.pt")
    parser.add_argument("--raw", required=True, help="Path to raw_mhr.pt")
    parser.add_argument("--out", default=None, help="Output directory (default: alongside raw file)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(args.raw), "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading {args.raw} ...")
    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = payload["frames"]
    meta = payload.get("meta", {})
    segment_id_mappings = load_segment_id_mappings_from_meta(meta, fallback_dir=os.path.dirname(args.raw))

    device = torch.device("cpu")
    mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all = stack_frames_to_tensors(frames, device=device)

    run_diagnostics(
        mhr=mhr,
        frame_obj_ids_slots=frame_obj_ids_slots,
        frame_names=frame_names,
        obj_ids_all=obj_ids_all,
        segment_id_mappings=segment_id_mappings,
        out_dir=out_dir,
        label="raw",
    )


if __name__ == "__main__":
    main()
