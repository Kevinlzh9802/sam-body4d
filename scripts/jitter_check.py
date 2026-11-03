#!/usr/bin/env python3
"""
Where does the mesh jitter come from? Attribution on a short piece of a sequence.

Builds camera-space meshes (MHR forward + pred_cam_t, as Stage 3 does) for
variants of the raw Stage 2 params and measures how much each one shakes:

  raw             every param as Stage 2 wrote it
  smooth          every param group smoothed over time (Gaussian, --smooth-sigma
                  frames, within each run of consecutive frames of a person)
  only:<group>    smooth, except <group> left raw -> the jitter that group causes
  stage3_option1  Stage 3's own smoothing (gated EMA on body_pose/hand/pred_cam_t,
                  global_rot EMA, shape/scale frozen to the first frame)

Groups: cam_depth (distance along the viewing ray, pred_cam_t z), cam_image (the
ray direction, i.e. where the person is in the image), global_rot, body_pose,
hand, shape_scale (shape + scale + face).

Meshes exported by stage3_mhr_full.py can be measured too (--stage3-dir).

Jitter per person = high-frequency part of the vertex trajectories,
v - G(v) with a Gaussian of --sigma frames (RMS, mm). It is split into the
translation of the whole body (mean vertex; also along / across the viewing
ray) and the rest (pose / shape change). "smooth" is the floor: real fast motion
plus what the metric cannot separate from it.

Usage (inside the container, see scripts/local_stage3.sh):
  python scripts/jitter_check.py --raw raw_mhr.pt --segment cam06_seg010 --write-slice slice.pt
  python scripts/jitter_check.py --raw slice.pt --config cfg.yaml --out jitter/ \
      --stage3-dir stage3_full=experiments/stage3_local/.../stage3_full
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import roma
import torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_DIR, os.path.join(REPO_DIR, "models", "sam_3d_body")):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr, save_raw_mhr
from smoothing.stage3_core import freeze_shape_scale_first_frame, stack_frames_to_tensors
from utils import ema_smooth_global_rot_per_obj_id_adaptive, kalman_smooth_mhr_params_per_obj_id_adaptive

GROUPS = {
    "cam_depth": None,  # handled via pred_cam_t
    "cam_image": None,
    "global_rot": ["global_rot"],
    "body_pose": ["body_pose"],
    "hand": ["hand"],
    "shape_scale": ["shape", "scale", "face"],
}


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_frames(payload, segment: Optional[str], frame_range: Optional[List[int]]):
    frames = payload["frames"]
    meta = payload.get("meta", {})
    if segment:
        segs = [s for s in meta.get("segment_id_mappings", []) if s.get("segment_key") == segment]
        if not segs:
            keys = [s.get("segment_key") for s in meta.get("segment_id_mappings", [])]
            raise SystemExit(f"Segment {segment!r} not in this file. Available: {keys}")
        fs, fe = int(segs[0]["frame_start"]), int(segs[0]["frame_end"]) + 1
    elif frame_range:
        fs, fe = frame_range
    else:
        return frames
    sel = [f for f in frames if fs <= int(f["frame"]) < fe]
    if not sel:
        raise SystemExit(f"No frames in [{fs}, {fe})")
    return sel


# ---------------------------------------------------------------------------
# Runs of consecutive visible frames per person
# ---------------------------------------------------------------------------

def person_runs(present: np.ndarray, frame_idx: np.ndarray, min_len: int) -> List[np.ndarray]:
    """Index arrays of runs where the person is present in consecutive frame numbers."""
    t = np.flatnonzero(present)
    if t.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frame_idx[t]) != 1) + 1
    return [r for r in np.split(t, breaks) if r.size >= min_len]


def smooth_rot_euler(eul: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth xyz Euler angles (L, 3) through sign-aligned quaternions."""
    q = roma.rotmat_to_unitquat(roma.euler_to_rotmat("xyz", torch.from_numpy(eul).double())).numpy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = roma.unitquat_to_rotmat(torch.from_numpy(q))
    return roma.rotmat_to_euler("xyz", R).float().numpy()


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def split_cam(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pred_cam_t (..., 3) -> depth (..., 1) and ray (..., 2) = (x/z, y/z)."""
    d = t[..., 2:3]
    return d, t[..., :2] / np.where(np.abs(d) < 1e-6, 1e-6, d)


def build_smoothed(raw: Dict[str, np.ndarray], runs, sigma: float) -> Dict[str, np.ndarray]:
    """Per-key (T, N, D) arrays smoothed within each run; cam split into depth/ray."""
    sm = {k: v.copy() for k, v in raw.items()}
    d, u = split_cam(raw["pred_cam_t"])
    sm["_depth"], sm["_ray"] = d.copy(), u.copy()
    for si, rs in runs.items():
        for r in rs:
            for k in raw:
                if k in ("pred_cam_t", "focal_length"):
                    continue
                if k == "global_rot":
                    sm[k][r, si] = smooth_rot_euler(raw[k][r, si], sigma)
                else:
                    sm[k][r, si] = gaussian_filter1d(raw[k][r, si], sigma, axis=0, mode="nearest")
            sm["_depth"][r, si] = gaussian_filter1d(d[r, si], sigma, axis=0, mode="nearest")
            sm["_ray"][r, si] = gaussian_filter1d(u[r, si], sigma, axis=0, mode="nearest")
    return sm


def compose(raw, sm, raw_groups: List[str]) -> Dict[str, np.ndarray]:
    """Smoothed params with the given groups taken from raw."""
    out = {}
    for g, keys in GROUPS.items():
        for k in keys or []:
            if k in raw:
                out[k] = raw[k] if g in raw_groups else sm[k]
    d_raw, u_raw = split_cam(raw["pred_cam_t"])
    d = d_raw if "cam_depth" in raw_groups else sm["_depth"]
    u = u_raw if "cam_image" in raw_groups else sm["_ray"]
    out["pred_cam_t"] = np.concatenate([u * d, d], axis=-1)
    return out


# ---------------------------------------------------------------------------
# Meshes and metrics
# ---------------------------------------------------------------------------

def camera_vertices(estimator, params: Dict[str, np.ndarray], T: int, N: int, device,
                    batch_size: int, vstep: int) -> np.ndarray:
    """(T, N, V', 3) camera-space vertices in metres (every vstep-th vertex)."""
    from stage3_mhr_full import run_batched_mhr_forward

    mhr = {k: torch.from_numpy(np.ascontiguousarray(v.reshape(T * N, -1))).float().to(device)
           for k, v in params.items()}
    verts, _j3d, _faces = run_batched_mhr_forward(estimator, mhr, device, batch_size, T, N)
    v = verts[:, ::vstep].numpy().reshape(T, N, -1, 3)
    del verts
    return v + params["pred_cam_t"].reshape(T, N, 1, 3)


def track_metrics(V: np.ndarray, sigma: float, ray: Optional[np.ndarray]) -> Dict[str, float]:
    """Sums of squares (m^2) of the high-pass residual of one run V (L, V', 3)."""
    c = V.mean(axis=1)
    hp_c = c - gaussian_filter1d(c, sigma, axis=0, mode="nearest")
    loc = V - c[:, None]
    hp_l = loc - gaussian_filter1d(loc, sigma, axis=0, mode="nearest")
    # Drop the ends, where the Gaussian sees padded values
    e = int(2 * sigma) if len(V) > 6 * sigma else 0
    sl = slice(e, len(V) - e)
    hp_c, hp_l = hp_c[sl], hp_l[sl]
    n = hp_c.shape[0]
    acc = V[2:] - 2 * V[1:-1] + V[:-2]
    m = {
        "n": n,
        "ss_trans": float((hp_c ** 2).sum()),
        "ss_local": float((hp_l ** 2).sum(axis=2).mean(axis=1).sum()),
        "acc_sum": float(np.linalg.norm(acc, axis=2).mean(axis=1).sum()),
        "acc_n": int(acc.shape[0]),
    }
    if ray is not None:
        along = hp_c @ ray
        m["ss_along_ray"] = float((along ** 2).sum())
    return m


def summarize(ms: List[Dict[str, float]]) -> Dict[str, float]:
    n = sum(m["n"] for m in ms)
    if n == 0:
        return {}
    to_mm = lambda ss: 1000.0 * float(np.sqrt(ss / n))
    trans, local = sum(m["ss_trans"] for m in ms), sum(m["ss_local"] for m in ms)
    out = {
        "frames": n,
        "total_mm": to_mm(trans + local),
        "translation_mm": to_mm(trans),
        "pose_shape_mm": to_mm(local),
        "accel_mm": 1000.0 * sum(m["acc_sum"] for m in ms) / max(1, sum(m["acc_n"] for m in ms)),
    }
    if all("ss_along_ray" in m for m in ms):
        along = sum(m["ss_along_ray"] for m in ms)
        out["trans_along_ray_mm"] = to_mm(along)
        out["trans_across_ray_mm"] = to_mm(max(trans - along, 0.0))
    return out


def measure(V: np.ndarray, runs, sigma: float, per_person: bool = False):
    ms_all, per = [], {}
    for si, rs in runs.items():
        ms = []
        for r in rs:
            Vr = V[r, si]
            ray = Vr.mean(axis=(0, 1))
            ms.append(track_metrics(Vr, sigma, ray / np.linalg.norm(ray)))
        ms_all += ms
        per[si] = summarize(ms)
    return (summarize(ms_all), per) if per_person else summarize(ms_all)


def measure_stage3_dir(path: str, sigma: float, vstep: int, min_len: int):
    """Same metric on stage3_mhr_full.py meshes (meshes_4d_individual/*.npz)."""
    opts = {}
    opt_file = os.path.join(path, "stage3_run_options.txt")
    if os.path.exists(opt_file):
        for line in open(opt_file, encoding="utf-8"):
            k, _, v = line.partition(": ")
            opts[k.strip()] = v.strip()
    world = opts.get("extrinsics_json", "None") != "None" and opts.get("export_camera_space") != "True"
    unit = float(opts.get("world_scale", 100.0)) if world else 1.0
    ms = []
    files = sorted(glob.glob(os.path.join(path, "meshes_4d_individual", "*.npz")))
    if not files:
        raise SystemExit(f"No meshes_4d_individual/*.npz in {path}")
    for f in files:
        z = np.load(f)
        V = z["vertices"][:, ::vstep] / unit
        idx = np.array([int(n) for n in z["frame_names"]])
        for r in person_runs(np.ones(len(idx), bool), idx, min_len):
            ray = None if world else V[r].mean(axis=(0, 1))
            ms.append(track_metrics(V[r], sigma, None if ray is None else ray / np.linalg.norm(ray)))
    return summarize(ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw_mhr.pt or a raw_mhr_segments/*.pt file")
    ap.add_argument("--segment", default=None, help="Only this segment, e.g. cam06_seg010")
    ap.add_argument("--frames", nargs=2, type=int, default=None, metavar=("START", "END"),
                    help="Only frame numbers in [START, END)")
    ap.add_argument("--write-slice", default=None, help="Save the selected frames as a raw file and exit")
    ap.add_argument("--config", default=os.path.join(REPO_DIR, "configs", "body4d.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="Output dir for jitter_report.json")
    ap.add_argument("--sigma", type=float, default=4.0, help="Metric high-pass width (frames)")
    ap.add_argument("--smooth-sigma", type=float, default=6.0, help="Smoothing of the 'smooth' reference (frames)")
    ap.add_argument("--min-run", type=int, default=30, help="Ignore visible runs shorter than this (frames)")
    ap.add_argument("--vertex-step", type=int, default=8, help="Use every n-th vertex for the metric")
    ap.add_argument("--mhr-batch-size", type=int, default=256)
    ap.add_argument("--stage3-dir", action="append", default=[], metavar="LABEL=DIR",
                    help="Also measure stage3_mhr_full.py output in DIR")
    args = ap.parse_args()

    payload = load_raw_mhr(args.raw, map_location="cpu")
    frames = select_frames(payload, args.segment, args.frames)
    if args.write_slice:
        meta = dict(payload.get("meta", {}))
        meta["note"] = f"{meta.get('note', '')} | slice {frames[0]['frame']}-{frames[-1]['frame']} of {args.raw}"
        os.makedirs(os.path.dirname(os.path.abspath(args.write_slice)), exist_ok=True)
        save_raw_mhr(args.write_slice, {"frames": frames, "meta": meta})
        print(f"[INFO] Wrote {len(frames)} frames ({frames[0]['frame']}-{frames[-1]['frame']}) to {args.write_slice}")
        return

    device = torch.device(args.device)
    mhr_t, slots, vis_flags, frame_names, obj_ids = stack_frames_to_tensors(frames, device=torch.device("cpu"))
    T, N = len(frame_names), len(obj_ids)
    frame_idx = np.array([int(n) for n in frame_names])
    present = np.array([[slots[t][si] == oid for si, oid in enumerate(obj_ids)] for t in range(T)])
    runs = {si: person_runs(present[:, si], frame_idx, args.min_run) for si in range(N)}
    n_vis = sum(len(r) for rs in runs.values() for r in rs)
    gb = T * N * (18439 // args.vertex_step + 18439) * 3 * 4 / 1e9
    print(f"[INFO] {T} frames ({frame_names[0]}-{frame_names[-1]}), {N} people, "
          f"{n_vis} person-frames in runs >= {args.min_run}; ~{gb:.1f} GB RAM per variant")

    raw = {k: v.numpy().reshape(T, N, -1) for k, v in mhr_t.items() if k != "focal_length"}
    sm = build_smoothed(raw, runs, args.smooth_sigma)

    cfg = OmegaConf.load(args.config)
    from models.sam_3d_body.sam_3d_body import load_sam_3d_body
    model, _ = load_sam_3d_body(cfg.sam_3d_body["ckpt_path"], device=device,
                                mhr_path=cfg.sam_3d_body.get("mhr_path", ""))
    estimator = SimpleNamespace(model=model, faces=model.head_pose.faces.cpu().numpy())

    def run(params):
        V = camera_vertices(estimator, params, T, N, device, args.mhr_batch_size, args.vertex_step)
        return measure(V, runs, args.sigma)

    groups = [g for g in GROUPS if g.startswith("cam_") or any(k in raw for k in GROUPS[g])]
    results: Dict[str, Dict[str, float]] = {}
    Vraw = camera_vertices(estimator, compose(raw, sm, groups), T, N, device, args.mhr_batch_size, args.vertex_step)
    results["raw"], per_person = measure(Vraw, runs, args.sigma, per_person=True)
    del Vraw
    results["smooth"] = run(compose(raw, sm, []))
    for g in groups:
        results[f"only:{g}"] = run(compose(raw, sm, [g]))

    # Stage 3 Option 1, exactly as stage3_mhr_full.py applies it
    s3 = {k: v.clone() for k, v in mhr_t.items()}
    s3 = kalman_smooth_mhr_params_per_obj_id_adaptive(
        mhr_dict=s3, num_frames=T, frame_obj_ids=slots,
        keys_to_smooth=[k for k in ["body_pose", "hand", "pred_cam_t"] if k in s3],
        kalman_cfg=None, vis_flags=copy.deepcopy(vis_flags),
    )
    freeze_shape_scale_first_frame(s3, T=T, N=N)
    s3 = ema_smooth_global_rot_per_obj_id_adaptive(
        mhr_dict=s3, num_frames=T, frame_obj_ids=slots,
        vis_flags=copy.deepcopy(vis_flags), key_name="global_rot",
    )
    results["stage3_option1"] = run({k: v.cpu().numpy().reshape(T, N, -1) for k, v in s3.items() if k != "focal_length"})

    for spec in args.stage3_dir:
        label, _, path = spec.partition("=")
        results[label] = measure_stage3_dir(path or label, args.sigma, args.vertex_step, args.min_run)

    # Report
    floor = results["smooth"]["total_mm"] ** 2
    excess = results["raw"]["total_mm"] ** 2 - floor
    print(f"\nJitter (high-pass RMS over vertices, Gaussian sigma={args.sigma} frames; mm)")
    cols = ["total_mm", "translation_mm", "trans_along_ray_mm", "trans_across_ray_mm", "pose_shape_mm", "accel_mm"]
    print(f"{'variant':<22}" + "".join(f"{c.replace('_mm', ''):>17}" for c in cols) + f"{'share of raw':>14}")
    for name, r in results.items():
        share = ""
        if name.startswith("only:") and excess > 0:
            share = f"{100 * (r['total_mm'] ** 2 - floor) / excess:.0f}%"
        print(f"{name:<22}" + "".join(f"{r.get(c, float('nan')):>17.2f}" for c in cols) + f"{share:>14}")
    print("\n'share of raw': extra jitter (in squared mm) that leaving one group raw adds over 'smooth',\n"
          "as a fraction of raw minus smooth. Shares need not sum to 100% (groups interact).")

    worst = sorted(per_person.items(), key=lambda kv: -kv[1].get("total_mm", 0))[:5]
    print("\nRaw jitter per person (worst 5):")
    for si, r in worst:
        if r:
            print(f"  id {obj_ids[si]:>3}: total {r['total_mm']:.1f} mm, translation {r['translation_mm']:.1f} "
                  f"(along ray {r.get('trans_along_ray_mm', float('nan')):.1f}), pose/shape {r['pose_shape_mm']:.1f}, "
                  f"{r['frames']} frames")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        report = {
            "raw_file": args.raw, "frames": [frame_names[0], frame_names[-1]], "sigma": args.sigma,
            "smooth_sigma": args.smooth_sigma, "vertex_step": args.vertex_step, "variants": results,
            "raw_per_person": {str(obj_ids[si]): r for si, r in per_person.items()},
        }
        path = os.path.join(args.out, "jitter_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Saved {path}")


if __name__ == "__main__":
    main()
