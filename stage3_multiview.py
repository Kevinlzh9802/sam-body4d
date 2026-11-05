#!/usr/bin/env python3
"""
Stage 3, multi-view option: metric scale and floor from the rig's other
cameras instead of an assumed camera height.

Input: Stage 3 outputs of several cameras of one recording session and batch,
exported in camera space without the ground step (stage3_mhr.py
--export-camera-space --no-ground; job_scripts/stage3_multiview.sh makes them
as <seq>/stage3_camera). cosilab has two sessions, separate recordings:
session 1 = cam06-10 (rig calibration parameters-camera-06-to-10.json),
session 2 = cam01-05 (parameters-camera-01-to-04.json; cam05 is not in it, so
it cannot take part). Within a session and batch the same frame number is the
same moment in every camera, and real person IDs are the same across cameras.
Inputs from different sessions or batches are refused. Negative IDs (people
outside the study) are ignored.

  1. scale     One camera cannot tell a person's size from their distance:
               scaling the camera-space mesh about the camera centre leaves the
               image unchanged. Two cameras can: the rig calibration has their
               poses in metres, and the viewing rays of the same body point meet
               at its true position. Per frame, the pelvis (mid-hip) and chest
               (mid-shoulder) of each person seen by >= 2 cameras are triangulated
               (least squares over the cameras); a camera's distance to the
               triangulated point over its own estimate is that camera's scale for
               the frame. One scale per camera and person: the median over the
               frames. Frames whose rays miss each other by more than
               --max-ray-gap or meet at less than --min-ray-angle are left out.
               People never seen by two cameras get their camera's median scale.
  2. floor     With people at metric scale, the floor is where their feet are.
               The only assumption: the rig's reference camera (cam06 / cam01)
               looks straight down, so the floor is horizontal in its frame, at
               the median height of every person-frame's lowest foot point. Its
               distance is the reference camera's height above the floor
               (summary: floor.camera_height_m, cf. the 3.5 m of the height option).
  3. contact   As in the height option, a small per-frame shift along the
               viewing ray puts each frame's lowest foot on that floor; the
               scale stays the triangulated one (--no-contact: skip).
  4. export    <seq>/<label>/ per camera (default label stage3_multiview):
               meshes_4d_individual/<pid>.npz in world coordinates (cm by default;
               floor z=0, z up, origin below the reference camera; same keys as
               stage3_mhr.py), extrinsics/extrinsic_cam_<MM>.json for all rig
               cameras at the estimated height, stage3_run_options.txt,
               mesh_heights_world.png. Summary of all cameras:
               stage3_multiview_summary.json in each output folder.

Sync: the summary also lists, per camera pair, the median distance between the
two cameras' rays through the same torso point for frame offsets -30..30. With
synchronized cameras it is smallest at offset 0; with slow movement the curve is
flat for a few frames around it (cam06/cam08, batch01 seg010: ~3.0 cm within
+-8 frames, 4 cm at +-30), so it rules out large offsets, not small ones.

Usage:
  python stage3_multiview.py <cam06>/stage3_camera <cam08>/stage3_camera ... \\
      --calib extrinsics/parameters-camera-06-to-10.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from scripts.make_ground_extrinsics import extrinsics_filename, ground_extrinsics
from scripts.view_stage3 import (COSILAB_SESSIONS, batch_number, camera_number, load_rig_calibration,
                                 read_run_options, session_of)
from smoothing.ground_plane_opt import FOOT_IDXS, GroundOptConfig, load_extrinsics_json, optimize_ground_ray_scale
from utils.plot_mesh_heights import plot_mesh_heights

# mhr70 torso points used for triangulation (left/right averaged, so a swapped
# left/right in one view does not matter)
HIPS, SHOULDERS = (9, 10), (5, 6)
SYNC_OFFSETS = range(-30, 31)


@dataclass
class CamPerson:
    frames: np.ndarray   # (T,) frame numbers
    torso: np.ndarray    # (T, 2, 3) mid-hip, mid-shoulder; camera space, m
    feet: np.ndarray     # (T, 8, 3) FOOT_IDXS keypoints; camera space, m
    scale: float = 1.0
    source: str = "none"
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CamInput:
    name: str            # cam06
    in_dir: str
    out_dir: str
    opts: Dict[str, str]
    R: np.ndarray        # rig -> camera: x_cam = R x_rig + t (m)
    t: np.ndarray
    people: Dict[int, CamPerson]

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t


def torso_points(kps: np.ndarray) -> np.ndarray:
    """(T, 70, 3) -> (T, 2, 3): mid-hip, mid-shoulder."""
    return np.stack([kps[:, list(HIPS)].mean(axis=1), kps[:, list(SHOULDERS)].mean(axis=1)], axis=1)


def load_camera(in_dir: str, out_label: str, rig: Dict[str, Any]) -> CamInput:
    opts = read_run_options(in_dir)
    if not opts:
        raise SystemExit(f"[ERROR] {in_dir}: no stage3_run_options.txt (not a stage3_mhr.py output?)")
    camera_space = opts.get("export_camera_space") == "True" or opts.get("extrinsics_json") in (None, "None")
    grounded = opts.get("extrinsics_json") not in (None, "None") and opts.get("no_ground") != "True"
    if not camera_space or grounded:
        raise SystemExit(f"[ERROR] {in_dir}: needs a camera-space Stage 3 output without the ground step "
                         "(stage3_mhr.py --export-camera-space --no-ground)")
    num = camera_number(in_dir, opts)
    name = f"cam{num:02d}" if num is not None else None
    if name not in rig:
        raise SystemExit(f"[ERROR] {in_dir}: camera {name} is not in the rig calibration ({sorted(rig)}); "
                         f"sessions: {COSILAB_SESSIONS}")
    mesh_dir = os.path.join(in_dir, "meshes_4d_individual")
    people: Dict[int, CamPerson] = {}
    for fn in sorted(os.listdir(mesh_dir)):
        if not fn.endswith(".npz"):
            continue
        pid = int(fn[:-4])
        if pid < 0:
            continue
        with np.load(os.path.join(mesh_dir, fn)) as z:
            if "keypoints" not in z.files or "cam_t" not in z.files:
                raise SystemExit(f"[ERROR] {in_dir}: {fn} has no keypoints / cam_t (made before they were "
                                 "exported); run stage3_mhr.py again for this camera")
            kps = z["keypoints"].astype(np.float64)
            people[pid] = CamPerson(frames=z["frame_names"].astype(np.int64), torso=torso_points(kps),
                                    feet=kps[:, list(FOOT_IDXS)])
    out_dir = os.path.join(os.path.dirname(os.path.normpath(in_dir)), out_label)
    cam = rig[name]
    return CamInput(name, in_dir, out_dir, opts, cam.R, cam.t, people)


# ---------------------------------------------------------------------------
# 1. scale from triangulation
# ---------------------------------------------------------------------------

def triangulate_person(cams: List[CamInput], pid: int, max_gap: float, min_angle_deg: float
                       ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per camera: (per-sample scale ratios, ray gaps in m) of the accepted
    (frame, torso point) samples of person pid."""
    seen = [c for c in cams if pid in c.people]
    frames = np.unique(np.concatenate([c.people[pid].frames for c in seen]))
    F = len(frames)
    A = np.zeros((F, 2, 3, 3))
    b = np.zeros((F, 2, 3))
    n_cams = np.zeros(F, dtype=np.int64)
    rays = {}
    for c in seen:
        p = c.people[pid]
        rows = np.searchsorted(frames, p.frames)
        pts = p.torso                                   # (T, 2, 3) camera space
        dist = np.linalg.norm(pts, axis=-1)             # (T, 2)
        u = (pts / dist[..., None]) @ c.R               # ray directions in the rig frame
        P = np.eye(3) - u[..., :, None] * u[..., None, :]
        A[rows] += P
        b[rows] += P @ c.centre
        n_cams[rows] += 1
        rays[c.name] = (rows, u, dist)

    multi = n_cams >= 2
    X = np.full((F, 2, 3), np.nan)
    # (tiny ridge: rays of one sample are never parallel, but keep solve() safe)
    X[multi] = np.linalg.solve(A[multi] + 1e-9 * np.eye(3), b[multi][..., None])[..., 0]

    ok = np.repeat(multi[:, None], 2, axis=1)           # (F, 2)
    depths = {}
    with np.errstate(invalid="ignore"):                 # NaN rows: frames this camera sees alone
        for c in seen:
            rows, u, dist = rays[c.name]
            d = X[rows] - c.centre                      # (T, 2, 3)
            lam = np.einsum("tpk,tpk->tp", d, u)
            gap = np.linalg.norm(d - lam[..., None] * u, axis=-1)
            depths[c.name] = (rows, lam, dist, gap)
            ok[rows] &= (lam > 0) & (gap <= max_gap)
    # Widest angle between any two rays of a sample
    cos_min = np.ones((F, 2))
    for a, bb in itertools.combinations(seen, 2):
        ra, ua, _ = rays[a.name]
        rb, ub, _ = rays[bb.name]
        Ua = np.full((F, 2, 3), np.nan)
        Ub = np.full((F, 2, 3), np.nan)
        Ua[ra], Ub[rb] = ua, ub
        cos = np.einsum("fpk,fpk->fp", Ua, Ub)
        both = ~np.isnan(cos)
        cos_min[both] = np.minimum(cos_min[both], cos[both])
    ok &= cos_min <= np.cos(np.radians(min_angle_deg))

    out = {}
    for c in seen:
        rows, lam, dist, gap = depths[c.name]
        sel = ok[rows]
        out[c.name] = (lam[sel] / dist[sel], gap[sel])
    return out


def estimate_scales(cams: List[CamInput], max_gap: float, min_angle_deg: float, min_samples: int
                    ) -> Dict[str, float]:
    """Sets scale/source/stats on every CamPerson; returns each camera's median scale."""
    pids = sorted({pid for c in cams for pid in c.people})
    for pid in pids:
        if sum(pid in c.people for c in cams) < 2:
            continue
        for name, (ratios, gaps) in triangulate_person(cams, pid, max_gap, min_angle_deg).items():
            cp = next(c for c in cams if c.name == name).people[pid]
            cp.stats = {"samples": int(ratios.size)}
            if ratios.size >= min_samples:
                cp.scale = float(np.median(ratios))
                cp.source = "triangulated"
                cp.stats.update({
                    "scale_p25_p75": [round(float(v), 4) for v in np.percentile(ratios, [25, 75])],
                    "ray_gap_cm_median": round(float(np.median(gaps)) * 100, 2),
                })

    tri = [cp.scale for c in cams for cp in c.people.values() if cp.source == "triangulated"]
    if not tri:
        raise SystemExit("[ERROR] Nobody could be triangulated: no person seen by two cameras in the same "
                         "frames within --max-ray-gap (check that the IDs match and the frames are in sync)")
    rig_median = float(np.median(tri))
    cam_median = {}
    for c in cams:
        own = [cp.scale for cp in c.people.values() if cp.source == "triangulated"]
        cam_median[c.name] = float(np.median(own)) if own else rig_median
        for cp in c.people.values():
            if cp.source != "triangulated":
                cp.scale = cam_median[c.name]
                cp.source = "camera_median" if own else "rig_median"
    return cam_median


def sync_check(cams: List[CamInput]) -> Dict[str, Any]:
    """Median distance (cm) between two cameras' rays through the same torso
    point, per frame offset of the second camera."""
    out = {}
    for a, b in itertools.combinations(cams, 2):
        shared = sorted(set(a.people) & set(b.people))
        if not shared:
            continue
        w = b.centre - a.centre
        med = {}
        for off in SYNC_OFFSETS:
            gaps = []
            for pid in shared:
                pa, pb = a.people[pid], b.people[pid]
                common, ia, ib = np.intersect1d(pa.frames, pb.frames + off, return_indices=True)
                if not len(common):
                    continue
                ua = pa.torso[ia] @ a.R
                ub = pb.torso[ib] @ b.R
                n = np.cross(ua, ub)
                nn = np.linalg.norm(n, axis=-1)
                good = nn > 1e-6
                gaps.append(np.abs(n[good] @ w) / nn[good])
            if gaps:
                med[off] = float(np.median(np.concatenate(gaps)))
        if med:
            best = min(med, key=med.get)
            out[f"{a.name}-{b.name}"] = {
                "best_offset": best,
                "median_gap_cm_at_0": round(med.get(0, float("nan")) * 100, 2),
                "median_gap_cm_at_best": round(med[best] * 100, 2),
                "people": len(shared),
            }
    return out


# ---------------------------------------------------------------------------
# 2. floor
# ---------------------------------------------------------------------------

def estimate_floor(cams: List[CamInput]) -> Dict[str, Any]:
    """Floor from the lowest foot point of every person-frame (triangulated
    scales only), in the reference camera frame where 'up' is -z (it looks
    straight down)."""
    lows, pts, per_cam = [], [], {}
    for c in cams:
        own = []
        for cp in c.people.values():
            if cp.source != "triangulated":
                continue
            feet = cp.feet * cp.scale                                # camera space, m
            feet_rig = (feet - c.t) @ c.R                            # rig = reference camera frame
            height = -feet_rig[..., 2]                               # along 'up'
            low = height.argmin(axis=1)
            own.append(height[np.arange(len(low)), low])
            pts.append(feet_rig[np.arange(len(low)), low])
        if own:
            per_cam[c.name] = round(-float(np.median(np.concatenate(own))), 3)
            lows.append(np.concatenate(own))
    low = np.concatenate(lows)
    pts = np.concatenate(pts)
    floor = float(np.median(low))
    # Diagnostic only: tilt of a plane fitted to the lowest feet near the floor
    near = np.abs(low - floor) < 0.15
    tilt = float("nan")
    if near.sum() >= 10:
        Amat = np.c_[pts[near, 0], pts[near, 1], np.ones(near.sum())]
        coef, *_ = np.linalg.lstsq(Amat, pts[near, 2], rcond=None)
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    return {
        "camera_height_m": round(-floor, 3),
        "lowest_foot_height_p25_p75_m": [round(float(v) - floor, 3) for v in np.percentile(low, [25, 75])],
        "person_frames": int(low.size),
        "per_camera_height_m": per_cam,
        "fitted_floor_tilt_deg": round(tilt, 2),
    }


# ---------------------------------------------------------------------------
# 3-4. contact + export
# ---------------------------------------------------------------------------

def export_camera(c: CamInput, extr_path: str, args, ground_cfg: Optional[GroundOptConfig]
                  ) -> Dict[str, Any]:
    ws = float(args.world_scale)
    extr = load_extrinsics_json(extr_path, device=torch.device("cpu"))
    R = extr.R.numpy().astype(np.float32)
    t = extr.t.numpy().reshape(1, 3).astype(np.float32)
    mesh_in = os.path.join(c.in_dir, "meshes_4d_individual")
    mesh_out = os.path.join(c.out_dir, "meshes_4d_individual")
    os.makedirs(mesh_out, exist_ok=True)
    contact, heights = {}, {}
    for pid in sorted(c.people):
        cp = c.people[pid]
        with np.load(os.path.join(mesh_in, f"{pid}.npz")) as z:
            verts, kps, cam_t = z["vertices"], z["keypoints"], z["cam_t"].astype(np.float32)
            faces, frame_names = z["faces"], z["frame_names"]
        verts -= cam_t[:, None, :]                  # local mesh
        kps = kps - cam_t[:, None, :]
        if ground_cfg is not None:
            t_out, _, m = optimize_ground_ray_scale(
                extr=extr, X3d=torch.from_numpy(kps), t0=torch.from_numpy(cam_t),
                frame_numbers=torch.from_numpy(cp.frames), cfg=ground_cfg, scale=cp.scale)
            t_out = t_out.numpy().astype(np.float32)
            contact[str(pid)] = m
        else:
            t_out = cam_t * np.float32(cp.scale)
        verts *= np.float32(cp.scale)
        verts += t_out[:, None, :]
        kps = kps * np.float32(cp.scale) + t_out[:, None, :]
        verts *= np.float32(ws)
        verts -= t
        v_world = verts @ R
        k_world = (kps * ws - t) @ R
        z_ = v_world[..., 2]
        heights[pid] = (cp.frames, z_.max(axis=1) - z_.min(axis=1))
        np.savez_compressed(os.path.join(mesh_out, f"{pid}.npz"), vertices=v_world.astype(np.float32, copy=False),
                            keypoints=k_world.astype(np.float32), cam_t=t_out, faces=faces, frame_names=frame_names)
        del verts, v_world
    plot_mesh_heights(heights, os.path.join(c.out_dir, "mesh_heights_world.png"),
                      title=f"Mesh Height (World Space) - Stage 3 multi-view, {c.name}",
                      world_units="cm" if ws == 100 else "world units")
    return {"contact": contact,
            "median_height": {str(pid): round(float(np.median(h)), 1) for pid, (_, h) in heights.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="camera-space, no-ground Stage 3 outputs of one rig and batch")
    ap.add_argument("--calib", required=True, help="rig calibration (parameters-camera-AA-to-BB.json)")
    ap.add_argument("--label", default="stage3_multiview", help="output folder name, next to each input")
    ap.add_argument("--world-scale", type=float, default=100.0, help="world units per metre (100 = cm)")
    ap.add_argument("--max-ray-gap", type=float, default=0.3, help="m; samples whose rays miss more are left out")
    ap.add_argument("--min-ray-angle", type=float, default=5.0, help="deg; narrower ray pairs are left out")
    ap.add_argument("--min-samples", type=int, default=60,
                    help="triangulated (frame, torso point) samples needed for a person's own scale")
    ap.add_argument("--no-contact", action="store_true", help="skip the per-frame floor contact")
    ap.add_argument("--ground-iters", type=int, default=GroundOptConfig.iters)
    args = ap.parse_args()

    rig_cams = load_rig_calibration(args.calib)
    rig = {c.name: c for c in rig_cams}
    recordings = {(session_of(camera_number(d, o)), batch_number(d, o))
                  for d, o in ((d, read_run_options(d)) for d in args.inputs)}
    if len(recordings) > 1:
        raise SystemExit(f"[ERROR] Inputs from different recordings (session, batch): {sorted(recordings, key=str)}; "
                         "only cameras of one session (cam06-10 / cam01-05) and batch show the same moments")
    cams = [load_camera(d, args.label, rig) for d in args.inputs]
    names = [c.name for c in cams]
    if len(set(names)) != len(names):
        raise SystemExit(f"[ERROR] Several inputs of the same camera: {names}")
    if len(cams) < 2:
        raise SystemExit("[ERROR] Needs at least two cameras")
    session, batch = next(iter(recordings))
    print(f"[INFO] Session {session}, batch {batch}, rig {os.path.basename(args.calib)}; cameras: "
          + ", ".join(f"{c.name} ({len(c.people)} people)" for c in cams), flush=True)

    sync = sync_check(cams)
    for pair, s in sync.items():
        # Slow movement makes the curve flat near 0; only a clear dip elsewhere is a warning
        off_sync = s["median_gap_cm_at_best"] < 0.8 * s["median_gap_cm_at_0"]
        flag = "  <-- clearly smaller away from offset 0: frames may be out of sync" if off_sync else ""
        print(f"[INFO] sync {pair}: ray gap {s['median_gap_cm_at_0']} cm at offset 0, "
              f"best {s['median_gap_cm_at_best']} cm at {s['best_offset']}{flag}")

    cam_median = estimate_scales(cams, args.max_ray_gap, args.min_ray_angle, args.min_samples)
    for c in cams:
        n_tri = sum(cp.source == "triangulated" for cp in c.people.values())
        print(f"[INFO] {c.name}: {n_tri}/{len(c.people)} people triangulated, "
              f"median scale {cam_median[c.name]:.3f}")

    floor = estimate_floor(cams)
    H = floor["camera_height_m"]
    print(f"[INFO] Floor: {rig_cams[0].name} {H:.3f} m above it (per camera: {floor['per_camera_height_m']}), "
          f"fitted tilt {floor['fitted_floor_tilt_deg']} deg", flush=True)

    source = (f"{os.path.basename(args.calib)}; {rig_cams[0].name} looking straight down, height "
              f"{H:.3f} m estimated from the people (stage3_multiview.py)")
    extr = ground_extrinsics(rig_cams, H, args.world_scale, source)
    ground_cfg = None if args.no_contact else GroundOptConfig(iters=int(args.ground_iters),
                                                              world_scale=float(args.world_scale))

    summary: Dict[str, Any] = {
        "session": session,
        "batch": batch,
        "calib": os.path.abspath(args.calib),
        "options": vars(args),
        "sync": sync,
        "floor": floor,
        "camera_median_scale": cam_median,
        "cameras": {},
    }
    for c in cams:
        ext_dir = os.path.join(c.out_dir, "extrinsics")
        os.makedirs(ext_dir, exist_ok=True)
        for name, d in extr.items():
            with open(os.path.join(ext_dir, extrinsics_filename(name)), "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
        extr_path = os.path.join(ext_dir, extrinsics_filename(c.name))
        print(f"[INFO] {c.name}: exporting {len(c.people)} people -> {c.out_dir}", flush=True)
        res = export_camera(c, extr_path, args, ground_cfg)
        summary["cameras"][c.name] = {
            "input": os.path.abspath(c.in_dir),
            "output": os.path.abspath(c.out_dir),
            "people": {str(pid): {"scale": round(cp.scale, 4), "source": cp.source, **cp.stats,
                                  "median_height": res["median_height"].get(str(pid))}
                       for pid, cp in sorted(c.people.items())},
            "contact": res["contact"],
        }
        with open(os.path.join(c.out_dir, "stage3_run_options.txt"), "w", encoding="utf-8") as fh:
            for k, v in {**c.opts, "mode": "multiview", "input": os.path.abspath(c.in_dir),
                         "extrinsics_json": extr_path, "export_camera_space": False,
                         "world_scale": args.world_scale, "calib": os.path.abspath(args.calib)}.items():
                fh.write(f"{k}: {v}\n")
    for c in cams:
        with open(os.path.join(c.out_dir, "stage3_multiview_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print(f"[INFO] Done: {', '.join(c.out_dir for c in cams)}")


if __name__ == "__main__":
    main()
