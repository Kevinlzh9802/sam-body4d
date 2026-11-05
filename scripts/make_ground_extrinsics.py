#!/usr/bin/env python3
"""
Floor extrinsics for the cosilab ceiling rigs, without ground markers.

Assumes the rig's reference camera (the first camera of a libCalib
parameters-camera-AA-to-BB.json, i.e. cam01 or cam06) hangs --height metres
above a flat floor and looks straight down. The other cameras of the rig
follow from the calibration. World frame: floor z=0, z up, x along the
reference camera's image x, y against its image y (right-handed).

Writes one extrinsic_cam_<MM>.json per camera (world_to_camera, translation
in cm to match Stage 3's --world-scale 100), the file name Stage 3's
launchers and scripts/local_stage3.sh look for.

Usage:
  python scripts/make_ground_extrinsics.py experiments/extrinsics/parameters-camera-06-to-10.json \
      experiments/extrinsics/parameters-camera-01-to-04.json --height 3.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_stage3 import load_rig_calibration  # noqa: E402

# Reference camera looking straight down: camera x = world x, camera y = -world y,
# camera z (viewing direction) = -world z
R_DOWN = np.diag([1.0, -1.0, -1.0])


def ground_extrinsics(cams, height: float, world_scale: float, source: str) -> Dict[str, dict]:
    """extrinsic_cam_<MM>.json contents per camera of a rig (load_rig_calibration),
    with the rig's reference camera ``height`` m above the floor looking straight down."""
    if np.abs(cams[0].R - np.eye(3)).max() > 1e-6 or np.abs(cams[0].t).max() > 1e-6:
        raise ValueError("the first camera is not the rig reference")
    # reference camera: x_ref = R_DOWN x_world + t_ref, centre at (0, 0, height)
    t_ref = -R_DOWN @ np.array([0.0, 0.0, height])
    out = {}
    for cam in cams:
        # x_cam = R_c x_ref + t_c  ->  x_cam = (R_c R_DOWN) x_world + (R_c t_ref + t_c)
        R = cam.R @ R_DOWN
        t = (cam.R @ t_ref + cam.t) * world_scale
        out[cam.name] = {
            "extrinsic_type": "world_to_camera",
            "rotation": R.tolist(),
            "translation": t.tolist(),
            "units": "cm" if world_scale == 100 else f"1/{world_scale} m",
            "camera_center_world_m": (-R.T @ t / world_scale).tolist(),
            "source": source,
        }
    return out


def extrinsics_filename(cam_name: str) -> str:
    return f"extrinsic_{cam_name.replace('cam', 'cam_')}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calibs", nargs="+", help="rig calibrations (parameters-camera-AA-to-BB.json)")
    ap.add_argument("--height", type=float, default=3.5, help="reference camera height above the floor (m)")
    ap.add_argument("--world-scale", type=float, default=100.0, help="world units per metre (Stage 3: 100 = cm)")
    ap.add_argument("--out-dir", help="default: the folder of each calibration")
    args = ap.parse_args()

    for calib in args.calibs:
        cams = load_rig_calibration(calib)
        if not cams[0].name.startswith("cam"):
            raise SystemExit(f"[ERROR] Cannot tell the camera numbers from {calib}")
        source = (f"{os.path.basename(calib)}; {cams[0].name} assumed {args.height} m above "
                  "the floor, looking straight down (no ground markers)")
        try:
            extr = ground_extrinsics(cams, args.height, args.world_scale, source)
        except ValueError as e:
            raise SystemExit(f"[ERROR] {calib}: {e}")
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(calib))
        os.makedirs(out_dir, exist_ok=True)
        for name, d in extr.items():
            path = os.path.join(out_dir, extrinsics_filename(name))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
            view = np.asarray(d["rotation"]).T @ [0, 0, 1]
            print(f"[INFO] {path}: centre {np.round(d['camera_center_world_m'], 3).tolist()} m, "
                  f"view dir {np.round(view, 3).tolist()}")


if __name__ == "__main__":
    main()
