#!/usr/bin/env python3
"""
Checks of the camera layer (utils/fisheye.py, utils/cosilab_camera.py), CPU only, no images:
  1. fisheye project/unproject round trip at full and half resolution, for every camera
  2. a virtual camera set up on a box renders the box centre at the crop centre (render_crops on a synthetic image)
  3. libCalib rig calibration (parameters-camera-AA-to-BB.json, the old Stage 3 multi-view input) vs the 4TU idiap
     extrinsics: camera centre distance (m) and rotation angle (deg) per camera, both in their reference camera's frame
  4. with --mhr-transform: utils/mhr_transform.py round trip (needs no model)

python scripts/check_cameras.py [--libcalib-dir <dir with parameters-camera-*-to-*.json>]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils import cosilab_camera as cc  # noqa: E402
from utils import fisheye  # noqa: E402


def check_roundtrip(cam: cc.Camera) -> float:
    rng = np.random.default_rng(0)
    w, h = cam.size
    pix = np.stack([rng.uniform(0.05 * w, 0.95 * w, 500), rng.uniform(0.05 * h, 0.95 * h, 500)], 1)
    rays = cam.rays(pix)
    back = cam.project(cam.centre + 3.0 * rays)
    return float(np.abs(back - pix).max())


def check_crop_centre(cam: cc.Camera) -> float:
    """Render a crop of a synthetic image whose value is the pixel x coordinate; the crop centre must read the box
    centre's x (up to the bilinear interpolation)."""
    w, h = cam.size
    box = np.array([[0.7 * w, 0.2 * h, 0.8 * w, 0.45 * h]])
    R_v, f_v = fisheye.setup_virtual_cameras(box, cam.K, cam.D, crop_size=256)
    xs = torch.arange(w, dtype=torch.float32)[None, None].expand(1, h, w)
    crop = fisheye.render_crops(xs, torch.tensor(R_v, dtype=torch.float32), torch.tensor(f_v, dtype=torch.float32),
                                torch.tensor(cam.K, dtype=torch.float32), torch.tensor(cam.D, dtype=torch.float32))
    centre_ray = R_v[0, 2]  # the virtual z axis in camera coordinates
    expected = fisheye.project_np(centre_ray, cam.K, cam.D)[0]
    got = float(crop[0, 0, 128, 128])
    return abs(got - expected)


def libcalib_vs_idiap(libcalib_dir: str) -> None:
    sys.path.insert(0, os.path.join(REPO_DIR, 'scripts'))
    from scripts.view_stage3 import load_rig_calibration
    for path in sorted(glob.glob(os.path.join(libcalib_dir, 'parameters-camera-*-to-*.json'))):
        print(f'{os.path.basename(path)}:')
        for rc in load_rig_calibration(path):
            cam = int(rc.name[3:])
            try:
                ic = cc.load_camera(cam)
            except FileNotFoundError as e:
                print(f'  {rc.name}: {e}')
                continue
            c_lib = -rc.R.T @ rc.t
            dR = rc.R @ ic.R.T
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            print(f'  {rc.name}: centre libCalib {c_lib.round(3)} idiap {ic.centre.round(3)} '
                  f'-> {np.linalg.norm(c_lib - ic.centre):.3f} m apart, rotated {ang:.2f} deg; '
                  f'f libCalib {rc.f:.1f} idiap {ic.K[0, 0]:.1f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--libcalib-dir', default=os.path.join(cc.PROJECT_ROOT, 'data_temp', 'keypoints', 'extrinsics'))
    ap.add_argument('--mhr-transform', action='store_true')
    args = ap.parse_args()

    worst = 0.0
    for session in (1, 2):
        for c in cc.session_cameras(session):
            cam = cc.load_camera(c)
            r_full, r_half = check_roundtrip(cam), check_roundtrip(cam.scaled(0.5))
            crop = check_crop_centre(cam.scaled(0.5))
            worst = max(worst, r_full, r_half)
            print(f'{cam.name} (session {session}): round trip max {r_full:.2e} px (full), {r_half:.2e} px (half); '
                  f'crop centre off by {crop:.3f} px; centre {cam.centre.round(3)} m')
    print(f'[{"OK" if worst < 1e-3 else "FAIL"}] fisheye round trip, worst {worst:.2e} px')
    if os.path.isdir(args.libcalib_dir):
        libcalib_vs_idiap(args.libcalib_dir)
    if args.mhr_transform:
        from utils.mhr_transform import self_check
        self_check()


if __name__ == '__main__':
    main()
