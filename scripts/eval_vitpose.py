#!/usr/bin/env python3
"""
Independent 2D check of Stage 3 bodies against the dataset's ViTPose keypoints (4TU keypoints/vitpose_results, the
annotated cameras 01, 03, 05, 06, 08, 10), which neither sam-body4d stage uses. Numbers only.

Inputs, any mix:
  --mvfit <dir>            a stage3_mvfit.py output: the world bodies, projected into every camera given in --cams
  --single <seq>=<dir>     a single-view stage3_mhr.py output in camera space (meshes_4d_individual/<pid>.npz with
                           keypoints (T, 70, 3) in metres), e.g. the --fisheye baseline; projected with its camera
Per camera and source: pixel error (full 1920x1080 resolution) of the 15 matching joints with ViTPose confidence >=
--conf (nose, neck, shoulders, elbows, wrists, hips, knees, ankles, feet = big toes), median / p90, and how many
person-frames match. ViTPose's head keypoint has no MHR counterpart and is left out.

python scripts/eval_vitpose.py --mvfit <dir> --cams 1 3 [--single cam01_batch01=<stage3 dir>] [--frames 6000 6300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import cosilab_camera as cc  # noqa: E402
from utils import fisheye  # noqa: E402

# ConfLab-17 (the dataset ViTPose layout) -> mhr70; None: no counterpart (head)
CONFLAB_TO_MHR70 = [None, 0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 18, 15]
KP_SCALE = 2.0  # the ViTPose results are in 960x540


def load_vitpose(seq: str) -> Dict[int, Dict[str, np.ndarray]]:
    path = os.path.join(cc.DATA_ROOT, 'keypoints', 'vitpose_results', seq, 'vitpose_keypoints.json')
    with open(path) as fh:
        ann = json.load(fh)['annotations']
    tracks: Dict[int, dict] = {}
    for key in sorted(ann, key=int):
        for pid, kp in ann[key].get('keypoints', {}).items():
            t = tracks.setdefault(int(pid), {'frames': [], 'kp': []})
            t['frames'].append(int(key))
            t['kp'].append(kp)
    out = {}
    for pid, t in tracks.items():
        kp = np.array(t['kp'], dtype=np.float64)
        kp[..., :2] *= KP_SCALE
        out[pid] = {'frames': np.array(t['frames']), 'kp': kp}
    return out


def errors(vit, pid: int, frames: np.ndarray, px: np.ndarray, conf: float, lo: int, hi: int) -> np.ndarray:
    """px: (n, 70, 2) full-res pixels of the body at `frames`; returns the per-joint errors (px) of the matches."""
    t = vit.get(pid)
    if t is None:
        return np.zeros(0)
    keep = (frames >= lo) & (frames < hi)
    common, ia, ib = np.intersect1d(frames[keep], t['frames'], return_indices=True)
    if not len(common):
        return np.zeros(0)
    ours = px[keep][ia]
    kp = t['kp'][ib]
    cols = [i for i, m in enumerate(CONFLAB_TO_MHR70) if m is not None]
    mh = [CONFLAB_TO_MHR70[i] for i in cols]
    d = np.linalg.norm(ours[:, mh] - kp[:, cols, :2], axis=-1)
    ok = kp[:, cols, 2] >= conf
    return d[ok]


def summarise(e: np.ndarray) -> str:
    if not e.size:
        return 'no matches'
    return f'median {np.median(e):6.1f} px, p90 {np.percentile(e, 90):6.1f} px, {e.size} joints'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mvfit', default=None)
    ap.add_argument('--cams', type=int, nargs='*', default=None, help='cameras for --mvfit (default: its inputs)')
    ap.add_argument('--single', nargs='*', default=[], help='<camXX_batchBB>=<stage3_mhr.py output dir>')
    ap.add_argument('--frames', type=int, nargs=2, default=None, help='batch frame range [a, b)')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    ap.add_argument('--out', default=None, help='JSON with the numbers')
    args = ap.parse_args()
    lo, hi = args.frames or (0, 10 ** 9)
    result = {}

    if args.mvfit:
        from utils.mhr_only import load_mhr_head
        from utils.world_bodies import WorldBody, camera_from_floor, load_mvfit, project
        summary, paths = load_mvfit(args.mvfit)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        head_pose, _ = load_mhr_head(OmegaConf.load(args.config), device)
        bodies = [WorldBody(pid, p) for pid, p in paths.items()]
        cams = args.cams or sorted({int(os.path.basename(os.path.dirname(p))[3:5]) for p in summary['inputs']})
        kps = {}
        for b in bodies:
            rows = np.nonzero((b.frames >= lo) & (b.frames < hi))[0]
            ks = []
            for s in range(0, len(rows), 512):
                _, k = b.floor_points(head_pose, rows[s:s + 512], device)
                ks.append(k.cpu().numpy().astype(np.float64))
            kps[b.pid] = (b.frames[rows], np.concatenate(ks) if ks else np.zeros((0, 70, 3)), b)
        for c in cams:
            seq = f'cam{c:02d}_batch{int(summary["batch"]):02d}'
            try:
                vit = load_vitpose(seq)
            except FileNotFoundError:
                print(f'[INFO] {seq}: no dataset ViTPose keypoints')
                continue
            cam = cc.load_camera(c)
            Rt = camera_from_floor(cam, summary)
            es, es_seen = [], []
            for pid, (fr, k, b) in kps.items():
                if not len(fr):
                    continue
                px, _ = project(cam, Rt, k)
                e = errors(vit, pid, fr, px, args.conf, lo, hi)
                es.append(e)
                if cam.name in set(map(str, b.p.get('cameras', []))):
                    es_seen.append(e)
            e = np.concatenate(es) if es else np.zeros(0)
            e_seen = np.concatenate(es_seen) if es_seen else np.zeros(0)
            result[f'mvfit {seq}'] = {'all': summarise(e), 'fitted_in_this_camera': summarise(e_seen),
                                      'median_px': float(np.median(e)) if e.size else None}
            print(f'[mvfit ] {seq}: {summarise(e)}  (people fitted with this camera: {summarise(e_seen)})')

    for item in args.single:
        seq, d = item.split('=', 1)
        cam_no = cc.parse_seq(seq)[0]
        cam = cc.load_camera(cam_no)
        vit = load_vitpose(seq)
        mesh_dir = os.path.join(d, 'meshes_4d_individual')
        es = []
        for fn in sorted(os.listdir(mesh_dir)):
            if not fn.endswith('.npz'):
                continue
            with np.load(os.path.join(mesh_dir, fn)) as z:
                k = z['keypoints'].astype(np.float64)
                fr = np.array([int(f) for f in z['frame_names']])
            px = fisheye.project_np(k, cam.K, cam.D)
            es.append(errors(vit, int(fn[:-4]), fr, px, args.conf, lo, hi))
        e = np.concatenate(es) if es else np.zeros(0)
        result[f'single {seq}'] = {'all': summarise(e), 'median_px': float(np.median(e)) if e.size else None}
        print(f'[single] {seq}: {summarise(e)}')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(result, fh, indent=1)


if __name__ == '__main__':
    main()
