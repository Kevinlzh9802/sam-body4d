#!/usr/bin/env python3
"""
Re-poses the world bodies of a stage3_mvfit.py output (world/<pid>.npz, stored without vertices unless the fit ran
with --save-vertices) with MHR, so scripts/view_stage3.py can show them with numpy + viser only (the `viz` group):

  <out>/<pid>.npz   vertices (n, V, 3) float32 cm in the floor frame, faces, frame_names ('%08d', the kept frames)

These are the keys of Stage 3's meshes_4d_individual/, uncompressed so the viewer streams the rows it shows.
About 0.22 MB per person-frame: limit with --frames / --stride / --ids. Reads only world/*.npz and the MHR
checkpoint (no images or masks). Needs torch and the checkpoints, so run it in the container (CPU is fine):

  bash scripts/local_sif.sh python scripts/export_mvfit_vertices.py <mvfit dir> [--frames 6000 6300] [--stride 2]
  uv run --no-sync python scripts/view_stage3.py <mvfit dir>

Default --out: experiments/mvfit_viewer/<mvfit dir name>/ (local disk; scripts/view_stage3.py looks there).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.mhr_only import load_mhr_head  # noqa: E402
from utils.world_bodies import WorldBody  # noqa: E402

CHUNK = 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mvfit_dir', help='stage3_mvfit.py output (with world/)')
    ap.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'), help='frame numbers [START, END)')
    ap.add_argument('--stride', type=int, default=1, help='keep every n-th frame')
    ap.add_argument('--ids', type=int, nargs='+', help='only these person IDs')
    ap.add_argument('--out', help='output dir (default: experiments/mvfit_viewer/<mvfit dir name>/)')
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    args = ap.parse_args()

    wdir = os.path.join(args.mvfit_dir, 'world')
    paths = {int(f[:-4]): os.path.join(wdir, f) for f in sorted(os.listdir(wdir)) if f.endswith('.npz')}
    if args.ids:
        paths = {pid: p for pid, p in paths.items() if pid in set(args.ids)}
    if not paths:
        raise SystemExit(f'[ERROR] No bodies in {wdir} (check --ids).')
    out = args.out or os.path.join(REPO_DIR, 'experiments', 'mvfit_viewer',
                                   os.path.basename(os.path.normpath(args.mvfit_dir)))
    if os.path.realpath(out).startswith('/tudelft.net/'):
        raise SystemExit(f'[ERROR] --out {out} is on /tudelft.net; write the export to local disk.')
    os.makedirs(out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head, faces = load_mhr_head(OmegaConf.load(args.config), device)
    print(f'[INFO] MHR on {device}; {len(paths)} people -> {out}', flush=True)
    total = 0
    for pid, path in paths.items():
        t0 = time.time()
        body = WorldBody(pid, path)
        keep = np.ones(len(body.frames), dtype=bool)
        if args.frames:
            keep &= (body.frames >= args.frames[0]) & (body.frames < args.frames[1])
        keep &= (body.frames - (args.frames[0] if args.frames else 0)) % args.stride == 0
        rows = np.nonzero(keep)[0]
        if len(rows) == 0:
            print(f'[INFO] id {pid}: no frames in range', flush=True)
            continue
        with np.load(path) as z:
            faces_p = z['faces'] if 'faces' in z.files else faces
        parts = []
        for a in range(0, len(rows), CHUNK):
            v, _ = body.floor_points(head, rows[a:a + CHUNK], device)
            parts.append(v.cpu().numpy().astype(np.float32))
        verts = np.concatenate(parts)
        dst = os.path.join(out, f'{pid}.npz')
        np.savez(dst, vertices=verts, faces=faces_p.astype(np.int32),
                 frame_names=np.array([f'{f:08d}' for f in body.frames[rows]]))
        z = verts[..., 2]
        total += os.path.getsize(dst)
        print(f'[INFO] id {pid}: {len(rows)} frames, {verts.shape[1]} verts, lowest z median {np.median(z.min(1)):.1f} cm, '
              f'height median {np.median(z.max(1) - z.min(1)):.1f} cm, {os.path.getsize(dst) / 1e6:.0f} MB, '
              f'{time.time() - t0:.0f} s', flush=True)
    print(f'[INFO] {total / 1e9:.2f} GB in {out} (--stride / --frames to shrink). View: '
          f'uv run --no-sync python scripts/view_stage3.py {args.mvfit_dir}', flush=True)


if __name__ == '__main__':
    main()
