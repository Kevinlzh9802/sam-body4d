#!/usr/bin/env python3
"""
Stage 1 prompts for a cosilab camera without annotations (side cams 12-15, top 02/04/07/09): the world bodies of a
stage3_mvfit.py run over other cameras, projected into this camera at the first frame of every 10 s segment (where
Stage 1 prompts SAM-3), written as Stage 1 annotation JSONs (one box per person, label = real person ID).

Only people with at least --min-visible of their projected silhouette not covered by a nearer person get a box
(people standing behind others in the side views would give SAM-3 the wrong person). A person missing at a segment's
first frame (not fitted there) gets no box in that segment, as in the annotated cameras.

Boxes are written in full-resolution pixels (Stage 1 scales them by --cam-res-scale 0.5 to the 960x540 segments).
The camera's extrinsics can be replaced (--extrinsics cam12=<json>), e.g. by a refit.

python scripts/project_prompts.py --mvfit <mvfit dir> --cam 13 --batch 1 [--segments 11] [--out-root ...]
Output: <out-root>/camXX_batchBB/camXX_segNNN.json and prompts_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.make_raw_segments import SEG_FRAMES, parse_range  # noqa: E402
from utils import cosilab_camera as cc  # noqa: E402
from utils.mhr_only import load_mhr_head  # noqa: E402
from utils.world_bodies import WorldBody, camera_from_floor, load_mvfit, visible_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mvfit', required=True, help='stage3_mvfit.py output folder (summary.json, world/)')
    ap.add_argument('--cam', type=int, required=True)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--segments', default=None, help='1-based segment numbers (default: all with fitted bodies)')
    ap.add_argument('--out-root', default=os.path.join(cc.OUTPUT_ROOT, 'video_frame_annotations'))
    ap.add_argument('--extrinsics', nargs='*', default=[], help='cam12=<rvec/tvec json>')
    ap.add_argument('--min-visible', type=float, default=0.5)
    ap.add_argument('--min-inside', type=float, default=0.5, help='fraction of the box that must be in the image')
    ap.add_argument('--pad', type=float, default=0.05, help='box padding, fraction of its size per side')
    ap.add_argument('--image-scale', type=float, default=0.5, help='Stage 1 image size / 1920x1080')
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    args = ap.parse_args()

    summary, paths = load_mvfit(args.mvfit)
    if int(summary['batch']) != args.batch:
        raise SystemExit(f"[ERROR] the fit is of batch {summary['batch']}, not {args.batch}")
    cam = cc.load_camera(args.cam, cc.parse_overrides(args.extrinsics).get(args.cam))
    if cam.session != int(summary['session']):
        raise SystemExit(f"[ERROR] cam{args.cam:02d} is in session {cam.session}, the fit in {summary['session']}")
    cam = cam.scaled(args.image_scale)
    Rt = camera_from_floor(cam, summary)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_pose, _ = load_mhr_head(OmegaConf.load(args.config), device)
    bodies = [WorldBody(pid, p) for pid, p in paths.items()]
    n_seg = cc.BATCH_FRAMES // SEG_FRAMES
    segs = parse_range(args.segments, n_seg) if args.segments else list(range(1, n_seg + 1))

    seq = f'cam{args.cam:02d}_batch{args.batch:02d}'
    out_dir = os.path.join(args.out_root, seq)
    os.makedirs(out_dir, exist_ok=True)
    W, H = cam.size
    report = {'mvfit': os.path.abspath(args.mvfit), 'camera': cam.name, 'extrinsics': cam.extrinsics_file,
              'min_visible': args.min_visible, 'segments': {}}
    for s in segs:
        f0 = (s - 1) * SEG_FRAMES
        people = []
        for b in bodies:
            r = b.rows([f0])
            if r[0] < 0:
                continue
            V, _ = b.floor_points(head_pose, r, device)
            people.append((b.pid, V[0].cpu().numpy().astype(np.float64)))
        if not people:
            continue
        _, sil, vis = visible_labels(cam, Rt, people)
        shapes, skipped = [], {}
        for pid, m in sil.items():
            ys, xs = np.nonzero(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            pw, ph = (x1 - x0) * args.pad, (y1 - y0) * args.pad
            box = np.array([x0 - pw, y0 - ph, x1 + pw, y1 + ph])
            clipped = np.clip(box, 0, [W, H, W, H])
            inside = ((clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
                      / max((box[2] - box[0]) * (box[3] - box[1]), 1e-6))
            if vis[pid] < args.min_visible or inside < args.min_inside:
                skipped[pid] = {'visible': round(vis[pid], 3), 'inside': round(float(inside), 3)}
                continue
            full = clipped / args.image_scale
            shapes.append({'label': int(pid), 'points': [[float(full[0]), float(full[1])], [float(full[2]), float(full[3])]],
                           'shape_type': 'rectangle', 'visible_fraction': round(vis[pid], 3)})
        key = f'cam{args.cam:02d}_seg{s:03d}'
        with open(os.path.join(out_dir, f'{key}.json'), 'w') as fh:
            json.dump({'shapes': shapes, 'imageWidth': round(W / args.image_scale),
                       'imageHeight': round(H / args.image_scale), 'batch_frame': f0,
                       'source': f'scripts/project_prompts.py from {os.path.abspath(args.mvfit)}'}, fh, indent=1)
        report['segments'][key] = {'frame': f0, 'prompted': [sh['label'] for sh in shapes], 'skipped': skipped}
        print(f'[INFO] {key} (frame {f0}): {len(shapes)} prompts {sorted(sh["label"] for sh in shapes)}, '
              f'{len(skipped)} left out (occluded / outside) {sorted(skipped)}', flush=True)
    with open(os.path.join(out_dir, 'prompts_summary.json'), 'w') as fh:
        json.dump(report, fh, indent=1)


if __name__ == '__main__':
    main()
