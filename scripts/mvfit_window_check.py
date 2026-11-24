#!/usr/bin/env python3
"""
Jumps at the chunk boundaries of a stage3_mvfit.py fit (as Video-OnlineHMR scripts/cosilab_window_check.py): per
person, the frame-to-frame step of the world translation (mm) and of the root rotation (deg) at the chunk boundaries
(--window of the fit, from summary.json) vs at all other frames. With one optimisation over all frames (summed chunk
gradients), boundary steps should look like any other step. Numbers only.

python scripts/mvfit_window_check.py <mvfit dir>
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)


def runs(frames):
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    b = np.concatenate([[0], breaks, [len(frames)]])
    return list(zip(b[:-1], b[1:]))


def main() -> None:
    d = sys.argv[1]
    summary = json.load(open(os.path.join(d, 'summary.json')))
    window = int(summary['config']['window'])
    wd = os.path.join(d, 'world')
    at, other = {'trans': [], 'rot': []}, {'trans': [], 'rot': []}
    for fn in sorted(os.listdir(wd)):
        if not fn.endswith('.npz'):
            continue
        z = np.load(os.path.join(wd, fn))
        frames = z['frame_names'].astype(np.int64)
        T, G = z['trans_cm'].astype(np.float64) * 10, z['g_world'].astype(np.float64)
        bounds = set()
        for s, e in runs(frames):
            n = int(math.ceil((e - s) / window))
            bounds |= set(np.linspace(s, e, n + 1).round().astype(int)[1:-1].tolist())
        for s, e in runs(frames):
            for i in range(s + 1, e):  # step from frame i-1 to i
                dt = float(np.linalg.norm(T[i] - T[i - 1]))
                tr = np.trace(G[i - 1].T @ G[i])
                dr = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
                tgt = at if i in bounds else other
                tgt['trans'].append(dt)
                tgt['rot'].append(dr)
    for k, unit in (('trans', 'mm'), ('rot', 'deg')):
        a, o = np.array(at[k]), np.array(other[k])
        if not a.size:
            print(f'{k}: no chunk boundaries (window {window})')
            continue
        print(f'{k} step ({unit}): at {a.size} boundaries median {np.median(a):.2f} / p99 {np.percentile(a, 99):.2f}; '
              f'elsewhere ({o.size}) median {np.median(o):.2f} / p99 {np.percentile(o, 99):.2f}')


if __name__ == '__main__':
    main()
