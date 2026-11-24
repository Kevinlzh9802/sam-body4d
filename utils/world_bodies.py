"""
The world bodies of a stage3_mvfit.py output, posed again from their stored MHR parameters (so callers don't need the
large vertex arrays), and seen from any cosilab camera, e.g. one the fit didn't use (side cameras, cam02/04/...).

Floor frame (the export frame): X_floor_cm = 100 Rf (X_world - o), with Rf, o in summary.json 'floor_frame'.
A world body at frame t: X_floor_cm = g_world[t] (size_factor F M_t) * 100 + trans_cm[t] (see smoothing/mv_fit.py).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from smoothing.mv_fit import F_AXES, MHR_KEYS
from utils import fisheye


def load_mvfit(mvfit_dir: str) -> Tuple[dict, Dict[int, str]]:
    with open(os.path.join(mvfit_dir, 'summary.json')) as fh:
        summary = json.load(fh)
    if 'floor_frame' not in summary:
        raise SystemExit(f'[ERROR] {mvfit_dir}: summary.json has no floor_frame (made by an older stage3_mvfit.py)')
    wdir = os.path.join(mvfit_dir, 'world')
    paths = {int(f[:-4]): os.path.join(wdir, f) for f in sorted(os.listdir(wdir)) if f.endswith('.npz')}
    return summary, paths


class WorldBody:
    PARAM_KEYS = ('g_world', 'trans_cm', 'body_pose', 'hand', 'face', 'shape', 'scale', 'size_factor', 'frame_names',
                  'cameras')

    def __init__(self, pid: int, path: str):
        self.pid = pid
        with np.load(path) as z:
            self.p = {k: z[k] for k in self.PARAM_KEYS if k in z.files}
        self.frames = self.p['frame_names'].astype(np.int64)
        self.index = {int(f): i for i, f in enumerate(self.frames)}

    def rows(self, frames) -> np.ndarray:
        return np.array([self.index.get(int(f), -1) for f in frames])

    @torch.no_grad()
    def floor_points(self, head_pose, rows: np.ndarray, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vertices (n, V, 3) and mhr70 keypoints (n, 70, 3) in the floor frame (cm) at the body's rows."""
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        n = len(rows)
        p = {'body_pose': t(self.p['body_pose'][rows]), 'hand': t(self.p['hand'][rows]),
             'face': t(self.p['face'][rows]), 'shape': t(self.p['shape'])[None].expand(n, -1),
             'scale': t(self.p['scale'])[None].expand(n, -1)}
        zero = torch.zeros(n, 3, device=device)
        out = head_pose.mhr_forward(global_trans=zero, global_rot=zero, body_pose_params=p['body_pose'],
                                    hand_pose_params=p['hand'], scale_params=p['scale'], shape_params=p['shape'],
                                    expr_params=p['face'], return_keypoints=True, do_pcblend=False)
        f = t(F_AXES)
        s = float(self.p.get('size_factor', 1.0)) * 100.0
        G, T = t(self.p['g_world'][rows]), t(self.p['trans_cm'][rows])
        conv = lambda X: torch.einsum('nij,nkj->nki', G, X * f * s) + T[:, None]  # noqa: E731
        return conv(out[0]), conv(out[1][:, :70])


def camera_from_floor(cam, summary: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(R, t) with X_cam_m = R X_floor_cm / 100 + t for a cosilab Camera (world = the session reference frame)."""
    Rf = np.asarray(summary['floor_frame']['R'])
    o = np.asarray(summary['floor_frame']['origin_m'])
    return cam.R @ Rf.T, cam.R @ o + cam.t


def project(cam, Rt, X_floor_cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Floor-frame points (..., 3) cm -> pixels (..., 2) of cam (its K, D, size), and camera depth z (m)."""
    R, t = Rt
    Xc = X_floor_cm / 100.0 @ R.T + t
    return fisheye.project_np(Xc, cam.K, cam.D), Xc[..., 2]


def silhouette(px: np.ndarray, z: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Projected vertices -> (H, W) bool silhouette (splat, dilate 3x3, close 5x5); vertices behind the camera or far
    outside the image are left out."""
    W, H = size
    m = np.zeros((H, W), np.uint8)
    xy = np.round(px).astype(np.int64)
    ok = (z > 0.05) & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    m[xy[ok, 1], xy[ok, 0]] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def visible_labels(cam, Rt, people: List[Tuple[int, np.ndarray]]) -> Tuple[np.ndarray, Dict[int, np.ndarray],
                                                                            Dict[int, float]]:
    """people: (pid, vertices floor cm). Returns the label image of the visible projected bodies (0 = none; nearest
    body per pixel, painted far to near by pelvis-ish depth = median camera distance), each body's full silhouette, and
    each body's visible fraction (visible pixels / silhouette pixels)."""
    W, H = cam.size
    sil, depth = {}, {}
    C = np.zeros(3)
    for pid, V in people:
        px, z = project(cam, Rt, V)
        s = silhouette(px, z, (W, H))
        if s.any():
            sil[pid] = s
            R, t = Rt
            depth[pid] = float(np.median(np.linalg.norm(V / 100.0 @ R.T + t - C, axis=-1)))
    labels = np.zeros((H, W), np.int32)
    for pid in sorted(sil, key=lambda p: -depth[p]):
        labels[sil[pid]] = pid
    vis = {pid: float((labels == pid).sum() / max(sil[pid].sum(), 1)) for pid in sil}
    return labels, sil, vis
