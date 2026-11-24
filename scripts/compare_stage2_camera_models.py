#!/usr/bin/env python3
"""
Stage 2 pinhole vs fisheye virtual-camera outputs on the same Stage 1 frames: how well each mesh projects onto the
person's Stage 1 mask with the camera model it was made for. Numbers only, no images written.

Per (frame, person) in both runs: MHR forward of the raw params, camera-space keypoints/vertices = local + pred_cam_t;
pinhole run projected with its K (as the old Stage 3 does), fisheye run with the fisheye K, D (utils/cosilab_camera.py).
  kp_on_mask   fraction of the 70 mhr70 keypoints whose pixel is on the person's own mask
  iou          IoU of the projected vertex silhouette (vertices splatted, closed with a small kernel) and the mask
  depth        pelvis depth (m) in the camera frame
Summary by image region (centre / middle / edge, by distance of the mask centroid from the principal point).

python scripts/compare_stage2_camera_models.py --stage1 <Stage 1 dir> --pinhole <dir>/raw_mhr.pt \
    --fisheye <dir>/raw_mhr.pt --intrinsic-path parameters-camera-01.json [--config configs/body4d.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr  # noqa: E402
from utils import cosilab_camera, fisheye  # noqa: E402
from utils.camera_utils import adjust_K, read_camera_intrinsics_new  # noqa: E402
from utils.id_mapping import load_segment_id_mappings, to_actual_pid  # noqa: E402
from utils.model_factory import build_sam3d_body_from_config  # noqa: E402

KEYS = ('global_rot', 'body_pose', 'hand', 'scale', 'shape', 'face')
PELVIS = (9, 10)  # mhr70 hips


def forward(head_pose, people, device):
    p = {k: torch.as_tensor(np.stack([x[k] for x in people]), dtype=torch.float32, device=device) for k in KEYS}
    with torch.no_grad():
        out = head_pose.mhr_forward(global_trans=p['global_rot'] * 0, global_rot=p['global_rot'],
                                    body_pose_params=p['body_pose'], hand_pose_params=p['hand'],
                                    scale_params=p['scale'], shape_params=p['shape'], expr_params=p['face'],
                                    return_keypoints=True)
    v, k = out[0].float(), out[1][:, :70].float()
    v[..., 1:3] *= -1
    k[..., 1:3] *= -1
    t = torch.as_tensor(np.stack([x['pred_cam_t'] for x in people]), dtype=torch.float32, device=device)
    return v + t[:, None], k + t[:, None]


def silhouette(px, shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    xy = np.round(px).astype(int)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    m[xy[ok, 1], xy[ok, 0]] = 1
    return cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage1', required=True)
    ap.add_argument('--pinhole', required=True)
    ap.add_argument('--fisheye', required=True)
    ap.add_argument('--intrinsic-path', required=True, help='the pinhole run\'s intrinsics JSON')
    ap.add_argument('--camera-scale', type=float, default=0.5)
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    ap.add_argument('--out', default=None, help='JSON with the per-person rows')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_pose = build_sam3d_body_from_config(OmegaConf.load(args.config), device).model.head_pose
    K_pin = adjust_K(read_camera_intrinsics_new(args.intrinsic_path)[0], args.camera_scale).astype(np.float64)
    runs = {'pinhole': load_raw_mhr(args.pinhole), 'fisheye': load_raw_mhr(args.fisheye)}
    meta = runs['fisheye']['meta']
    if meta.get('camera_model') != 'fisheye_virtual':
        raise SystemExit(f'{args.fisheye}: not a --fisheye Stage 2 output')
    K_f, D_f = np.array(meta['fisheye_K']), np.array(meta['fisheye_D'])
    seg_maps = load_segment_id_mappings(args.stage1)
    mask_dir = os.path.join(args.stage1, 'masks')

    rows = []
    for name, payload in runs.items():
        for fr in payload['frames']:
            people = [p for p in fr['people'] if p.get('global_rot') is not None]
            if not people:
                continue
            labels = np.array(Image.open(os.path.join(mask_dir, f"{fr['frame']}.png")).convert('P'))
            H, W = labels.shape
            verts, kps = forward(head_pose, people, device)
            verts, kps = verts.cpu().numpy().astype(np.float64), kps.cpu().numpy().astype(np.float64)
            for j, p in enumerate(people):
                if name == 'pinhole':
                    proj = lambda X: (X[:, :2] / X[:, 2:]) * K_pin[[0, 1], [0, 1]] + K_pin[:2, 2]  # noqa: E731
                else:
                    proj = lambda X: fisheye.project_np(X, K_f, D_f)  # noqa: E731
                own = labels == p['tracking_id']
                if not own.any():
                    continue
                kp = proj(kps[j])
                xy = np.round(kp).astype(int)
                ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
                on = np.zeros(len(kp), bool)
                on[ok] = own[xy[ok, 1], xy[ok, 0]]
                sil = silhouette(proj(verts[j]), (H, W)) > 0
                iou = (sil & own).sum() / max((sil | own).sum(), 1)
                ys, xs = np.nonzero(own)
                r = np.hypot((xs.mean() - K_f[0, 2]) / W, (ys.mean() - K_f[1, 2]) / W)
                rows.append({'run': name, 'frame': fr['frame'], 'pid': int(p['obj_id']), 'kp_on_mask': float(on.mean()),
                             'iou': float(iou), 'depth': float(kps[j][list(PELVIS), 2].mean()), 'radius': float(r)})

    def summ(sel):
        return (f"n {len(sel):4d}  kp_on_mask {np.mean([r['kp_on_mask'] for r in sel]):.3f}  "
                f"iou {np.mean([r['iou'] for r in sel]):.3f}" if sel else 'n    0')
    bands = [('centre', 0, 0.15), ('middle', 0.15, 0.3), ('edge', 0.3, 9)]
    for name in runs:
        print(f'{name}:')
        for b, lo, hi in [('all', 0, 9)] + bands:
            print(f'  {b:7s} ' + summ([r for r in rows if r['run'] == name and lo <= r['radius'] < hi]))
    pin = {(r['frame'], r['pid']): r for r in rows if r['run'] == 'pinhole'}
    dd = [r['depth'] - pin[(r['frame'], r['pid'])]['depth'] for r in rows
          if r['run'] == 'fisheye' and (r['frame'], r['pid']) in pin]
    if dd:
        print(f'pelvis depth fisheye - pinhole: median {np.median(dd):+.3f} m, p10 {np.percentile(dd, 10):+.3f}, '
              f'p90 {np.percentile(dd, 90):+.3f} ({len(dd)} person-frames)')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()
