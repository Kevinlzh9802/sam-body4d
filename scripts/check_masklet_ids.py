#!/usr/bin/env python3
"""
ID check of Stage 1 masklets in a camera prompted with scripts/project_prompts.py (no annotations): every masklet is
compared with the world bodies of a stage3_mvfit.py run over the other cameras, projected into this camera, and gets
the person ID it overlaps most, or none. Run it between Stage 1 and Stage 2. It doesn't use this camera's own fit.

Per segment, on every --step-th frame: IoU of each masklet (Stage 1 tracking ID) with each projected body's visible
silhouette (nearer bodies drawn over farther ones), intersections and unions summed over the frames; Hungarian
assignment on the summed IoU; matches below --min-iou are dropped (real ID -1, which Stage 2/3 ignore). A masklet whose
best match differs from its prompted ID is relabelled (a swap).

The camera is accepted when the median IoU of its matched masklets is >= --accept-iou (cams 12/14 with their stored
extrinsics are expected to fail: their projections are 80-340 px off). A rejected camera keeps its prompted IDs and
is written as rejected, and the multi-view fit should leave it out.

Writes into the Stage 1 folder: id_mapping.json (checked; the prompted one is kept as id_mapping_prompted.json), the
same mappings in masklets_meta.json, and id_check.json (per segment and masklet: IoU with every candidate,
decision; per camera: match rate, median IoU, accepted).

python scripts/check_masklet_ids.py --mvfit <mvfit dir> --stage1 <Stage 1 dir of camXX_batchBB>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.render_overlay_video import Stage1Frames  # noqa: E402
from utils import cosilab_camera as cc  # noqa: E402
from utils.id_mapping import load_segment_id_mappings  # noqa: E402
from utils.mhr_only import load_mhr_head  # noqa: E402
from utils.world_bodies import WorldBody, camera_from_floor, load_mvfit, visible_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mvfit', required=True)
    ap.add_argument('--stage1', required=True, help='Stage 1 output folder of the camera (camXX_batchBB)')
    ap.add_argument('--extrinsics', nargs='*', default=[], help='cam12=<rvec/tvec json>')
    ap.add_argument('--step', type=int, default=10, help='every n-th frame of a segment is compared')
    ap.add_argument('--min-iou', type=float, default=0.2)
    ap.add_argument('--accept-iou', type=float, default=0.5,
                    help='VOH clip: cams 13/15/02 0.68-0.76, cam04 0.52; cams 12/14 (stored extrinsics 80-340 px off) 0.45/0.48, '
                         'and adding them made every other camera worse')
    ap.add_argument('--image-scale', type=float, default=0.5)
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    ap.add_argument('--dry-run', action='store_true', help='write id_check.json only')
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.stage1))
    cam_no, batch, session = cc.parse_seq(seq)
    summary, paths = load_mvfit(args.mvfit)
    if (int(summary['session']), int(summary['batch'])) != (session, batch):
        raise SystemExit(f"[ERROR] {seq} is session {session} batch {batch}; the fit "
                         f"{summary['session']} / {summary['batch']}")
    cam = cc.load_camera(cam_no, cc.parse_overrides(args.extrinsics).get(cam_no)).scaled(args.image_scale)
    Rt = camera_from_floor(cam, summary)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_pose, _ = load_mhr_head(OmegaConf.load(args.config), device)
    bodies = [WorldBody(pid, p) for pid, p in paths.items()]
    masks = Stage1Frames(args.stage1, 'masks', ('.png', '.bmp'))
    segments = load_segment_id_mappings(args.stage1)
    if not segments:
        raise SystemExit(f'[ERROR] {args.stage1}: no id_mapping.json')

    checked, report, ious_matched = [], {'mvfit': os.path.abspath(args.mvfit), 'camera': cam.name, 'segments': {}}, []
    n_masklets = n_matched = n_relabelled = 0
    for seg in segments:
        c2a = {int(k): int(v) for k, v in seg['consecutive_to_actual'].items()}
        new = dict(c2a)
        frames = [f for f in range(seg['frame_start'], seg['frame_end'] + 1, args.step) if f'{f:08d}' in masks.names]
        if not c2a or not frames:
            checked.append(seg)
            continue
        inter, union = {}, {}
        for f in frames:
            img = Image.open(io.BytesIO(masks.read(f'{f:08d}')))
            lab = np.array(img) if img.mode == 'P' else np.array(img.convert('L'))
            people = []
            for b in bodies:
                r = b.rows([f])
                if r[0] >= 0:
                    V, _ = b.floor_points(head_pose, r, device)
                    people.append((b.pid, V[0].cpu().numpy().astype(np.float64)))
            labels, _, _ = visible_labels(cam, Rt, people) if people else (np.zeros_like(lab, np.int32), {}, {})
            for tid in c2a:
                m = lab == tid
                if not m.any():
                    continue
                for pid in {p for p, _ in people}:
                    s = labels == pid
                    inter[tid, pid] = inter.get((tid, pid), 0) + int((m & s).sum())
                    union[tid, pid] = union.get((tid, pid), 0) + int((m | s).sum())
        tids = sorted({t for t, _ in inter})
        pids = sorted({p for _, p in inter})
        seg_rep = {}
        if tids and pids:
            iou = np.array([[inter.get((t, p), 0) / max(union.get((t, p), 0), 1) for p in pids] for t in tids])
            rows, cols = linear_sum_assignment(-iou)
            match = {tids[r]: (pids[c], iou[r, c]) for r, c in zip(rows, cols)}
            for i, t in enumerate(tids):
                n_masklets += 1
                pid, v = match.get(t, (-1, 0.0))
                decision = 'kept'
                if v < args.min_iou:
                    new[t], decision = -1, 'dropped'
                else:
                    n_matched += 1
                    ious_matched.append(v)
                    if pid != c2a[t]:
                        new[t], decision = pid, f'relabelled from {c2a[t]}'
                        n_relabelled += 1
                top = np.argsort(-iou[i])[:3]
                seg_rep[str(t)] = {'prompted': c2a[t], 'final': new[t], 'decision': decision,
                                   'iou_top': {str(pids[j]): round(float(iou[i, j]), 3) for j in top}}
        report['segments'][seg['segment_key']] = seg_rep
        checked.append(dict(seg, consecutive_to_actual={str(k): v for k, v in new.items()}))
        print(f"[INFO] {seg['segment_key']}: " + ', '.join(f"{t}->{d['final']} ({d['decision']}, "
                                                          f"IoU {max(d['iou_top'].values()):.2f})"
                                                          for t, d in seg_rep.items()), flush=True)
    med = float(np.median(ious_matched)) if ious_matched else 0.0
    accepted = med >= args.accept_iou and n_matched > 0
    report.update({'masklets': n_masklets, 'matched': n_matched, 'relabelled': n_relabelled,
                   'median_iou_matched': round(med, 3), 'accepted': accepted, 'min_iou': args.min_iou,
                   'accept_iou': args.accept_iou})
    print(f'[INFO] {cam.name}: {n_matched}/{n_masklets} masklets matched, {n_relabelled} relabelled, median IoU '
          f'{med:.3f} -> {"ACCEPTED" if accepted else "REJECTED (leave this camera out of the fit)"}')
    with open(os.path.join(args.stage1, 'id_check.json'), 'w') as fh:
        json.dump(report, fh, indent=1)
    if args.dry_run or not accepted:
        return
    idp = os.path.join(args.stage1, 'id_mapping.json')
    if not os.path.exists(os.path.join(args.stage1, 'id_mapping_prompted.json')):
        shutil.copy(idp, os.path.join(args.stage1, 'id_mapping_prompted.json'))
    with open(idp) as fh:
        data = json.load(fh)
    data['segments'] = checked
    data['id_check'] = 'scripts/check_masklet_ids.py'
    with open(idp, 'w') as fh:
        json.dump(data, fh, indent=1)
    mp = os.path.join(args.stage1, 'masklets_meta.json')
    if os.path.exists(mp):
        with open(mp) as fh:
            meta = json.load(fh)
        meta['segment_id_mappings'] = checked
        with open(mp, 'w') as fh:
            json.dump(meta, fh, indent=1)


if __name__ == '__main__':
    main()
