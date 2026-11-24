#!/usr/bin/env python3
"""
Stage 3, multi-view: one world-frame MHR body per person, fitted to every camera of a cosilab session and batch at
once (smoothing/mv_fit.py has the model and the losses). Replaces the scale/floor-only stage3_multiview.py.

Input: Stage 2 outputs made with --fisheye (raw_mhr.pt with real-camera params, fisheye 2D keypoints kp2d / kp_conf),
one per camera of the same session and batch, and their Stage 1 folders (masks). Real person IDs are shared between
cameras; negative IDs (people outside the study) are ignored. People seen by one camera only are fitted too (their
scale comes from SAM-3D-Body's depth and the floor term).

Steps:
  1. load every camera: per person and camera the params, kp2d, kp_conf, and a 64 px soft-mask ROI per frame
  2. sync check (diagnostic): median ray gap of torso points between camera pairs for frame offsets -30..30
  3. start body per person (best camera per frame, triangulated translation, temporal smoothing)
  4. floor plane from everyone's start bodies (lowest foot per person-frame, robust plane fit)
  5. fit per person (Adam over all its frames; keypoints + silhouettes in every camera, priors, floor)
  6. floor again from the fitted bodies; export in a floor frame: cm, floor z = 0, z up, origin below the session's
     reference camera (cam06 / cam01), x along the reference camera's x

Output (--out-dir, default <SB4D_OUTPUT_ROOT>/mvfit/session<S>_batch<BB>[_f<start>-<end>]):
  world/<pid>.npz            the body once per person: vertices (T, V, 3) float32 cm only with --save-vertices
                             (~4 GB per person and batch; utils/world_bodies.py re-poses them from the params), mhr70
                             keypoints (T, 70, 3) cm, faces, frame_names (batch frames, '%08d'), params (g_world
                             (T, 3, 3) in the floor frame, trans_cm, body_pose, hand, face, shape, scale), cameras
  cameras/<camXX_batchBB>/   per camera: stage3_run_options.txt (fisheye camera, Stage 1 folder, mesh_dir = world/),
                             extrinsics/extrinsic_cam_<MM>.json (floor frame, cm, with K and D) for every camera,
                             reproj_overlay.mp4 (scripts/render_overlay_video.py; --no-video to skip)
  summary.json               sync, floor, per person: start/final metrics per camera, loss history, timing

Usage (one camera's Stage 2 folder or raw_mhr.pt per argument):
  python stage3_mvfit.py <raw_dir>/cam01_batch01 <raw_dir>/cam03_batch01 ... [--start 6000 --num-frames 600]
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_DIR, os.path.join(REPO_DIR, 'models', 'sam_3d_body')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import load_raw_mhr  # noqa: E402
from scripts.render_overlay_video import Stage1Frames, render_overlay_video  # noqa: E402
from smoothing import mv_fit  # noqa: E402
from smoothing.mv_fit import BODY_KPS, HIPS, SHOULDERS, CamTrack, FitConfig  # noqa: E402
from utils import cosilab_camera as cc  # noqa: E402
from utils.mhr_only import load_mhr_head  # noqa: E402

ROI = 64             # soft-mask ROI size (px)
ROI_PAD = 0.25       # ROI side = (1 + 2 ROI_PAD) x the mask box side
MASK_PTS = 64        # mask pixels sampled per person-frame for the fill term
SYNC_OFFSETS = range(-30, 31)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def raw_path(p: str) -> str:
    return os.path.join(p, 'raw_mhr.pt') if os.path.isdir(p) else p


def stage1_dir_for(meta: dict, seq: str, stage1_root: Optional[str]) -> str:
    cands = ([os.path.join(stage1_root, seq)] if stage1_root else []) + [meta.get('input_dir', '')]
    for c in cands:
        if c and (os.path.isdir(os.path.join(c, 'masks')) or os.path.isfile(os.path.join(c, 'masks.zip'))):
            return c
    raise SystemExit(f'[ERROR] {seq}: no Stage 1 masks in {cands} (use --stage1-root)')


def load_camera_run(path: str, args, frame_range) -> Dict[str, object]:
    payload = load_raw_mhr(raw_path(path))
    meta = payload.get('meta', {})
    if meta.get('camera_model') != 'fisheye_virtual':
        raise SystemExit(f'[ERROR] {path}: not a Stage 2 --fisheye output (camera_model '
                         f'{meta.get("camera_model", "pinhole")!r}); run stage2_sam3body.py --fisheye')
    seq = os.path.basename(os.path.dirname(raw_path(path)))
    cam_no, batch, session = cc.parse_seq(seq)
    overrides = cc.parse_overrides(args.extrinsics)
    cam = cc.load_camera(cam_no, overrides.get(cam_no))
    cam = cam.scaled(meta['image_size'][0] / cam.size[0])
    if np.abs(cam.K - np.array(meta['fisheye_K'])).max() > 1e-6 or np.abs(cam.D - np.array(meta['fisheye_D'])).max() > 1e-9:
        raise SystemExit(f'[ERROR] {seq}: the Stage 2 intrinsics differ from the current calibration')
    weights = {int(k.replace('cam', '')): float(v) for k, v in (w.split('=') for w in args.view_weight)}
    excluded = {(int(c.split('_')[0].replace('cam', '')), int(p)) for c, p in (e.split(':') for e in args.exclude)}
    holdout = cam.name in args.holdout

    per: Dict[int, Dict[str, list]] = {}
    lo, hi = frame_range
    for fr in payload['frames']:
        f = int(fr['frame'])
        if not lo <= f < hi:
            continue
        seen = set()
        for p in fr['people']:
            pid = int(p['obj_id'])
            if pid < 0 or pid in seen or p.get('kp2d') is None or (cam_no, pid) in excluded:
                continue
            seen.add(pid)
            d = per.setdefault(pid, {k: [] for k in ('frame', 'tid', 'cam_t', 'kp2d', 'kp_conf') + mv_fit.MHR_KEYS
                                     + ('global_rot',)})
            d['frame'].append(f)
            d['tid'].append(int(p['tracking_id']))
            d['cam_t'].append(np.asarray(p['pred_cam_t'], np.float32))
            d['kp2d'].append(np.asarray(p['kp2d'], np.float32))
            d['kp_conf'].append(np.asarray(p['kp_conf'], np.float32))
            for k in mv_fit.MHR_KEYS + ('global_rot',):
                d[k].append(np.asarray(p[k], np.float32))
    tracks = {}
    for pid, d in per.items():
        if len(d['frame']) < args.min_frames:
            continue
        order = np.argsort(d['frame'])
        tracks[pid] = CamTrack(
            cam=cam, frames=np.asarray(d['frame'])[order], tid=np.asarray(d['tid'])[order],
            params={k: np.stack(d[k])[order] for k in mv_fit.MHR_KEYS + ('global_rot',)},
            cam_t=np.stack(d['cam_t'])[order], kp2d=np.stack(d['kp2d'])[order], kp_conf=np.stack(d['kp_conf'])[order],
            euler=meta.get('mhr_euler_convention', 'xyz'), weight=weights.get(cam_no, 1.0), holdout=holdout)
    return {'seq': seq, 'cam': cam, 'batch': batch, 'session': session, 'meta': meta, 'tracks': tracks,
            'stage1': stage1_dir_for(meta, seq, args.stage1_root)}


def load_mask_rois(run: Dict[str, object]) -> None:
    """One pass over the camera's Stage 1 masks: per track and frame a ROI x ROI soft mask around the person's mask,
    its box, the mask centroid / extent, and the box size (px) used as the keypoint residual unit."""
    tracks: Dict[int, CamTrack] = run['tracks']
    masks = Stage1Frames(run['stage1'], 'masks', ('.png', '.bmp'))
    rng = np.random.default_rng(0)
    by_frame: Dict[int, list] = {}
    for pid, tr in tracks.items():
        n = len(tr.frames)
        tr.roi = np.zeros((n, ROI, ROI), np.uint8)
        tr.roi_other = np.zeros((n, ROI, ROI), np.uint8)
        tr.roi_box = np.zeros((n, 4), np.float32)
        tr.m_center = np.zeros((n, 2), np.float32)
        tr.m_ext = np.zeros((n, 2), np.float32)
        tr.mask_pts = np.zeros((n, MASK_PTS, 2), np.float32)
        tr.has_mask = np.zeros(n, bool)
        for i, f in enumerate(tr.frames):
            by_frame.setdefault(int(f), []).append((tr, i))
    for f, items in sorted(by_frame.items()):
        name = f'{f:08d}'
        if name not in masks.names:
            continue
        img = Image.open(io.BytesIO(masks.read(name)))
        labels = np.array(img) if img.mode == 'P' else np.array(img.convert('L'))
        H, W = labels.shape
        for tr, i in items:
            m = labels == tr.tid[i]
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            side = max(x1 - x0, y1 - y0) * (1 + 2 * ROI_PAD)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], np.float32)
            # The ROI covers the image region [box[0] - 0.5, box[0] - 0.5 + side) (pixel centres at integers):
            # ROI pixel centre u <- image x = box[0] - 0.5 + (u + 0.5) s, i.e. u = (x + 0.5 - box[0]) / s - 0.5.
            # Blurred by ~s/2 first, so the bilinear resample approximates an area average.
            s = side / ROI
            A = np.array([[1 / s, 0, (0.5 - box[0]) / s - 0.5], [0, 1 / s, (0.5 - box[1]) / s - 0.5]], np.float32)
            for dst, region in ((tr.roi, m), (tr.roi_other, (labels > 0) & ~m)):
                soft = cv2.warpAffine(cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(s / 2, 0.5)), A,
                                      (ROI, ROI), flags=cv2.INTER_LINEAR, borderValue=0)
                dst[i] = np.clip(np.round(soft * 255), 0, 255).astype(np.uint8)
            tr.roi_box[i] = box
            tr.m_center[i] = (xs.mean(), ys.mean())
            tr.m_ext[i] = (x1 - x0, y1 - y0)
            pick = rng.choice(len(xs), MASK_PTS, replace=len(xs) < MASK_PTS)
            tr.mask_pts[i] = np.stack([xs[pick], ys[pick]], 1)
            tr.has_mask[i] = True
    for tr in tracks.values():
        ok = tr.kp_conf[:, BODY_KPS] >= mv_fit.TRI_CONF
        size = np.zeros(len(tr.frames), np.float32)
        for i in range(len(size)):
            if tr.has_mask[i]:
                size[i] = max(tr.m_ext[i])
            elif ok[i].sum() >= 2:
                p = tr.kp2d[i, BODY_KPS][ok[i]]
                size[i] = (p.max(0) - p.min(0)).max()
        good = size > 0
        tr.box_size = np.where(good, size, np.median(size[good]) if good.any() else 100.0).astype(np.float32)
        # no mask (e.g. a Stage 1 frame without masks): no keypoint confidence either
        tr.kp_conf = tr.kp_conf * tr.has_mask[:, None]


# ---------------------------------------------------------------------------
# sync check (from the old stage3_multiview.py, on torso rays of the 2D keypoints)
# ---------------------------------------------------------------------------

def sync_check(runs_: List[Dict[str, object]]) -> Dict[str, object]:
    out = {}
    for a, b in itertools.combinations(runs_, 2):
        shared = sorted(set(a['tracks']) & set(b['tracks']))
        if not shared:
            continue
        w = b['cam'].centre - a['cam'].centre
        med = {}
        for off in SYNC_OFFSETS:
            gaps = []
            for pid in shared:
                ta, tb = a['tracks'][pid], b['tracks'][pid]
                _, ia, ib = np.intersect1d(ta.frames, tb.frames + off, return_indices=True)
                if not len(ia):
                    continue
                for pair in (HIPS, SHOULDERS):
                    ok = (ta.kp_conf[ia][:, pair] >= mv_fit.TRI_CONF).all(1) & (tb.kp_conf[ib][:, pair] >= mv_fit.TRI_CONF).all(1)
                    if not ok.any():
                        continue
                    ua = a['cam'].rays(ta.kp2d[ia][ok][:, pair].mean(1))
                    ub = b['cam'].rays(tb.kp2d[ib][ok][:, pair].mean(1))
                    n = np.cross(ua, ub)
                    nn = np.linalg.norm(n, axis=-1)
                    g = nn > 1e-6
                    gaps.append(np.abs(n[g] @ w) / nn[g])
            if gaps:
                med[off] = float(np.median(np.concatenate(gaps)))
        if med:
            best = min(med, key=med.get)
            out[f"{a['cam'].name}-{b['cam'].name}"] = {
                'best_offset': best, 'median_gap_cm_at_0': round(med.get(0, float('nan')) * 100, 2),
                'median_gap_cm_at_best': round(med[best] * 100, 2), 'people': len(shared)}
    return out


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def floor_frame(floor: Dict[str, object]):
    """Rotation Rf (rows x, y, up) and origin o (m, world): X_floor = Rf (X_w - o). Origin below the reference camera
    (the world origin), x along the world x projected onto the floor."""
    up = np.asarray(floor['up'], float)
    c = np.asarray(floor['point'], float)
    o = (c @ up) * up
    x = np.array([1.0, 0, 0]) - up[0] * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up]), o


def camera_json(cam, Rf, o, ws: float, source: str) -> dict:
    R = cam.R @ Rf.T
    t = (cam.R @ o + cam.t) * ws
    return {'extrinsic_type': 'world_to_camera', 'rotation': R.tolist(), 'translation': t.tolist(),
            'units': 'cm' if ws == 100 else f'1/{ws} m', 'camera_center_world_m': (Rf @ (cam.centre - o)).tolist(),
            'camera_model': 'fisheye', 'K': cam.K.tolist(), 'D': cam.D.tolist(), 'image_size': list(cam.size),
            'source': source}


@torch.no_grad()
def export_person(path: str, head_pose, body: mv_fit.Body, Rf, o, ws: float, faces: np.ndarray, cams: List[str],
                  save_vertices: bool) -> Dict[str, float]:
    T = len(body.frames)
    Rf_t = torch.as_tensor(Rf, dtype=torch.float32, device=body.trans.device)
    o_t = torch.as_tensor(o, dtype=torch.float32, device=body.trans.device)
    verts, kps = [], []
    for a in range(0, T, 256):
        b = min(T, a + 256)
        v, k = mv_fit.mhr_full(head_pose, body.params(a, b))
        conv = lambda X: (((body.world(X, a, b) - o_t) @ Rf_t.T) * ws).cpu().numpy().astype(np.float32)  # noqa
        kps.append(conv(k))
        if save_vertices:
            verts.append(conv(v))
    kps = np.concatenate(kps)
    G = mv_fit.rot6d_to_rotmat(body.g6d).cpu().numpy()
    data = dict(keypoints=kps, faces=faces, frame_names=np.array([f'{f:08d}' for f in body.frames]),
                g_world=np.einsum('ij,njk->nik', Rf, G).astype(np.float32),
                trans_cm=(((body.trans.cpu().numpy() - o) @ Rf.T) * ws).astype(np.float32),
                body_pose=body.pose.cpu().numpy(), hand=body.hand.cpu().numpy(), face=body.face.cpu().numpy(),
                shape=body.shape.cpu().numpy()[0], scale=body.scale.cpu().numpy()[0],
                size_factor=float(torch.exp(body.log_s)), cameras=np.array(cams))
    if save_vertices:
        data['vertices'] = np.concatenate(verts)
    np.savez_compressed(path, **data)
    z = kps[..., 2]
    return {'height_cm_median': float(np.median(z.max(1) - z.min(1))),
            'lowest_foot_cm_median': float(np.median(kps[:, list(mv_fit.FOOT_IDXS), 2].min(1)))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='Stage 2 --fisheye outputs (<raw_dir>/<camXX_batchBB>[/raw_mhr.pt])')
    ap.add_argument('--stage1-root', default=None, help='folder of <camXX_batchBB>/ Stage 1 outputs (default: the '
                                                        'Stage 2 meta input_dir)')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--start', type=int, default=0, help='first batch frame')
    ap.add_argument('--num-frames', type=int, default=None)
    ap.add_argument('--extrinsics', nargs='*', default=[], help="replace a camera's extrinsics: cam12=<rvec/tvec json>")
    ap.add_argument('--view-weight', nargs='*', default=[], help='keypoint/silhouette weight per camera: cam12=0.5')
    ap.add_argument('--holdout', nargs='*', default=[], help='cameras evaluated but not fitted (leave-one-out)')
    ap.add_argument('--exclude', nargs='*', default=[], help='person/camera pairs left out: cam12_batch01:14')
    ap.add_argument('--people', type=int, nargs='*', default=None, help='only these person IDs')
    ap.add_argument('--part', default=None, metavar='K/N',
                    help='fit only every N-th person starting at K (sorted IDs), e.g. one SLURM array task each; the '
                         'start bodies and the start floor still use everyone, so all parts get the same floor. '
                         'Writes <out-dir>/parts/<pid>.pt and parts/summary_<K>.json; then run once with --merge')
    ap.add_argument('--merge', action='store_true', help='no fitting: collect parts/, refit the floor, export')
    ap.add_argument('--min-frames', type=int, default=10, help='shorter person/camera tracks are dropped')
    ap.add_argument('--config', default=os.path.join(REPO_DIR, 'configs', 'body4d.yaml'))
    ap.add_argument('--world-scale', type=float, default=100.0, help='world units per metre (100 = cm)')
    ap.add_argument('--save-vertices', action='store_true',
                    help='also store the vertices in world/<pid>.npz (large; by default they are re-posed when needed)')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-floor-term', action='store_true', help='w_floor = 0')
    ap.add_argument('--no-smooth-post', action='store_true', help='no Hampel + Gaussian of the fitted bodies')
    ap.add_argument('--no-smooth-start', action='store_true', help='no smoothing of the start bodies')
    for k, v in vars(FitConfig()).items():
        if isinstance(v, bool):
            continue
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    args = ap.parse_args()
    cfg = FitConfig(**{k: getattr(args, k) for k in vars(FitConfig()) if hasattr(args, k)})
    if args.no_floor_term:
        cfg.w_floor = 0.0
    cfg.smooth_post = not args.no_smooth_post
    cfg.smooth_start = not args.no_smooth_start
    t_start = time.time()

    hi = args.start + args.num_frames if args.num_frames else 10 ** 9
    runs_ = [load_camera_run(p, args, (args.start, hi)) for p in args.inputs]
    sessions = {(r['session'], r['batch']) for r in runs_}
    if len(sessions) != 1:
        raise SystemExit(f'[ERROR] inputs from several (session, batch): {sorted(sessions)}')
    names = [r['cam'].name for r in runs_]
    if len(set(names)) != len(names):
        raise SystemExit(f'[ERROR] several inputs of one camera: {names}')
    session, batch = sessions.pop()
    unknown = set(args.holdout) - set(names)
    if unknown:
        raise SystemExit(f'[ERROR] --holdout cameras not in the inputs: {sorted(unknown)}')
    for r in runs_:
        if args.merge:  # no fitting: only the people and their cameras are needed
            continue
        load_mask_rois(r)
        log(f"[INFO] {r['seq']}: {len(r['tracks'])} people, Stage 1 {r['stage1']}"
            f"{' (held out)' if r['cam'].name in args.holdout else ''}")
    clip = f'_f{args.start}-{hi}' if args.num_frames else ''
    out_dir = args.out_dir or os.path.join(cc.OUTPUT_ROOT, 'mvfit', f'session{session}_batch{batch:02d}{clip}')
    os.makedirs(os.path.join(out_dir, 'world'), exist_ok=True)
    parts_dir = os.path.join(out_dir, 'parts')
    part = None
    if args.part:
        k, n = map(int, args.part.split('/'))
        if not 0 <= k < n:
            raise SystemExit(f'[ERROR] --part {args.part}: need 0 <= K < N')
        part = (k, n)
        os.makedirs(parts_dir, exist_ok=True)

    sync = sync_check(runs_) if not args.merge else {}
    for pair, s in sync.items():
        flag = '  <-- clearly smaller away from offset 0' if s['median_gap_cm_at_best'] < 0.8 * s['median_gap_cm_at_0'] else ''
        log(f"[INFO] sync {pair}: ray gap {s['median_gap_cm_at_0']} cm at offset 0, best "
            f"{s['median_gap_cm_at_best']} cm at {s['best_offset']}{flag}")

    pids = sorted({pid for r in runs_ for pid in r['tracks']})
    if args.people is not None:
        pids = [p for p in pids if p in set(args.people)]
    tracks = {pid: [r['tracks'][pid] for r in runs_ if pid in r['tracks']] for pid in pids}
    pids = [p for p in pids if any(not t.holdout for t in tracks[p])]
    log(f'[INFO] session {session}, batch {batch}: {len(pids)} people; seen by >= 2 fitted cameras: '
        f'{sum(sum(not t.holdout for t in tracks[p]) >= 2 for p in pids)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_pose, faces = load_mhr_head(OmegaConf.load(args.config), device)
    if args.merge:
        merge_and_export(args, out_dir, parts_dir, pids, tracks, runs_, head_pose, faces, device, t_start)
        return

    starts = {}
    for pid in pids:
        starts[pid] = mv_fit.start_body(head_pose, tracks[pid], device, cfg)
        st = starts[pid]
        log(f"[INFO] person {pid}: {len(st.frames)} frames, {int((st.n_views >= 2).sum())} with >= 2 usable cameras, "
            f"start shift {st.stats.get('start_shift_mean_m', 0):.2f} m "
            f"({st.stats.get('triangulated_frames', 0)} triangulated frames)")
    down = np.array([0.0, 0.0, 1.0])  # the reference camera (world frame) looks down along +z
    floor0 = mv_fit.fit_floor(head_pose, list(starts.values()), down)
    log(f"[INFO] floor from the start bodies: spread {floor0['spread_m']:.3f} m, kept {floor0['kept']:.2f}, "
        f"up {np.round(floor0['up'], 3).tolist()}, reference camera {float(-floor0['point'] @ floor0['up']):.3f} m above it")

    summary = {'session': session, 'batch': batch, 'inputs': [os.path.abspath(raw_path(p)) for p in args.inputs],
               'frame_range': [args.start, hi], 'config': vars(cfg), 'args': vars(args), 'sync': sync,
               'floor_start': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in floor0.items()},
               'people': {}}
    fitted = {}
    fit_pids = pids if part is None else pids[part[0]::part[1]]
    if part is not None:
        log(f'[INFO] part {part[0]}/{part[1]}: fitting {len(fit_pids)} of {len(pids)} people: {fit_pids}')
    for pid in fit_pids:
        t0 = time.time()
        log(f'[INFO] person {pid}: fitting ({len(starts[pid].frames)} frames, '
            f'cameras {[t.cam.name + (" (held out)" if t.holdout else "") for t in tracks[pid]]})')
        m_start = mv_fit.evaluate(head_pose, tracks[pid], starts[pid], device)
        fitted[pid] = mv_fit.fit_person(head_pose, tracks[pid], starts[pid], cfg, device,
                                        floor=floor0 if cfg.w_floor > 0 else None, log=log)
        m_final = mv_fit.evaluate(head_pose, tracks[pid], fitted[pid], device)
        summary['people'][str(pid)] = {'start': m_start, 'final': m_final, 'stats': fitted[pid].stats,
                                       'seconds': round(time.time() - t0, 1)}
        for cam in m_final:
            if cam == '_body':
                continue
            s, f = m_start[cam], m_final[cam]
            log(f"    {cam}: kp {s['kp_px_median']:.1f} -> {f['kp_px_median']:.1f} px (p90 {s['kp_px_p90']:.1f} -> "
                f"{f['kp_px_p90']:.1f}), IoU {s.get('iou', float('nan')):.3f} -> {f.get('iou', float('nan')):.3f}, "
                f"verts in mask {s.get('verts_in_mask', float('nan')):.3f} -> {f.get('verts_in_mask', float('nan')):.3f}")
        log(f"    accel {m_start['_body']['accel_mm_per_frame2']:.2f} -> {m_final['_body']['accel_mm_per_frame2']:.2f} "
            f"mm/frame^2; {time.time() - t0:.0f} s")
        if part is not None:
            torch.save(body_state(fitted[pid]), os.path.join(parts_dir, f'{pid}.pt'))
    if part is not None:
        with open(os.path.join(parts_dir, f'summary_{part[0]}.json'), 'w') as fh:
            json.dump(summary, fh, indent=1, default=float)
        log(f'[INFO] part {part[0]}/{part[1]} done in {time.time() - t_start:.0f} s -> {parts_dir}')
        return
    finish(args, out_dir, summary, fitted, pids, tracks, runs_, head_pose, faces, t_start)


def body_state(b: mv_fit.Body) -> dict:
    return {'frames': b.frames, 'g6d': b.g6d.cpu(), 'trans': b.trans.cpu(), 'pose': b.pose.cpu(), 'hand': b.hand.cpu(),
            'face': b.face.cpu(), 'shape': b.shape.cpu(), 'scale': b.scale.cpu(), 'log_s': b.log_s.cpu(),
            'best_cam': b.best_cam, 'n_views': b.n_views, 'stats': b.stats}


def body_from_state(d: dict, device) -> mv_fit.Body:
    t = {k: d[k].to(device) for k in ('g6d', 'trans', 'pose', 'hand', 'face', 'shape', 'scale', 'log_s')}
    return mv_fit.Body(frames=d['frames'], best_cam=d['best_cam'], n_views=d['n_views'], stats=d['stats'], **t)


def merge_and_export(args, out_dir, parts_dir, pids, tracks, runs_, head_pose, faces, device, t_start) -> None:
    missing = [p for p in pids if not os.path.exists(os.path.join(parts_dir, f'{p}.pt'))]
    if missing:
        raise SystemExit(f'[ERROR] --merge: no fitted part for people {missing} in {parts_dir}')
    fitted = {p: body_from_state(torch.load(os.path.join(parts_dir, f'{p}.pt'), weights_only=False), device)
              for p in pids}
    summary = None
    for fn in sorted(os.listdir(parts_dir)):
        if fn.startswith('summary_') and fn.endswith('.json'):
            with open(os.path.join(parts_dir, fn)) as fh:
                part_summary = json.load(fh)
            if summary is None:
                summary = part_summary
            else:
                summary['people'].update(part_summary['people'])
    summary['args'] = dict(summary['args'], merged_parts=True)
    log(f'[INFO] merged {len(fitted)} people from {parts_dir}')
    finish(args, out_dir, summary, fitted, pids, tracks, runs_, head_pose, faces, t_start)


def finish(args, out_dir, summary, fitted, pids, tracks, runs_, head_pose, faces, t_start) -> None:
    session, batch = summary['session'], summary['batch']
    hi = summary['frame_range'][1]
    down = np.array([0.0, 0.0, 1.0])

    floor = mv_fit.fit_floor(head_pose, list(fitted.values()), down)
    Rf, o = floor_frame(floor)
    ws = float(args.world_scale)
    summary['floor'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in floor.items()}
    summary['floor_frame'] = {'R': Rf.tolist(), 'origin_m': o.tolist(), 'world_scale': ws,
                              'note': 'X_floor = world_scale * R (X_world - origin_m); world = session reference camera'}
    summary['floor']['reference_camera_height_m'] = float(-floor['point'] @ floor['up'])
    log(f"[INFO] floor from the fitted bodies: spread {floor['spread_m']:.3f} m, reference camera "
        f"{summary['floor']['reference_camera_height_m']:.3f} m above it")
    for pid in pids:
        summary['people'][str(pid)]['export'] = export_person(
            os.path.join(out_dir, 'world', f'{pid}.npz'), head_pose, fitted[pid], Rf, o, ws, faces,
            [t.cam.name for t in tracks[pid]], args.save_vertices)

    source = f'stage3_mvfit.py: idiap_params calibration, floor fitted to the people (session {session}, batch {batch})'
    extr = {r['cam'].name: camera_json(r['cam'], Rf, o, ws, source) for r in runs_}
    for r in runs_:
        cdir = os.path.join(out_dir, 'cameras', r['seq'])
        os.makedirs(os.path.join(cdir, 'extrinsics'), exist_ok=True)
        for name, d in extr.items():
            with open(os.path.join(cdir, 'extrinsics', f"extrinsic_{name.replace('cam', 'cam_')}.json"), 'w') as fh:
                json.dump(d, fh, indent=2)
        opts = {'mode': 'mvfit', 'camera': r['cam'].name, 'input_dir': r['stage1'], 'raw': os.path.abspath(raw_path(args.inputs[runs_.index(r)])),
                'mesh_dir': os.path.abspath(os.path.join(out_dir, 'world')),
                'extrinsics_json': os.path.abspath(os.path.join(cdir, 'extrinsics',
                                                                f"extrinsic_{r['cam'].name.replace('cam', 'cam_')}.json")),
                'export_camera_space': False, 'world_scale': ws, 'camera_model': 'fisheye',
                'fisheye_K': json.dumps(r['cam'].K.tolist()), 'fisheye_D': json.dumps(r['cam'].D.tolist()),
                'camera_scale': 1.0, 'frame_range': f'{args.start} {hi}', 'config': os.path.abspath(args.config)}
        with open(os.path.join(cdir, 'stage3_run_options.txt'), 'w') as fh:
            for k, v in opts.items():
                fh.write(f'{k}: {v}\n')
        if not args.no_video:
            try:
                render_overlay_video(cdir)
            except Exception as e:  # the fit is saved; the video is a convenience
                log(f"[WARN] {r['seq']}: overlay video failed: {e}")
    summary['seconds'] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    log(f'[INFO] Done in {summary["seconds"]:.0f} s -> {out_dir}')


if __name__ == '__main__':
    main()
