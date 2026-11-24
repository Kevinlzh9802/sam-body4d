"""
Joint multi-view fit of one world-frame MHR body per person (stage3_mvfit.py), modelled on Video-OnlineHMR's
scripts/run_cosilab_multiview.py (init_person / refine_person), with MHR, fisheye cameras, SAM-3D-Body 2D keypoints and
SAM-3 masks.

Body in the world frame (metres, the session's reference camera frame): X = G (s F M(pose, shape, scale)) + T, with M
the MHR output for global_rot = 0, F = diag(1, -1, -1) (MHR -> camera axes, as Stage 2/3), G a per-frame rotation
(6D), T a per-frame translation and s = exp(log_s) one size factor per person. s is what a single camera can't see
(size vs distance, as the old Stage 3 ground step's scale about the camera): with two or more cameras the data fix it,
with one the floor term does; MHR's own scale params stay near their start. A rigid rotation of an MHR body equals a change of its global_rot plus a shift, so
G and T span exactly the bodies MHR's own global rotation/translation span.

Start (start_body):
  1. per frame, the body of the camera with the most mask-visible keypoints (kp_conf), moved into the world frame;
     one shape and scale per person: the median over all cameras and frames
  2. translation shifted onto the joints triangulated from all cameras (least squares over the rays of every camera
     with kp_conf >= TRI_CONF; median over joints, median-filtered over time, interpolated where nobody triangulates)
  3. Hampel + zero-phase Gaussian per run of consecutive frames (pose, G as quaternions, T as xyz)
Fit (fit_person): Adam over all frames of the person at once, in chunks with context whose gradients are summed:
  keypoints  GMoF of the fisheye-projected mhr70 keypoints vs Stage 2's kp2d in every camera, x kp_conf, box units
  silhouette both ways, every camera: sampled vertices inside the person's Stage 1 mask (soft mask ROI; vertices
             on another person's mask count as inside: that person may hide this one), and points
             sampled from the mask near a projected vertex (fill: distance to the nearest one, box units, capped), so
             that a body can't satisfy the first term by shrinking
  pose       stay near the start pose; shape/scale near the start
  accel      3D acceleration of the body keypoints across consecutive frames
  highpass   |x - G_sigma * x|^2 of G (6D), pose and T: what the Stage 3 Gaussian smoothing would remove
  depth      frames with fewer than two usable cameras: distance from the best camera to the pelvis stays at the start
  floor      Huber of the lowest foot's height above a floor plane fitted to all people's start bodies
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import roma
import torch
import torch.nn.functional as Fn
from scipy.ndimage import median_filter

from smoothing.ground_plane_opt import FOOT_IDXS
from smoothing.temporal import _lowpass, SmoothConfig
from utils import fisheye

F_AXES = (1.0, -1.0, -1.0)
BODY_KPS = list(range(0, 21)) + list(range(63, 70))   # mhr70 body, feet and extra arm/neck points (not the hands)
TRI_KPS = list(range(5, 21))                           # shoulders .. feet: triangulated for the start translation
HIPS, SHOULDERS = (9, 10), (5, 6)
TRI_CONF = 0.7
MHR_KEYS = ('body_pose', 'hand', 'scale', 'shape', 'face')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def rot6d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = Fn.normalize(a1, dim=-1)
    b2 = Fn.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def gaussian_lowpass_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Zero-phase Gaussian along dim 0, replicate padding, differentiable (Video-OnlineHMR temporal_smooth.py)."""
    if sigma <= 0 or x.shape[0] < 2:
        return x
    r = int(4 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    shape = x.shape
    y = x.reshape(shape[0], -1).T[:, None]
    y = Fn.pad(y, (r, r), mode='replicate')
    return Fn.conv1d(y, k)[:, 0].T.reshape(shape)


def runs(frames: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end) index pairs of the runs of consecutive frame numbers in sorted `frames`."""
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(frames)]])
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))


def mhr_body(head_pose, p: Dict[str, torch.Tensor], vert_idx: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """MHR at global_rot = 0 in camera axes: keypoints (n, 70, 3) and, with vert_idx, those vertices (n, k, 3)."""
    n = p['body_pose'].shape[0]
    zero = torch.zeros(n, 3, device=p['body_pose'].device, dtype=p['body_pose'].dtype)
    out = head_pose.mhr_forward(global_trans=zero, global_rot=zero, body_pose_params=p['body_pose'],
                                hand_pose_params=p['hand'], scale_params=p['scale'], shape_params=p['shape'],
                                expr_params=p['face'], return_keypoints=True, do_pcblend=False)
    f = torch.tensor(F_AXES, device=zero.device, dtype=zero.dtype)
    kp = out[1][:, :70] * f
    v = out[0][:, vert_idx] * f if vert_idx is not None else None
    return kp, v


def mhr_full(head_pose, p: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """MHR at global_rot = 0 in camera axes: all vertices (n, V, 3) and keypoints (n, 70, 3), no grad."""
    n = p['body_pose'].shape[0]
    zero = torch.zeros(n, 3, device=p['body_pose'].device, dtype=p['body_pose'].dtype)
    with torch.no_grad():
        out = head_pose.mhr_forward(global_trans=zero, global_rot=zero, body_pose_params=p['body_pose'],
                                    hand_pose_params=p['hand'], scale_params=p['scale'], shape_params=p['shape'],
                                    expr_params=p['face'], return_keypoints=True, do_pcblend=False)
    f = torch.tensor(F_AXES, device=zero.device, dtype=zero.dtype)
    return out[0] * f, out[1][:, :70] * f


# ---------------------------------------------------------------------------
# data of one person in one camera
# ---------------------------------------------------------------------------

@dataclass
class CamTrack:
    """One person in one camera (Stage 2 --fisheye output + Stage 1 mask ROIs)."""
    cam: object                      # utils.cosilab_camera.Camera at the Stage 1 image size
    frames: np.ndarray               # (n,) batch frame numbers, sorted
    params: Dict[str, np.ndarray]    # global_rot, body_pose, hand, scale, shape, face: (n, D)
    cam_t: np.ndarray                # (n, 3)
    kp2d: np.ndarray                 # (n, 70, 2)
    kp_conf: np.ndarray              # (n, 70)
    tid: np.ndarray                  # (n,) Stage 1 tracking (mask pixel) ID
    euler: str = 'xyz'               # MHR global_rot convention (Stage 2 meta)
    weight: float = 1.0
    holdout: bool = False            # evaluated, not fitted (leave-one-camera-out)
    # mask ROIs (filled by stage3_mvfit.load_mask_rois): soft mask (n, R, R) uint8, ROI box (n, 4) xyxy px,
    # mask centroid / extent (n, 2) px, has_mask (n,)
    roi: Optional[np.ndarray] = None
    roi_other: Optional[np.ndarray] = None  # (n, R, R) uint8 soft mask of everyone else (possible occluders)
    roi_box: Optional[np.ndarray] = None
    m_center: Optional[np.ndarray] = None
    m_ext: Optional[np.ndarray] = None
    mask_pts: Optional[np.ndarray] = None  # (n, P, 2) px sampled from the mask
    has_mask: Optional[np.ndarray] = None
    box_size: Optional[np.ndarray] = None  # (n,) px, for the keypoint residual units

    def usable(self, min_kp: int = 3) -> np.ndarray:
        return (self.kp_conf[:, BODY_KPS] >= TRI_CONF).sum(1) >= min_kp


@dataclass
class FitConfig:
    iters: int = 300
    lr: float = 0.01
    window: int = 256           # frames per chunk (memory only: gradients of all chunks are summed)
    n_verts: int = 1000         # sampled vertices for the silhouette terms
    gmof_sigma: float = 0.15    # box-size units
    hand_weight: float = 0.2    # keypoint weight of the hand keypoints vs the body ones
    w_kp: float = 20.0
    w_sil: float = 0.3
    w_fill: float = 5.0
    fill_cap: float = 0.3       # box units: mask points further than this from the body cost the same
    w_cov: float = 0.0          # centroid / extent of the projected vertices vs the mask's (mask_reproj_opt.py)
    w_pose: float = 0.01
    w_shape: float = 0.01
    w_size: float = 0.1         # prior on log_s (the per-person size factor)
    w_accel: float = 1000.0
    w_highpass: float = 1.0
    w_depth: float = 0.1
    w_floor: float = 0.5
    floor_huber: float = 0.10   # m
    sigma_pose: float = 2.0
    sigma_rot: float = 3.0
    sigma_trans: float = 4.0
    hampel_half_window: int = 3
    hampel_n_sigma: float = 3.0
    smooth_start: bool = True


@dataclass
class Body:
    frames: np.ndarray                 # (T,) batch frame numbers
    g6d: torch.Tensor                  # (T, 6)
    trans: torch.Tensor                # (T, 3)
    pose: torch.Tensor                 # (T, 133)
    hand: torch.Tensor                 # (T, 108)
    face: torch.Tensor                 # (T, 72)
    shape: torch.Tensor                # (1, 45)
    scale: torch.Tensor                # (1, 28)
    log_s: torch.Tensor = None         # (1,) size factor s = exp(log_s)
    best_cam: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # (T,) index into the tracks
    n_views: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # (T,) cameras with usable keypoints
    stats: Dict[str, float] = field(default_factory=dict)

    def params(self, a: int, b: int) -> Dict[str, torch.Tensor]:
        n = b - a
        return {'body_pose': self.pose[a:b], 'hand': self.hand[a:b], 'face': self.face[a:b],
                'shape': self.shape.expand(n, -1), 'scale': self.scale.expand(n, -1)}

    def world(self, local: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """(n, k, 3) MHR camera-axes points of frames [a, b) -> world."""
        R = rot6d_to_rotmat(self.g6d[a:b])
        s = torch.exp(self.log_s) if self.log_s is not None else 1.0
        return torch.einsum('nij,nkj->nki', R, local * s) + self.trans[a:b, None]


def to_world(cam, X_cam: np.ndarray) -> np.ndarray:
    return (X_cam - cam.t) @ cam.R


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def triangulate(tracks: List[CamTrack], frames: np.ndarray, max_gap: float = 0.3, min_angle_deg: float = 5.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """TRI_KPS joints triangulated per frame by least squares over every camera with kp_conf >= TRI_CONF.
    Returns X (T, J, 3) (NaN where not triangulated) and the number of cameras used (T, J)."""
    T, J = len(frames), len(TRI_KPS)
    A = np.zeros((T, J, 3, 3))
    b = np.zeros((T, J, 3))
    n = np.zeros((T, J), int)
    rays = []
    for tr in tracks:
        if tr.holdout:
            continue
        rows = np.searchsorted(frames, tr.frames)
        u = tr.cam.rays(tr.kp2d[:, TRI_KPS])                    # (n, J, 3) world
        ok = tr.kp_conf[:, TRI_KPS] >= TRI_CONF
        P = np.eye(3) - u[..., :, None] * u[..., None, :]
        P[~ok] = 0
        A[rows] += P
        b[rows] += P @ tr.cam.centre
        n[rows] += ok
        rays.append((rows, u, ok, tr.cam.centre))
    X = np.full((T, J, 3), np.nan)
    multi = n >= 2
    X[multi] = np.linalg.solve(A[multi] + 1e-9 * np.eye(3), b[multi][..., None])[..., 0]
    good = multi.copy()
    cos_min = np.ones((T, J))
    us = []
    for rows, u, ok, C in rays:
        d = X[rows] - C
        lam = np.einsum('tjk,tjk->tj', d, u)
        gap = np.linalg.norm(d - lam[..., None] * u, axis=-1)
        bad = ok & ~((lam > 0) & (gap <= max_gap))
        good[rows] &= ~bad
        U = np.full((T, J, 3), np.nan)
        U[rows[:, None].repeat(J, 1)[ok], np.nonzero(ok)[1]] = u[ok]
        us.append(U)
    for i in range(len(us)):
        for k in range(i + 1, len(us)):
            c = np.einsum('tjk,tjk->tj', us[i], us[k])
            both = ~np.isnan(c)
            cos_min[both] = np.minimum(cos_min[both], c[both])
    good &= cos_min <= math.cos(math.radians(min_angle_deg))
    X[~good] = np.nan
    return X, np.where(good, n, 0)


def start_body(head_pose, tracks: List[CamTrack], device, cfg: FitConfig) -> Body:
    fit_tracks = [t for t in tracks if not t.holdout]
    frames = np.unique(np.concatenate([t.frames for t in fit_tracks]))
    T = len(frames)
    score = np.full((len(fit_tracks), T), -1.0)
    for i, tr in enumerate(fit_tracks):
        score[i, np.searchsorted(frames, tr.frames)] = tr.kp_conf[:, BODY_KPS].sum(1)
    best = score.argmax(0)
    n_views = np.zeros(T, int)
    for tr in fit_tracks:
        n_views[np.searchsorted(frames, tr.frames)] += tr.usable()

    keys = ('global_rot',) + MHR_KEYS
    sel = {k: np.zeros((T, fit_tracks[0].params[k].shape[1]), np.float32) for k in keys}
    cam_t = np.zeros((T, 3), np.float32)
    cam_R = np.zeros((T, 3, 3))
    cam_tv = np.zeros((T, 3))
    euler = fit_tracks[0].euler
    for i, tr in enumerate(fit_tracks):
        rows = np.searchsorted(frames, tr.frames)
        mine = best[rows] == i
        for k in keys:
            sel[k][rows[mine]] = tr.params[k][mine]
        cam_t[rows[mine]] = tr.cam_t[mine]
        cam_R[rows[mine]] = tr.cam.R
        cam_tv[rows[mine]] = tr.cam.t
    shape = np.median(np.concatenate([t.params['shape'] for t in fit_tracks]), 0)[None].astype(np.float32)
    scale = np.median(np.concatenate([t.params['scale'] for t in fit_tracks]), 0)[None].astype(np.float32)

    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    G = np.zeros((T, 3, 3))
    trans = np.zeros((T, 3))
    f = np.diag(F_AXES)
    chunk = 1024
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        own = {k: tt(sel[k][a:b]) for k in MHR_KEYS}
        med = dict(own, shape=tt(shape).expand(b - a, -1), scale=tt(scale).expand(b - a, -1))
        with torch.no_grad():
            out = head_pose.mhr_forward(global_trans=tt(sel['global_rot'][a:b]) * 0,
                                        global_rot=tt(sel['global_rot'][a:b]), body_pose_params=own['body_pose'],
                                        hand_pose_params=own['hand'], scale_params=own['scale'],
                                        shape_params=own['shape'], expr_params=own['face'], return_keypoints=True,
                                        do_pcblend=False)
            kp_own = (out[1][:, :70] * tt(F_AXES)).double().cpu().numpy()
            kp0 = mhr_body(head_pose, med)[0].double().cpu().numpy()
        # camera -> world: X_w = R^T (X_c - t)
        Rc, tc = cam_R[a:b], cam_tv[a:b]
        target = np.einsum('nji,nkj->nki', Rc, kp_own + cam_t[a:b, None] - tc[:, None])
        E = roma.euler_to_rotmat(euler, torch.as_tensor(sel['global_rot'][a:b], dtype=torch.float64)).numpy()
        G[a:b] = np.einsum('nji,jk,nkl,lm->nim', Rc, f, E, f)   # R_c^T F E F
        trans[a:b] = (target - np.einsum('nij,nkj->nki', G[a:b], kp0)).mean(1)
    stats = {}

    # translation onto the triangulated joints
    X_tri, _ = triangulate(fit_tracks, frames)
    shift = {}
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        med = {k: tt(sel[k][a:b]) for k in MHR_KEYS}
        med.update(shape=tt(shape).expand(b - a, -1), scale=tt(scale).expand(b - a, -1))
        with torch.no_grad():
            kp0 = mhr_body(head_pose, med)[0].double().cpu().numpy()
        Xw = np.einsum('nij,nkj->nki', G[a:b], kp0[:, TRI_KPS]) + trans[a:b, None]
        d = X_tri[a:b] - Xw
        for i in range(b - a):
            ok = ~np.isnan(d[i, :, 0])
            if ok.sum() >= 3:
                shift[a + i] = np.median(d[i, ok], 0)
    stats['triangulated_frames'] = len(shift)
    if len(shift) >= 10:
        known = np.array(sorted(shift))
        s = median_filter(np.stack([shift[k] for k in known]), size=(31, 1), mode='nearest')
        full = np.stack([np.interp(np.arange(T), known, s[:, i]) for i in range(3)], 1)
        trans = trans + full
        stats['start_shift_mean_m'] = float(np.linalg.norm(full, axis=1).mean())

    if cfg.smooth_start:
        scfg = SmoothConfig(cfg.sigma_pose, cfg.sigma_rot, cfg.sigma_trans, cfg.hampel_half_window,
                            cfg.hampel_n_sigma)
        q = roma.rotmat_to_unitquat(torch.as_tensor(G)).numpy()
        for s0, e0 in runs(frames):
            if e0 - s0 < 2:
                continue
            for k in ('body_pose', 'hand', 'face'):
                sel[k][s0:e0] = _lowpass(sel[k][s0:e0].astype(np.float64), cfg.sigma_pose, scfg)
            qq = q[s0:e0].copy()
            flip = np.cumprod(np.where(np.sum(qq[1:] * qq[:-1], axis=1) < 0, -1.0, 1.0))
            qq[1:] *= flip[:, None]
            qq = _lowpass(qq, cfg.sigma_rot, scfg)
            q[s0:e0] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
            trans[s0:e0] = _lowpass(trans[s0:e0], cfg.sigma_trans, scfg)
        G = roma.unitquat_to_rotmat(torch.as_tensor(q)).numpy()

    return Body(frames=frames, g6d=rotmat_to_6d(tt(G)), trans=tt(trans), pose=tt(sel['body_pose']),
                hand=tt(sel['hand']), face=tt(sel['face']), shape=tt(shape), scale=tt(scale),
                log_s=torch.zeros(1, device=device), best_cam=best, n_views=n_views, stats=stats)


# ---------------------------------------------------------------------------
# floor
# ---------------------------------------------------------------------------

def plane_fit(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Robust plane fit (Video-OnlineHMR floor_spread): residual std (m), unit normal, centroid, kept fraction."""
    keep = np.ones(len(points), dtype=bool)
    for _ in range(3):
        c = points[keep].mean(0)
        normal = np.linalg.svd(points[keep] - c, full_matrices=False)[2][-1]
        res = (points - c) @ normal
        keep = np.abs(res) < 3 * res[keep].std()
    c = points[keep].mean(0)
    return float(((points[keep] - c) @ normal).std()), normal, c, float(keep.mean())


def lowest_feet(head_pose, bodies: List[Body], up: Optional[np.ndarray]) -> np.ndarray:
    """Per person-frame the foot keypoint lowest along `up` (or all foot keypoints when up is None), world (N, 3)."""
    pts = []
    for bd in bodies:
        for a in range(0, len(bd.frames), 1024):
            b = min(len(bd.frames), a + 1024)
            with torch.no_grad():
                kp = bd.world(mhr_body(head_pose, bd.params(a, b))[0], a, b)[:, list(FOOT_IDXS)]
            kp = kp.double().cpu().numpy()
            if up is None:
                pts.append(kp.reshape(-1, 3))
            else:
                h = kp @ up
                pts.append(kp[np.arange(len(kp)), h.argmin(1)])
    return np.concatenate(pts)


def fit_floor(head_pose, bodies: List[Body], down_hint: np.ndarray) -> Dict[str, object]:
    """Floor plane from the lowest foot of every person-frame. down_hint: a rough world 'down' direction (e.g. the
    reference camera's optical axis); only its sign matters."""
    _, n0, _, _ = plane_fit(lowest_feet(head_pose, bodies, None))
    up = -n0 if n0 @ down_hint > 0 else n0
    pts = lowest_feet(head_pose, bodies, up)
    spread, normal, c, kept = plane_fit(pts)
    up = -normal if normal @ down_hint > 0 else normal
    return {'point': c, 'up': up, 'spread_m': spread, 'kept': kept, 'n': len(pts)}


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class _TrackTensors:
    """Per-camera tensors of one person on the fit's frame axis."""

    def __init__(self, tr: CamTrack, frames: np.ndarray, device, cfg: FitConfig):
        self.tr = tr
        self.rows = torch.as_tensor(np.searchsorted(frames, tr.frames), device=device)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.R, self.t = t(tr.cam.R), t(tr.cam.t)
        self.K, self.D = t(tr.cam.K), t(tr.cam.D)
        self.C = t(tr.cam.centre)
        self.kp2d = t(tr.kp2d)
        jw = np.full(70, cfg.hand_weight, np.float32)
        jw[BODY_KPS] = 1.0
        self.kw = t(tr.kp_conf * jw[None]) * tr.weight
        self.size = t(tr.box_size)
        self.has_mask = torch.as_tensor(tr.has_mask, device=device)
        self.roi = torch.as_tensor(tr.roi, device=device)
        self.roi_other = torch.as_tensor(tr.roi_other, device=device)
        self.roi_box = t(tr.roi_box)
        self.m_center, self.m_ext = t(tr.m_center), t(tr.m_ext)
        self.mask_pts = t(tr.mask_pts)

    def project(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Xc = X @ self.R.T + self.t
        return fisheye.project_torch(Xc, self.K, self.D), Xc[..., 2]


def sample_roi(roi: torch.Tensor, box: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    """Bilinear soft-mask values (n, k) of ROI masks (n, R, R) uint8 at image pixels px (n, k, 2); 0 outside."""
    R = roi.shape[-1]
    side = (box[:, 2] - box[:, 0]).clamp(min=1)
    u = (px + 0.5 - box[:, None, :2]) / side[:, None, None]           # [0, 1] across the ROI (see load_mask_rois)
    grid = (2 * u - 1).unsqueeze(1)                                   # (n, 1, k, 2), align_corners=False
    m = roi.float().unsqueeze(1) / 255.0
    return Fn.grid_sample(m, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0, 0]


def view_losses(tt: _TrackTensors, kp_w: torch.Tensor, v_w: torch.Tensor, f_idx: torch.Tensor, t_idx: torch.Tensor,
                cfg: FitConfig) -> Dict[str, torch.Tensor]:
    """Keypoint and silhouette sums of one camera over the samples (chunk frame f_idx, track row t_idx)."""
    px, z = tt.project(kp_w[f_idx])
    res = ((px - tt.kp2d[t_idx]) / tt.size[t_idx, None, None]).pow(2).sum(-1)
    s2 = cfg.gmof_sigma ** 2
    l_kp = (tt.kw[t_idx] * (s2 * res / (s2 + res)) * (z > 0.05)).sum()
    out = {'kp': l_kp}
    hm = tt.has_mask[t_idx]
    if hm.any():
        fi, ti = f_idx[hm], t_idx[hm]
        vpx, vz = tt.project(v_w[fi])
        front = (vz > 0.05).float()
        inside = torch.maximum(sample_roi(tt.roi[ti], tt.roi_box[ti], vpx),
                               sample_roi(tt.roi_other[ti], tt.roi_box[ti], vpx))
        n_front = front.sum(1).clamp(min=1)
        out['sil'] = (1 - (inside * front).sum(1) / n_front).sum()
        size = tt.size[ti, None]
        far = torch.where(front[..., None] > 0, vpx, torch.full_like(vpx, 1e6))
        # exact distances: the default matmul form is imprecise in float32 (and not reproducible on the GPU)
        d = torch.cdist(tt.mask_pts[ti], far, compute_mode='donot_use_mm_for_euclid_dist').amin(-1) / size  # (n, P)
        out['fill'] = d.clamp(max=cfg.fill_cap).mean(-1).sum()
        cen = (vpx * front[..., None]).sum(1) / n_front[:, None]
        big = torch.where(front[..., None] > 0, vpx, torch.full_like(vpx, -1e6)).amax(1)
        small = torch.where(front[..., None] > 0, vpx, torch.full_like(vpx, 1e6)).amin(1)
        out['cov'] = (((cen - tt.m_center[ti]) / size).pow(2).sum(-1)
                      + ((big - small - tt.m_ext[ti]) / size).pow(2).sum(-1)).sum()
    return out


def fit_person(head_pose, tracks: List[CamTrack], body: Body, cfg: FitConfig, device,
               floor: Optional[Dict[str, object]] = None, log=print) -> Body:
    frames = body.frames
    T = len(frames)
    fit_tracks = [t for t in tracks if not t.holdout]
    tts = [_TrackTensors(t, frames, device, cfg) for t in fit_tracks]
    gen = torch.Generator().manual_seed(0)
    vert_idx = torch.randperm(18439, generator=gen)[:cfg.n_verts].to(device)

    single = torch.as_tensor(body.n_views < 2, device=device)
    best_tr = [tts[i] for i in range(len(tts))]
    best = torch.as_tensor(body.best_cam, device=device)
    C_best = torch.stack([tt.C for tt in best_tr])[best]              # (T, 3)
    with torch.no_grad():
        pelvis0 = torch.cat([body.world(mhr_body(head_pose, body.params(a, min(T, a + 1024)))[0][:, list(HIPS)],
                                        a, min(T, a + 1024)).mean(1) for a in range(0, T, 1024)])
    depth0 = (pelvis0 - C_best).norm(dim=-1)

    # normalisers over the whole sequence, so that summed chunk losses equal the full loss
    kw_total = sum(tt.kw.sum() for tt in tts).clamp(min=1)
    sil_total = max(sum(int(tt.has_mask.sum()) for tt in tts), 1)
    single_total = max(int(single.sum()), 1)
    run_list = runs(frames)
    n_accel = max(sum(max(e - s - 2, 0) for s, e in run_list), 1)
    margin = max(2, int(4 * max(cfg.sigma_pose, cfg.sigma_rot, cfg.sigma_trans) + 0.5))
    chunks = []
    for s, e in run_list:
        n = int(math.ceil((e - s) / cfg.window))
        bounds = np.linspace(s, e, n + 1).round().astype(int)
        chunks += [(cs, ce, max(s, cs - margin), min(e, ce + margin)) for cs, ce in zip(bounds[:-1], bounds[1:])]
    # per chunk and camera: (frame index in the chunk context, track row) of the core samples
    samples = []
    for cs, ce, a, b in chunks:
        per = []
        for tt in tts:
            m = (tt.rows >= cs) & (tt.rows < ce)
            t_idx = torch.nonzero(m)[:, 0]
            per.append((tt.rows[t_idx] - a, t_idx))
        samples.append(per)

    pose0 = body.pose.clone()
    shape0, scale0 = body.shape.clone(), body.scale.clone()
    g6d = body.g6d.clone().requires_grad_(True)
    trans = body.trans.clone().requires_grad_(True)
    pose = body.pose.clone().requires_grad_(True)
    shape = body.shape.clone().requires_grad_(True)
    scale = body.scale.clone().requires_grad_(True)
    log_s = body.log_s.clone().requires_grad_(True)
    opt = torch.optim.Adam([g6d, trans, pose, shape, scale, log_s], lr=cfg.lr)
    if floor is not None:
        f_c = torch.as_tensor(floor['point'], dtype=torch.float32, device=device)
        f_up = torch.as_tensor(floor['up'], dtype=torch.float32, device=device)
    body_kps = torch.as_tensor(BODY_KPS, device=device)

    hist = []
    with torch.enable_grad():
        for it in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            parts = {}
            for k_chunk, (cs, ce, a, b) in enumerate(chunks):
                core = slice(cs - a, ce - a)
                fb = Body(frames, g6d, trans, pose, body.hand, body.face, shape, scale, log_s)
                kp_l, v_l = mhr_body(head_pose, fb.params(a, b), vert_idx)
                kp_w, v_w = fb.world(kp_l, a, b), fb.world(v_l, a, b)
                loss = {'kp': 0, 'sil': 0, 'fill': 0, 'cov': 0}
                for tt, (f_idx, t_idx) in zip(tts, samples[k_chunk]):
                    if len(f_idx) == 0:
                        continue
                    for k, v in view_losses(tt, kp_w, v_w, f_idx, t_idx, cfg).items():
                        loss[k] = loss[k] + v
                loss['kp'] = loss['kp'] / kw_total
                loss['sil'] = loss['sil'] / sil_total
                loss['cov'] = loss['cov'] / sil_total
                loss['fill'] = loss['fill'] / sil_total
                loss['pose'] = (pose[cs:ce] - pose0[cs:ce]).pow(2).sum() / T
                kb = kp_w[:, body_kps]
                m0, m1 = max(cs - a, 1), min(ce - a, b - a - 1)
                loss['accel'] = ((kb[m0 + 1:m1 + 1] - 2 * kb[m0:m1] + kb[m0 - 1:m1 - 1]).pow(2).sum(-1).mean(-1).sum()
                                 / n_accel) if m1 > m0 else torch.zeros((), device=device)
                hp = lambda x, s: (x - gaussian_lowpass_torch(x, s))[core].pow(2).flatten(1).sum(-1).sum()  # noqa
                loss['highpass'] = (hp(g6d[a:b], cfg.sigma_rot) + hp(pose[a:b], cfg.sigma_pose)
                                    + hp(trans[a:b], cfg.sigma_trans)) / T
                pel = kp_w[core][:, list(HIPS)].mean(1)
                d = (pel - C_best[cs:ce]).norm(dim=-1)
                loss['depth'] = (single[cs:ce].float() * (d - depth0[cs:ce]).pow(2)).sum() / single_total
                if floor is not None and cfg.w_floor > 0:
                    h = ((kp_w[core][:, list(FOOT_IDXS)] - f_c) @ f_up).amin(1)
                    loss['floor'] = Fn.huber_loss(h, torch.zeros_like(h), delta=cfg.floor_huber,
                                                  reduction='sum') / T
                total = (cfg.w_kp * loss['kp'] + cfg.w_sil * loss['sil'] + cfg.w_fill * loss['fill']
                         + cfg.w_cov * loss['cov']
                         + cfg.w_pose * loss['pose'] + cfg.w_accel * loss['accel']
                         + cfg.w_highpass * loss['highpass'] + cfg.w_depth * loss['depth']
                         + cfg.w_floor * loss.get('floor', 0))
                if k_chunk == 0:
                    loss['shape'] = (shape - shape0).pow(2).sum() + (scale - scale0).pow(2).sum()
                    loss['size'] = log_s.pow(2).sum()
                    total = total + cfg.w_shape * loss['shape'] + cfg.w_size * loss['size']
                total.backward()
                for k, v in loss.items():
                    parts[k] = parts.get(k, 0.0) + float(v)
            opt.step()
            if it in (0, cfg.iters - 1) or (it + 1) % 50 == 0:
                hist.append((it, parts))
                log(f'    iter {it + 1:4d}: ' + ', '.join(f'{k} {v:.4g}' for k, v in parts.items()))
    out = Body(frames, g6d.detach(), trans.detach(), pose.detach(), body.hand, body.face, shape.detach(),
               scale.detach(), log_s.detach(), body.best_cam, body.n_views, dict(body.stats))
    out.stats['size_factor'] = float(torch.exp(log_s.detach()))
    out.stats['loss_first'] = hist[0][1] if hist else {}
    out.stats['loss_last'] = hist[-1][1] if hist else {}
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def roi_silhouette(roi_box: torch.Tensor, px: torch.Tensor, front: torch.Tensor, R: int) -> torch.Tensor:
    """Projected vertices (n, V, 2) splatted into the ROI grid (n, R, R) bool, closed (dilate 3x3, erode 3x3)."""
    n = px.shape[0]
    side = (roi_box[:, 2] - roi_box[:, 0]).clamp(min=1)
    u = ((px + 0.5 - roi_box[:, None, :2]) / side[:, None, None] * R).floor().long()
    ok = front & (u >= 0).all(-1) & (u < R).all(-1)
    occ = torch.zeros(n, R * R, device=px.device)
    flat = (u[..., 1].clamp(0, R - 1) * R + u[..., 0].clamp(0, R - 1))
    occ.scatter_(1, torch.where(ok, flat, torch.zeros_like(flat)), ok.float())
    occ = occ.view(n, 1, R, R)
    occ = Fn.max_pool2d(occ, 3, 1, 1)
    occ = -Fn.max_pool2d(-occ, 3, 1, 1)
    return occ[:, 0] > 0.5


@torch.no_grad()
def evaluate(head_pose, tracks: List[CamTrack], body: Body, device) -> Dict[str, Dict[str, float]]:
    """Per camera: keypoint reprojection px (median / p90, kp_conf >= TRI_CONF body keypoints) and the fraction of
    sampled vertices inside the mask; plus the body's 3D acceleration (mm/frame^2, body keypoints)."""
    frames = body.frames
    gen = torch.Generator().manual_seed(1)
    vert_idx = torch.randperm(18439, generator=gen)[:400].to(device)
    T = len(frames)
    kp_all, v_all = [], []
    for a in range(0, T, 512):
        b = min(T, a + 512)
        kp, v = mhr_body(head_pose, body.params(a, b), vert_idx)
        kp_all.append(body.world(kp, a, b))
        v_all.append(body.world(v, a, b))
    kp_w, v_w = torch.cat(kp_all), torch.cat(v_all)

    def full_verts(rows: torch.Tensor) -> torch.Tensor:
        """All vertices (k, V, 3) of the fit-axis frames `rows`, world."""
        idx = rows.cpu().numpy()
        p = {k: v[idx] for k, v in body.params(0, T).items()}
        v, _ = mhr_full(head_pose, p)
        R = rot6d_to_rotmat(body.g6d[idx])
        size = torch.exp(body.log_s) if body.log_s is not None else 1.0
        return torch.einsum('nij,nkj->nki', R, v * size) + body.trans[idx, None]
    out = {}
    cfg = FitConfig()
    for tr in tracks:
        tt = _TrackTensors(tr, frames, device, cfg)
        px, z = tt.project(kp_w[tt.rows])
        err = (px - tt.kp2d).norm(dim=-1)
        sel = (torch.as_tensor(tr.kp_conf, device=device) >= TRI_CONF) & (z > 0.05)
        sel[:, [i for i in range(70) if i not in BODY_KPS]] = False
        e = err[sel].cpu().numpy()
        r = {'kp_px_median': float(np.median(e)) if e.size else float('nan'),
             'kp_px_p90': float(np.percentile(e, 90)) if e.size else float('nan'), 'samples': int(len(tr.frames))}
        hm = tt.has_mask
        if hm.any():
            vpx, vz = tt.project(v_w[tt.rows[hm]])
            inside = sample_roi(tt.roi[hm], tt.roi_box[hm], vpx)
            r['verts_in_mask'] = float(((inside > 0.5) & (vz > 0.05)).float().mean())
            # silhouette IoU in the mask ROI, all vertices
            rows = tt.rows[hm]
            ious = []
            for s in range(0, len(rows), 64):
                rr = rows[s:s + 64]
                vv = full_verts(rr)
                ppx, pz = tt.project(vv)
                sil = roi_silhouette(tt.roi_box[hm][s:s + 64], ppx, pz > 0.05, tt.roi.shape[-1])
                m = tt.roi[hm][s:s + 64] > 127
                ious.append(((sil & m).flatten(1).sum(1).float() / (sil | m).flatten(1).sum(1).clamp(min=1)))
            r['iou'] = float(torch.cat(ious).mean())
        out[f'{tr.cam.name}{" (held out)" if tr.holdout else ""}'] = r
    kb = kp_w[:, BODY_KPS]
    acc = []
    for s, e in runs(frames):
        if e - s >= 3:
            acc.append((kb[s + 2:e] - 2 * kb[s + 1:e - 1] + kb[s:e - 2]).norm(dim=-1).mean(-1))
    out['_body'] = {'accel_mm_per_frame2': float(torch.cat(acc).mean() * 1000) if acc else float('nan'),
                    'frames': T, 'multi_view_frames': int((body.n_views >= 2).sum())}
    return out
