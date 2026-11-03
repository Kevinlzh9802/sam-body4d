"""
Stage 3 temporal smoothing of the raw MHR parameters, per person.

Stage 2 estimates every frame independently, so its noise is close to white:
a zero-phase low-pass removes it without the lag of a causal filter (EMA,
Kalman). Per person and per run of consecutive frames:

  body_pose, hand, face   Hampel filter (isolated spikes -> local median),
                          then Gaussian, sigma_pose frames
  global_rot (xyz Euler)  same on sign-aligned unit quaternions (Euler angles
                          wrap at +-pi: a body seen from above is ~180 deg
                          about x), renormalized
  pred_cam_t              same, on depth z and on the image ray (x/z, y/z)
                          separately, so smoothing never moves the person
                          sideways in the image more than the noise does
  shape, scale            one value per person: the median over all of the
                          person's frames (body size and proportions do not
                          change; the first frame may be a bad one)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import roma
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d


@dataclass
class SmoothConfig:
    # Gaussian widths in frames. At 60 fps a Gaussian of sigma frames keeps
    # motion up to ~8/sigma Hz (half power): gestures need ~4 Hz, turning the
    # body and moving around much less. Chosen with scripts/jitter_check.py
    # (cam06 seg010, cam08 seg040): beyond these, what the smoothing removes
    # grows well past the raw noise level, i.e. real motion goes too.
    sigma_pose: float = 2.0     # body_pose, hand, face: ~4 Hz
    sigma_rot: float = 3.0      # global_rot: ~2.7 Hz
    sigma_cam: float = 4.0      # pred_cam_t: ~2 Hz (mask reprojection refines it after)
    hampel_half_window: int = 3  # spike detection window: 2h+1 frames
    hampel_n_sigma: float = 3.0  # spike = further than this many robust stds from the local median


def person_runs(present: np.ndarray, frame_numbers: np.ndarray) -> List[np.ndarray]:
    """Index arrays of the runs where the person is present in consecutive frame numbers."""
    t = np.flatnonzero(present)
    if t.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frame_numbers[t]) != 1) + 1
    return np.split(t, breaks)


def hampel(x: np.ndarray, half_window: int, n_sigma: float) -> np.ndarray:
    """Replace samples further than n_sigma robust stds from the local median by
    that median (per dimension). x: (L, D)."""
    L = x.shape[0]
    if half_window <= 0 or L < 2 * half_window + 1:
        return x
    padded = np.pad(x, ((half_window, half_window), (0, 0)), mode="edge")
    win = sliding_window_view(padded, 2 * half_window + 1, axis=0)  # (L, D, w)
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    out = x.copy()
    spike = np.abs(x - med) > n_sigma * np.maximum(mad, 1e-9)
    out[spike] = med[spike]
    return out


def _lowpass(x: np.ndarray, sigma: float, cfg: SmoothConfig) -> np.ndarray:
    x = hampel(x, cfg.hampel_half_window, cfg.hampel_n_sigma)
    return gaussian_filter1d(x, sigma, axis=0, mode="nearest") if sigma > 0 else x


def smooth_euler_xyz(eul: np.ndarray, sigma: float, cfg: SmoothConfig) -> np.ndarray:
    """(L, 3) xyz Euler angles, smoothed as sign-aligned unit quaternions."""
    q = roma.rotmat_to_unitquat(roma.euler_to_rotmat("xyz", torch.from_numpy(eul).double())).numpy()
    flip = np.cumprod(np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1.0, 1.0))
    q[1:] *= flip[:, None]
    q = _lowpass(q, sigma, cfg)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return roma.rotmat_to_euler("xyz", roma.unitquat_to_rotmat(torch.from_numpy(q))).numpy()


def smooth_cam_t(t: np.ndarray, sigma: float, cfg: SmoothConfig) -> np.ndarray:
    """(L, 3) pred_cam_t, smoothed as depth and image ray separately."""
    z = t[:, 2:3]
    ray = t[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)
    z = _lowpass(z, sigma, cfg)
    ray = _lowpass(ray, sigma, cfg)
    return np.concatenate([ray * z, z], axis=1)


def smooth_mhr_params(
    mhr: Dict[str, torch.Tensor],
    frame_names: List[str],
    frame_obj_ids_slots: List[List[int]],
    obj_ids_all: List[int],
    cfg: SmoothConfig = SmoothConfig(),
) -> Dict[str, torch.Tensor]:
    """Smoothed copy of the stacked (T*N, D) params (see the module docstring).
    Absent (frame, person) slots are left as they are."""
    T, N = len(frame_names), len(obj_ids_all)
    frame_numbers = np.array([int(n) for n in frame_names])
    out = {}
    arrays = {k: v.detach().cpu().double().numpy().reshape(T, N, -1).copy() for k, v in mhr.items()}
    for si, oid in enumerate(obj_ids_all):
        present = np.array([frame_obj_ids_slots[t][si] == oid for t in range(T)])
        if not present.any():
            continue
        for k in ("shape", "scale"):
            if k in arrays:
                arrays[k][present, si] = np.median(arrays[k][present, si], axis=0)
        for run in person_runs(present, frame_numbers):
            if len(run) < 2:
                continue
            for k in ("body_pose", "hand", "face"):
                if k in arrays:
                    arrays[k][run, si] = _lowpass(arrays[k][run, si], cfg.sigma_pose, cfg)
            if "global_rot" in arrays:
                arrays["global_rot"][run, si] = smooth_euler_xyz(arrays["global_rot"][run, si], cfg.sigma_rot, cfg)
            if "pred_cam_t" in arrays:
                arrays["pred_cam_t"][run, si] = smooth_cam_t(arrays["pred_cam_t"][run, si], cfg.sigma_cam, cfg)
    for k, v in mhr.items():
        out[k] = torch.from_numpy(arrays[k].reshape(T * N, -1)).to(dtype=v.dtype, device=v.device)
    return out
