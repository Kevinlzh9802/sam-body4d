"""
Rigid transforms of MHR bodies between camera frames (Stage 2 virtual pinhole crops -> the real fisheye camera).

Camera-space body of SAM-3D-Body: X = F M(r, ...) + t, with M the MHR keypoints/vertices for global rotation r (Euler)
and global_trans = 0, F = diag(1, -1, -1) (MHR -> camera axes) and t = pred_cam_t. Rotating that body rigidly by
Q (X' = Q X, e.g. Q = R_v^T from a virtual camera to the real one) changes the Euler angles to those of F Q F R(r) and
adds a constant offset to t, because MHR rotates about a body point (the root) rather than its origin:

  t' = mean over keypoints of [ Q (F M(r) + t) - F M(r') ]

computed with one extra MHR forward. The residual of that fit is ~0 for a correct conversion and is returned so the
caller can log it. The Euler convention of MHR's global_rot is detected from the residual on the first call ('ZYX' as
the MHR head builds it, or 'xyz' as smoothing/temporal.py reads it), so a wrong guess can't pass silently.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import roma
import torch

F_AXES = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
CONVENTIONS = ('ZYX', 'xyz')
_detected: Dict[str, str] = {}
MAX_RESIDUAL_M = 1e-3  # a correct conversion is exact up to float error


def _mhr_keypoints(head_pose, params: Dict[str, torch.Tensor], global_rot: torch.Tensor) -> torch.Tensor:
    """(B, 70, 3) keypoints in camera axes (y, z flipped), local (no pred_cam_t)."""
    out = head_pose.mhr_forward(
        global_trans=global_rot * 0, global_rot=global_rot,
        body_pose_params=params['body_pose'], hand_pose_params=params['hand'],
        scale_params=params['scale'], shape_params=params['shape'], expr_params=params['face'],
        return_keypoints=True, return_joint_coords=False, return_model_params=False, return_joint_rotations=False)
    k = out[1][:, :70].float().clone()
    k[..., 1:3] *= -1
    return k


def rotate_euler(global_rot: torch.Tensor, Q: torch.Tensor, convention: str) -> torch.Tensor:
    """Euler angles of F Q F R(global_rot): the MHR global rotation of the body rotated by Q in camera axes."""
    F = F_AXES.to(Q)
    R = roma.euler_to_rotmat(convention, global_rot.to(Q))
    return roma.rotmat_to_euler(convention, F @ Q @ F @ R)


@torch.no_grad()
def rotate_body(head_pose, params: Dict[str, torch.Tensor], kp_local: torch.Tensor, cam_t: torch.Tensor,
                Q: torch.Tensor, convention: Optional[str] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Rotate B camera-space MHR bodies by Q (B, 3, 3) about the camera centre.
    params: MHR params of the bodies (global_rot, body_pose, hand, scale, shape, face), (B, D) tensors on the model's
    device; kp_local (B, 70, 3) their keypoints in camera axes without cam_t; cam_t (B, 3).
    Returns (global_rot', cam_t', kp_local', residual (B,) in m, convention).
    """
    target = torch.einsum('bij,bkj->bki', Q, kp_local + cam_t[:, None])
    tried = [convention] if convention else ([_detected['c']] if 'c' in _detected else list(CONVENTIONS))
    best = None
    for conv in tried:
        rot = rotate_euler(params['global_rot'].float(), Q.float(), conv)
        kp = _mhr_keypoints(head_pose, params, rot.to(params['global_rot']))
        t_new = (target - kp).mean(1)
        res = (kp + t_new[:, None] - target).norm(dim=-1).max(1).values
        if best is None or res.max() < best[3].max():
            best = (rot, t_new, kp, res, conv)
    rot, t_new, kp, res, conv = best
    if 'c' not in _detected:
        if res.max() > MAX_RESIDUAL_M:
            raise RuntimeError(f'MHR rigid rotation: residual {res.max():.4f} m with every Euler convention '
                               f'{CONVENTIONS}; the global_rot convention is not understood')
        _detected['c'] = conv
        print(f'[INFO] MHR global_rot Euler convention: {conv!r} (rigid-rotation residual {res.max() * 1e3:.3f} mm)')
    return rot, t_new, kp, res, conv


def self_check() -> None:
    """Model-free check of the rotation algebra: a toy 'MHR' rotating points about a pivot."""
    torch.manual_seed(0)
    pts0 = torch.randn(70, 3, dtype=torch.float64)
    pivot = torch.tensor([0.1, -0.9, 0.05], dtype=torch.float64)

    class Toy:
        def mhr_forward(self, global_rot, **kw):
            R = roma.euler_to_rotmat('ZYX', global_rot.double())
            k = torch.einsum('bij,kj->bki', R, pts0 - pivot) + pivot
            return None, k

    B = 8
    params = {k: torch.zeros(B, 1, dtype=torch.float64) for k in ('body_pose', 'hand', 'scale', 'shape', 'face')}
    params['global_rot'] = torch.randn(B, 3, dtype=torch.float64)
    kp = _mhr_keypoints(Toy(), params, params['global_rot']).double()
    t = torch.randn(B, 3, dtype=torch.float64) + torch.tensor([0, 0, 4.0], dtype=torch.float64)
    Q = roma.random_rotmat(B).double()
    _detected.clear()
    rot, t2, kp2, res, conv = rotate_body(Toy(), params, kp, t, Q)
    want = torch.einsum('bij,bkj->bki', Q, kp + t[:, None])
    err = (kp2 + t2[:, None] - want).norm(dim=-1).max()
    _detected.clear()
    print(f'[{"OK" if err < 1e-5 and conv == "ZYX" else "FAIL"}] mhr_transform toy check: max error {err:.2e} m, '
          f'convention {conv}')
