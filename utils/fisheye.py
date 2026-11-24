"""
OpenCV fisheye camera helpers and per-person virtual pinhole cameras.

Ported from Video-OnlineHMR lib/utils/fisheye.py (render_crops also takes nearest-neighbour masks here). SAM-3D-Body
assumes a pinhole camera. For fisheye footage, each person is cropped through a virtual pinhole camera that shares the
optical centre of the real camera and looks at the person; the frames themselves are never undistorted. The model output
is then rotated from the virtual camera back into the real camera frame (utils/mhr_transform.py).

Conventions: K is 3x3, D = [k1, k2, k3, k4] (cv2.fisheye). R_v maps real camera coordinates to virtual camera
coordinates, X_v = R_v @ X_c. Scaling the image by s scales K (focal and principal point) only: D acts on the ray angle.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def unproject(pts, K, D):
    """Pixels (N, 2) -> unit rays (N, 3) in camera coordinates."""
    # a contiguous copy: cv2 misreads strided views such as kp[..., :2]
    pts = np.array(pts, dtype=np.float64, order='C').reshape(-1, 1, 2)
    xy = cv2.fisheye.undistortPoints(pts, np.asarray(K, np.float64), np.asarray(D, np.float64)).reshape(-1, 2)
    rays = np.concatenate([xy, np.ones_like(xy[:, :1])], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_torch(X, K, D, eps=1e-8):
    """
    Differentiable OpenCV fisheye projection.
    Args:
        X: (..., 3) points in camera coordinates
        K: (3, 3) tensor, D: (4,) tensor
    Returns:
        (..., 2) pixel coordinates
    """
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r = torch.sqrt(x**2 + y**2 + eps)
    theta = torch.atan2(r, z)
    t2 = theta**2
    theta_d = theta * (1 + D[0]*t2 + D[1]*t2**2 + D[2]*t2**3 + D[3]*t2**4)
    xd = theta_d * x / r
    yd = theta_d * y / r
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return torch.stack([u, v], dim=-1)


def virtual_camera(rays):
    """
    Rotations of virtual cameras looking along the given rays.
    The virtual x axis is the real x axis made orthogonal to the ray, so no roll is added.
    Args:
        rays: (P, 3) unit rays in camera coordinates
    Returns:
        R_v: (P, 3, 3), rows are the virtual x, y, z axes in camera coordinates
    """
    z = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    ex = np.array([1.0, 0.0, 0.0])
    x = ex[None] - (z @ ex)[:, None] * z
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def virtual_focal(R_v, rays, crop_size=256, dilate=1.2):
    """
    Focal length of each virtual camera such that the person fills the crop like TrackDataset does
    (square box of side max(w, h) * dilate around the box centre).
    Args:
        R_v: (P, 3, 3)
        rays: (P, M, 3) rays to points on the box outline
    """
    pts = np.einsum('pij,pmj->pmi', R_v, rays)
    xy = pts[..., :2] / np.clip(pts[..., 2:], 1e-6, None)
    half_extent = np.abs(xy).max(axis=(1, 2))
    return (crop_size / 2) / (dilate * half_extent)


def box_outline(boxes, n=3):
    """Points on the outline of xyxy boxes (P, 4) -> (P, 4*n, 2). Edges bend under fisheye, so sample them."""
    x1, y1, x2, y2 = [boxes[:, i:i+1] for i in range(4)]
    s = np.linspace(0, 1, n, endpoint=False)[None]
    top = np.stack([x1 + (x2-x1)*s, np.repeat(y1, n, 1)], -1)
    right = np.stack([np.repeat(x2, n, 1), y1 + (y2-y1)*s], -1)
    bottom = np.stack([x2 - (x2-x1)*s, np.repeat(y2, n, 1)], -1)
    left = np.stack([np.repeat(x1, n, 1), y2 - (y2-y1)*s], -1)
    return np.concatenate([top, right, bottom, left], axis=1)


def setup_virtual_cameras(boxes, K, D, crop_size=256, dilate=1.2):
    """
    Virtual cameras for xyxy boxes (P, 4) in full-resolution fisheye pixels.
    Returns R_v (P, 3, 3) and f_v (P,).
    """
    P = boxes.shape[0]
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    R_v = virtual_camera(unproject(centers, K, D))
    outline = box_outline(boxes)
    rays = unproject(outline.reshape(-1, 2), K, D).reshape(P, -1, 3)
    f_v = virtual_focal(R_v, rays, crop_size, dilate)
    return R_v, f_v


def render_crops(img, R_v, f_v, K, D, crop_size=256, mode='bilinear'):
    """
    Render the virtual pinhole crops from a fisheye image.
    Args:
        img: (C, H, W) float tensor, or (P, C, H, W) with one image per crop (e.g. per-person masks)
        R_v: (P, 3, 3) tensor, f_v: (P,) tensor, K: (3, 3) tensor, D: (4,) tensor
        mode: 'bilinear' for images, 'nearest' for label masks
    Returns:
        (P, C, crop_size, crop_size) tensor
    """
    P = R_v.shape[0]
    H, W = img.shape[-2:]
    device = img.device
    c = crop_size / 2
    uv = torch.arange(crop_size, device=device, dtype=torch.float32) - c
    vv, uu = torch.meshgrid(uv, uv, indexing='ij')
    rays_v = torch.stack([uu[None] / f_v[:, None, None],
                          vv[None] / f_v[:, None, None],
                          torch.ones(P, crop_size, crop_size, device=device)], dim=-1)
    rays_c = torch.einsum('pji,phwj->phwi', R_v, rays_v)  # R_v^T @ ray
    pix = project_torch(rays_c, K, D)
    grid = torch.stack([(2 * pix[..., 0] + 1) / W - 1, (2 * pix[..., 1] + 1) / H - 1], dim=-1)
    src = img if img.dim() == 4 else img[None].expand(P, -1, -1, -1)
    return F.grid_sample(src, grid, mode=mode, padding_mode='zeros', align_corners=False)


def virtual_K(f_v, crop_size=256):
    """(P,) focal lengths -> (P, 3, 3) intrinsics of the virtual pinhole crops (principal point at the crop centre,
    matching render_crops: crop pixel u looks along x = (u - crop_size/2) / f_v)."""
    P = f_v.shape[0]
    K = np.zeros((P, 3, 3), dtype=np.float64)
    K[:, 0, 0] = K[:, 1, 1] = f_v
    K[:, 0, 2] = K[:, 1, 2] = crop_size / 2
    K[:, 2, 2] = 1
    return K


def project_np(X, K, D):
    """Numpy wrapper of project_torch: camera points (..., 3) -> pixels (..., 2)."""
    out = project_torch(torch.as_tensor(np.asarray(X), dtype=torch.float64),
                        torch.as_tensor(np.asarray(K), dtype=torch.float64),
                        torch.as_tensor(np.asarray(D), dtype=torch.float64))
    return out.numpy()
