"""
Stage 2 on fisheye footage without undistorting it: every (frame, person) is cropped through a virtual pinhole camera
that shares the optical centre of the real camera and looks at the person (utils/fisheye.py, the Video-OnlineHMR
trick). SAM-3D-Body runs on the crop with the crop's pinhole intrinsics; its output is rotated back into the real
camera (utils/mhr_transform.py), so raw_mhr.pt holds real-camera global_rot / pred_cam_t as before and Stage 3's
smoothing works unchanged.

Per person the output also has:
  R_v (3, 3), f_v        the virtual camera (X_v = R_v X_c) and its focal length in crop pixels
  kp2d (70, 2)           the mhr70 keypoints projected with the real fisheye model, in Stage 1 image pixels
  kp_conf (70,)          from the Stage 1 masks (SAM-3D-Body gives no confidence): 1 on the person's own mask, 0.7
                         within a few pixels of it, 0 on another person's mask or outside the image, else 0.3
  rot_residual           max keypoint error (m) of the virtual->real rigid rotation (~0)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from utils import fisheye
from utils.mhr_transform import rotate_body

MARGIN_FRAC = 0.05  # people entirely within this border band are left out (as process_image_with_mask does)
PARAM_KEYS = {'global_rot': 'global_rot', 'body_pose': 'body_pose_params', 'hand': 'hand_pose_params',
              'scale': 'scale_params', 'shape': 'shape_params', 'face': 'expr_params'}


def frame_people(mask_path: str, ignore: Optional[set]) -> Tuple[np.ndarray, List[int]]:
    """Label mask (H, W) and the IDs of the people to send to the model (margin rule as in Stage 2)."""
    labels = np.array(Image.open(mask_path).convert('P'))
    H, W = labels.shape
    mh, mw = int(H * MARGIN_FRAC), int(W * MARGIN_FRAC)
    inner = labels[mh:H - mh, mw:W - mw]
    ids = [int(o) for o in np.unique(labels) if o != 0 and not (ignore and int(o) in ignore)]
    ids = [o for o in ids if (inner == o).any()]
    return labels, ids


def mask_box(mask: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def keypoint_confidence(kp2d: np.ndarray, labels: np.ndarray, own: int, near_px: int = 4) -> np.ndarray:
    """(70, 2) pixels -> (70,) confidence from the Stage 1 label mask (see module doc)."""
    H, W = labels.shape
    xy = np.round(kp2d).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    conf = np.zeros(len(kp2d), dtype=np.float32)
    own_mask = (labels == own).astype(np.uint8)
    near = cv2.dilate(own_mask, np.ones((2 * near_px + 1, 2 * near_px + 1), np.uint8)) > 0
    x, y = xy[inside, 0], xy[inside, 1]
    lab = labels[y, x]
    c = np.full(len(x), 0.3, dtype=np.float32)
    c[near[y, x]] = 0.7
    c[lab == own] = 1.0
    c[(lab != 0) & (lab != own)] = 0.0
    conf[inside] = c
    return conf


class VirtualCropRunner:
    """Runs SAM-3D-Body on virtual pinhole crops of fisheye frames. cam: utils.cosilab_camera.Camera at the Stage 1
    image resolution."""

    def __init__(self, estimator, cam, device, crop_size: int = 512, dilate: float = 1.2, max_crops: int = 64):
        self.estimator, self.cam, self.device = estimator, cam, device
        self.crop_size, self.dilate, self.max_crops = crop_size, dilate, max_crops
        self.K = torch.tensor(cam.K, dtype=torch.float32, device=device)
        self.D = torch.tensor(cam.D, dtype=torch.float32, device=device)
        self.residuals: List[float] = []
        self.convention: Optional[str] = None

    def _crops(self, img_bgr: np.ndarray, labels: np.ndarray, ids: Sequence[int]):
        boxes = np.stack([mask_box(labels == o) for o in ids])
        R_v, f_v = fisheye.setup_virtual_cameras(boxes, self.cam.K, self.cam.D, self.crop_size, self.dilate)
        Rt = torch.tensor(R_v, dtype=torch.float32, device=self.device)
        ft = torch.tensor(f_v, dtype=torch.float32, device=self.device)
        rgb = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1).float()
        crops = fisheye.render_crops(rgb, Rt, ft, self.K, self.D, self.crop_size)
        own = torch.stack([torch.from_numpy(labels == o) for o in ids]).to(self.device)[:, None].float()
        masks = fisheye.render_crops(own, Rt, ft, self.K, self.D, self.crop_size, mode='nearest')
        crops = crops.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        masks = (masks[:, 0] > 0.5).byte().cpu().numpy() * 255
        out = []
        S = self.crop_size
        for j, o in enumerate(ids):
            if masks[j].any():
                box = mask_box(masks[j])
            else:  # the mask is thinner than a crop pixel: the virtual camera still looks at its box
                m = S * (1 - 1 / self.dilate) / 2
                box = np.array([m, m, S - m, S - m], dtype=np.float32)
            out.append({'id': o, 'img': crops[j], 'mask': masks[j], 'box': box, 'R_v': R_v[j], 'f_v': float(f_v[j])})
        return out

    def _to_real(self, crops: List[dict], outs: List[dict], labels_of: List[np.ndarray]) -> List[dict]:
        dev = self.device
        params = {k: torch.as_tensor(np.stack([o[v] for o in outs]), dtype=torch.float32, device=dev)
                  for k, v in PARAM_KEYS.items()}
        kp_local = torch.as_tensor(np.stack([o['pred_keypoints_3d'] for o in outs]), dtype=torch.float32, device=dev)
        cam_t = torch.as_tensor(np.stack([o['pred_cam_t'] for o in outs]), dtype=torch.float32, device=dev)
        Q = torch.as_tensor(np.stack([c['R_v'].T for c in crops]), dtype=torch.float32, device=dev)
        rot, t_real, kp_real, res, conv = rotate_body(self.estimator.model.head_pose, params, kp_local, cam_t, Q,
                                                      self.convention)
        self.convention = conv
        self.residuals.extend(res.tolist())
        kp2d = fisheye.project_torch(kp_real + t_real[:, None], self.K, self.D).cpu().numpy()
        rot, t_real = rot.cpu().numpy(), t_real.cpu().numpy()
        people = []
        for j, (c, o) in enumerate(zip(crops, outs)):
            p = dict(o)
            p['global_rot'] = rot[j].astype(np.float32)
            p['pred_cam_t'] = t_real[j].astype(np.float32)
            p['focal_length_virtual'] = p.pop('focal_length', None)
            p['R_v'] = c['R_v'].astype(np.float32)
            p['f_v'] = c['f_v']
            p['kp2d'] = kp2d[j].astype(np.float32)
            p['kp_conf'] = keypoint_confidence(kp2d[j], labels_of[j], c['id'])
            p['rot_residual'] = float(res[j])
            for k in ('pred_vertices', 'mask', 'pred_keypoints_2d', 'pred_joint_coords', 'pred_global_rots',
                      'pred_keypoints_3d'):
                p.pop(k, None)  # virtual-camera quantities; not stored
            people.append(p)
        return people

    def process(self, image_paths: Sequence[str], mask_paths: Sequence[str], ignore_ids: Sequence[Optional[set]]):
        """Same contract as notebook.utils.process_image_with_mask: (outputs per non-empty frame, IDs per non-empty
        frame, indices of empty frames)."""
        frame_crops, frame_labels, empty = [], [], []
        for i, (ip, mp) in enumerate(zip(image_paths, mask_paths)):
            labels, ids = frame_people(mp, ignore_ids[i] if ignore_ids else None)
            if not ids:
                empty.append(i)
                continue
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img.shape[:2] != labels.shape:
                raise ValueError(f'{ip}: image {img.shape[:2]} and mask {labels.shape} sizes differ')
            if (img.shape[1], img.shape[0]) != tuple(self.cam.size):
                raise ValueError(f'{ip}: image size {img.shape[1]}x{img.shape[0]} does not match the camera '
                                 f'{self.cam.size}; check --camera-scale')
            frame_crops.append(self._crops(img, labels, ids))
            frame_labels.append(labels)

        flat = [(fi, c) for fi, cs in enumerate(frame_crops) for c in cs]
        people_flat: List[dict] = []
        for s in range(0, len(flat), self.max_crops):
            chunk = flat[s:s + self.max_crops]
            crops = [c for _, c in chunk]
            K_v = fisheye.virtual_K(np.array([c['f_v'] for c in crops]), self.crop_size)
            res = self.estimator.process_frames(
                [c['img'] for c in crops], bboxes=[c['box'][None] for c in crops],
                masks=[c['mask'][None] for c in crops], id_batch=[[1]] * len(crops),
                idx_path={}, idx_dict={}, mhr_shape_scale_dict={}, occ_dict={},
                cam_int=[torch.tensor(k, dtype=torch.float32)[None] for k in K_v])
            if len(res) != len(crops) or any(len(r) != 1 for r in res):
                raise RuntimeError(f'SAM-3D-Body returned {[len(r) for r in res]} people for {len(crops)} crops')
            people_flat += self._to_real(crops, [r[0] for r in res], [frame_labels[fi] for fi, _ in chunk])

        outputs, id_batch, k = [], [], 0
        for cs in frame_crops:
            outputs.append(people_flat[k:k + len(cs)])
            id_batch.append([c['id'] for c in cs])
            k += len(cs)
        return outputs, id_batch, empty
