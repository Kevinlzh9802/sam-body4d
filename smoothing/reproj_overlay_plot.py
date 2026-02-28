"""
Overlay projected 3D meshes / keypoints on original 2D images + detected masks.

Produces diagnostic images at a configurable frame interval, together with
per-frame reprojection error saved as JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .obs_kps import ObsKps, build_obj_id_to_bbox_idx, get_obs_for_person, load_bboxes_kps_pkl

# --------------------------------------------------------------------------- #
#  Colour palette (RGB, 0-255) — distinct colours for up to 20 people.        #
# --------------------------------------------------------------------------- #
_PAL = [
    (230, 25, 75),  (60, 180, 75),  (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (66, 212, 244),  (240, 50, 230),
    (191, 239, 69), (250, 190, 212),(70, 153, 144),  (220, 190, 255),
    (154, 99, 36),  (255, 250, 200),(128, 0, 0),     (170, 255, 195),
    (128, 128, 0),  (255, 216, 177),(0, 0, 117),     (169, 169, 169),
]


def _bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


# --------------------------------------------------------------------------- #
#  Projection / rasterisation helpers                                          #
# --------------------------------------------------------------------------- #

def _project(K: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pinhole projection.  K (3,3), pts (..., 3) → (..., 2) pixels."""
    z = np.clip(pts[..., 2], 1e-6, None)
    u = K[0, 0] * (pts[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (pts[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def _rasterize_mesh(
    uv: np.ndarray,
    faces: np.ndarray,
    z: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Return an (H, W) uint8 mask (0/255) of the projected mesh silhouette."""
    mask = np.zeros((h, w), dtype=np.uint8)
    ok = z > 0.01
    keep = ok[faces].all(axis=1)
    if not keep.any():
        return mask
    tris = np.clip(uv[faces[keep]].astype(np.int32), -5000, max(h, w) + 5000)
    for i in range(len(tris)):
        cv2.fillConvexPoly(mask, tris[i], 255)
    return mask


# --------------------------------------------------------------------------- #
#  Reprojection-error computation                                              #
# --------------------------------------------------------------------------- #

def _reproj_errors(
    K: np.ndarray,
    kps_cam: np.ndarray,       # (Kpts, 3)
    obs_kp_idx: np.ndarray,    # (M,) int – mhr70 indices
    obs_xy: np.ndarray,        # (M, 2) stored pixel coords
    obs_scale: float,
) -> Dict[str, Any]:
    proj = _project(K, kps_cam)
    out: List[Dict[str, Any]] = []
    for i, ki in enumerate(obs_kp_idx):
        p = proj[int(ki)]
        o = obs_xy[i] * obs_scale
        err = float(np.linalg.norm(p - o))
        out.append({
            "kp_idx": int(ki),
            "pred_px": [round(float(p[0]), 2), round(float(p[1]), 2)],
            "obs_px": [round(float(o[0]), 2), round(float(o[1]), 2)],
            "error_px": round(err, 2),
        })
    mean = float(np.mean([e["error_px"] for e in out])) if out else 0.0
    return {"keypoints": out, "mean_error_px": round(mean, 2)}


def _pick_obs_scale(
    K: np.ndarray,
    kps_cam: np.ndarray,
    obs_kp_idx: np.ndarray,
    obs_xy: np.ndarray,
    candidates: Tuple[float, ...] = (1.0, 0.5, 2.0),
) -> float:
    """Choose the obs_scale that minimises initial mean reprojection error."""
    proj = _project(K, kps_cam)
    best_s, best_err = float(candidates[0]), float("inf")
    for s in candidates:
        errs = [float(np.linalg.norm(proj[int(ki)] - obs_xy[i] * s))
                for i, ki in enumerate(obs_kp_idx)]
        m = float(np.mean(errs)) if errs else float("inf")
        if m < best_err:
            best_err, best_s = m, float(s)
    return best_s


# --------------------------------------------------------------------------- #
#  Single-frame overlay rendering                                              #
# --------------------------------------------------------------------------- #

def render_overlay_frame(
    *,
    image: np.ndarray,                             # (H, W, 3) RGB uint8
    mask: Optional[np.ndarray],                    # (H, W) palette indices
    K: np.ndarray,                                 # (3, 3)
    faces: np.ndarray,                             # (F, 3)
    persons: Dict[int, Dict[str, np.ndarray]],
    # persons = {obj_id: {"verts_cam": (V,3), "kps_cam": (70,3),
    #            optional "obs_kp_idx": (M,), "obs_xy": (M,2)}}
    frame_name: str,
    obs_scale: float = 1.0,
    id_labels: Optional[Dict[int, str]] = None,
    oid_to_mask_id: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compose a single diagnostic overlay.  Returns (canvas_bgr, errors)."""
    h, w = image.shape[:2]
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_errors: Dict[str, Any] = {}

    # ---- 1. detected-mask contours (thick, per-person colour) ------------- #
    if mask is not None:
        for uid in np.unique(mask):
            if uid == 0:
                continue
            ci = (int(uid) - 1) % len(_PAL)
            bm = (mask == uid).astype(np.uint8)
            cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, _bgr(_PAL[ci]), 2)

    # ---- 2. mesh silhouettes (all persons → one overlay, single blend) ---- #
    overlay = np.zeros_like(canvas)
    mesh_masks: Dict[int, np.ndarray] = {}
    for pidx, (oid, pd) in enumerate(sorted(persons.items())):
        ci = pidx % len(_PAL)
        col = _bgr(_PAL[ci])
        v = pd["verts_cam"]
        uv = _project(K, v)
        msk = _rasterize_mesh(uv, faces, v[:, 2], h, w)
        overlay[msk > 0] = col
        mesh_masks[oid] = msk
    cv2.addWeighted(canvas, 1.0, overlay, 0.25, 0, dst=canvas)

    # mesh contours (thin)
    for pidx, (oid, msk) in enumerate(sorted(mesh_masks.items())):
        ci = pidx % len(_PAL)
        mc, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, mc, -1, _bgr(_PAL[ci]), 1)

    # ---- 2b. mask-vs-mesh silhouette IoU per person ---------------------- #
    if mask is not None:
        for oid, mesh_msk in mesh_masks.items():
            mask_pixel_id = oid_to_mask_id.get(oid, oid) if oid_to_mask_id else oid
            det_bin = (mask == mask_pixel_id).astype(np.uint8)
            mesh_bin = (mesh_msk > 0).astype(np.uint8)
            inter = int((det_bin & mesh_bin).sum())
            union = int((det_bin | mesh_bin).sum())
            iou = inter / union if union > 0 else 0.0
            oid_key = str(oid)
            if oid_key not in frame_errors:
                frame_errors[oid_key] = {}
            frame_errors[oid_key]["mask_mesh_iou"] = round(iou, 4)
            frame_errors[oid_key]["mask_area_px"] = int(det_bin.sum())
            frame_errors[oid_key]["mesh_area_px"] = int(mesh_bin.sum())

    # ---- 3. projected keypoints + observed keypoints + error lines -------- #
    for pidx, (oid, pd) in enumerate(sorted(persons.items())):
        ci = pidx % len(_PAL)
        col = _bgr(_PAL[ci])

        kps = pd.get("kps_cam")
        if kps is not None:
            uv_kps = _project(K, kps)
            ok = kps[:, 2] > 0.01
            for ki in range(len(uv_kps)):
                if ok[ki]:
                    pt = (int(uv_kps[ki, 0]), int(uv_kps[ki, 1]))
                    cv2.circle(canvas, pt, 4, col, -1)
                    cv2.circle(canvas, pt, 4, (255, 255, 255), 1)

        obs_idx = pd.get("obs_kp_idx")
        obs_xy = pd.get("obs_xy")
        if (
            obs_idx is not None
            and obs_xy is not None
            and len(obs_idx) > 0
            and kps is not None
        ):
            uv_kps = _project(K, kps)
            for i, ki in enumerate(obs_idx):
                op = (obs_xy[i] * obs_scale).astype(int)
                pp = uv_kps[int(ki)].astype(int)
                cv2.drawMarker(
                    canvas, tuple(op), (255, 255, 255),
                    cv2.MARKER_TILTED_CROSS, 10, 2,
                )
                cv2.line(canvas, tuple(pp), tuple(op), (0, 0, 255), 1, cv2.LINE_AA)
            frame_errors[str(oid)] = _reproj_errors(
                K, kps, obs_idx, obs_xy, obs_scale,
            )

    # ---- 3b. person ID labels ---------------------------------------------- #
    if id_labels:
        for pidx, (oid, pd) in enumerate(sorted(persons.items())):
            label = id_labels.get(oid)
            if not label:
                continue
            ci = pidx % len(_PAL)
            col = _bgr(_PAL[ci])
            v = pd["verts_cam"]
            uv = _project(K, v)
            ok = v[:, 2] > 0.01
            if not ok.any():
                continue
            uv_ok = uv[ok]
            lx = max(0, int(uv_ok[:, 0].min()))
            ly = max(20, int(uv_ok[:, 1].min()) - 8)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 4), (0, 0, 0), -1)
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # ---- 4. header text --------------------------------------------------- #
    errs = [e["error_px"] for ed in frame_errors.values() for e in ed.get("keypoints", [])]
    ious = [ed["mask_mesh_iou"] for ed in frame_errors.values() if "mask_mesh_iou" in ed]
    parts = [f"Frame {frame_name}"]
    if errs:
        parts.append(f"reproj err = {float(np.mean(errs)):.1f} px")
    if ious:
        parts.append(f"mask IoU = {float(np.mean(ious)):.2f}")
    txt = "  |  ".join(parts)
    cv2.putText(canvas, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 5. legend -------------------------------------------------------- #
    y0 = 50
    cv2.circle(canvas, (20, y0), 4, (200, 200, 200), -1)
    cv2.putText(canvas, "Projected kp", (32, y0 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.drawMarker(canvas, (20, y0 + 22), (255, 255, 255),
                   cv2.MARKER_TILTED_CROSS, 8, 2)
    cv2.putText(canvas, "Observed kp", (32, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.line(canvas, (12, y0 + 44), (28, y0 + 44), (0, 0, 255), 1)
    cv2.putText(canvas, "Reproj error", (32, y0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return canvas, frame_errors


# --------------------------------------------------------------------------- #
#  Batch entry-point (called from smooth_mhr_and_export_meshes.py)             #
# --------------------------------------------------------------------------- #

def plot_reproj_overlays_from_stage3(
    *,
    verts: torch.Tensor,               # (T*N, V, 3) local, camera-axis-flipped
    keypoints3d_local: torch.Tensor,   # (T*N, K_kps, 3)
    pred_cam_t: torch.Tensor,          # (T*N, 3)
    faces: np.ndarray,                 # (F, 3)
    K: np.ndarray,                     # (3, 3) scaled intrinsics
    images_dir: str,
    masks_dir: Optional[str],
    T: int,
    N: int,
    obj_ids_all: List[int],
    frame_obj_ids_slots: List[List[int]],
    frame_names: List[str],
    output_dir: str,
    frame_interval: int = 100,
    bboxes_kps_data: Optional[Any] = None,
    obj_id_to_bbox_idx: Optional[Dict[int, int]] = None,
    obs_scale: Optional[float] = None,
    obs_scale_candidates: Tuple[float, ...] = (1.0, 0.5, 2.0),
    segment_id_mappings: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Plot reprojection overlay images every *frame_interval* frames."""
    os.makedirs(output_dir, exist_ok=True)

    K_body = min(keypoints3d_local.shape[1], 70)

    # --- determine obs_scale once from first usable observation ------------ #
    if obs_scale is None and bboxes_kps_data is not None and obj_id_to_bbox_idx is not None:
        for ti in range(0, T, frame_interval):
            found = False
            for si, oid in enumerate(obj_ids_all):
                if frame_obj_ids_slots[ti][si] != oid:
                    continue
                obs = get_obs_for_person(
                    bboxes_kps_data, ti, oid, obj_id_to_bbox_idx, device="cpu",
                )
                if obs is None or len(obs.kp_idx) == 0:
                    continue
                bi = ti * N + si
                kps_cam = (
                    keypoints3d_local[bi, :K_body].detach().cpu() + pred_cam_t[bi].detach().cpu()
                ).numpy()
                obs_scale = _pick_obs_scale(
                    K, kps_cam,
                    obs.kp_idx.cpu().numpy(),
                    obs.xy.cpu().numpy(),
                    obs_scale_candidates,
                )
                found = True
                break
            if found:
                break
    if obs_scale is None:
        obs_scale = 1.0
    print(f"[INFO] Reproj overlay: obs_scale={obs_scale}")

    # --- iterate over sampled frames --------------------------------------- #
    all_errors: Dict[str, Any] = {}
    frames_to_plot = list(range(0, T, frame_interval))

    for ti in tqdm(frames_to_plot, desc="Reproj overlay"):
        fname = frame_names[ti]

        # load image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = os.path.join(images_dir, fname + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        image = np.array(Image.open(img_path).convert("RGB"))

        # load mask
        mask_arr: Optional[np.ndarray] = None
        if masks_dir is not None:
            for ext in (".png", ".bmp"):
                mp = os.path.join(masks_dir, fname + ext)
                if os.path.exists(mp):
                    mask_pil = Image.open(mp)
                    # palette PNGs: np.array gives (H,W) uint8 of palette indices
                    mask_arr = np.array(mask_pil)
                    if mask_arr.ndim == 3:
                        # fallback: convert to grayscale
                        mask_arr = np.array(mask_pil.convert("L"))
                    break

        # gather per-person data
        persons: Dict[int, Dict[str, np.ndarray]] = {}
        for si, oid in enumerate(obj_ids_all):
            if frame_obj_ids_slots[ti][si] != oid:
                continue
            bi = ti * N + si
            cam_t_bi = pred_cam_t[bi].detach().cpu()
            v_cam = (verts[bi].cpu() + cam_t_bi).numpy()
            kps_cam = (
                keypoints3d_local[bi, :K_body].detach().cpu() + cam_t_bi
            ).numpy()
            pdata: Dict[str, Any] = {"verts_cam": v_cam, "kps_cam": kps_cam}
            if bboxes_kps_data is not None and obj_id_to_bbox_idx is not None:
                obs = get_obs_for_person(
                    bboxes_kps_data, ti, oid, obj_id_to_bbox_idx, device="cpu",
                )
                if obs is not None and len(obs.kp_idx) > 0:
                    pdata["obs_kp_idx"] = obs.kp_idx.cpu().numpy()
                    pdata["obs_xy"] = obs.xy.cpu().numpy()
            persons[oid] = pdata

        if not persons:
            continue

        # Build per-person ID labels and mask-pixel-ID mapping from segment mappings
        frame_id_labels: Optional[Dict[int, str]] = None
        oid_to_mask_id: Optional[Dict[int, int]] = None
        if segment_id_mappings:
            frame_idx_int = int(fname)
            a2c: Dict[int, int] = {}
            for seg in segment_id_mappings:
                fs = seg.get("frame_start", 0)
                fe = seg.get("frame_end", 999_999_999)
                if fs <= frame_idx_int <= fe:
                    c2a = seg.get("consecutive_to_actual", {})
                    a2c = {int(v): int(k) for k, v in c2a.items()}
                    break
            frame_id_labels = {oid: f"T:{a2c.get(oid, '?')} R:{oid}" for oid in persons}
            oid_to_mask_id = a2c  # actual PID → consecutive (mask pixel) ID

        canvas_bgr, ferrs = render_overlay_frame(
            image=image,
            mask=mask_arr,
            K=K,
            faces=faces,
            persons=persons,
            frame_name=fname,
            obs_scale=obs_scale,
            id_labels=frame_id_labels,
            oid_to_mask_id=oid_to_mask_id,
        )
        out_path = os.path.join(output_dir, f"{fname}.jpg")
        cv2.imwrite(out_path, canvas_bgr)
        all_errors[fname] = ferrs

    # --- save errors JSON -------------------------------------------------- #
    err_path = os.path.join(output_dir, "reproj_errors.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2)
    print(f"[INFO] Reproj overlay: saved {len(all_errors)} frames to {output_dir}")
