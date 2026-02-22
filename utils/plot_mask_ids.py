"""
Visualize masks with tracking-ID / real-ID labels overlaid on each person.

Produces one annotated JPEG per sampled frame into ``<output_dir>/mask_ids/``.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

_COLORS_BGR = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67),
    (49, 130, 245), (180, 30, 145), (244, 212, 66), (230, 50, 240),
    (69, 239, 191), (212, 190, 250), (144, 153, 70), (255, 190, 220),
]


def _find_mapping(
    frame_idx: int,
    segment_id_mappings: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], str]:
    for seg in segment_id_mappings:
        fs = int(seg["frame_start"])
        fe = int(seg["frame_end"])
        if fs <= frame_idx <= fe:
            c2a = seg.get("consecutive_to_actual", {})
            c2a_int = {int(k): int(v) for k, v in c2a.items()}
            return c2a_int, seg.get("segment_key", "?")
    return {}, "?"


def plot_mask_id_visualization(
    masks_dir: str,
    images_dir: str,
    segment_id_mappings: List[Dict[str, Any]],
    output_dir: str,
    frame_interval: int = 200,
) -> str:
    """Plot masks with tracking ID and real ID labels every *frame_interval* frames.

    Saves annotated frames to ``<output_dir>/mask_ids/``.
    """
    mask_id_dir = os.path.join(output_dir, "mask_ids")
    os.makedirs(mask_id_dir, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not mask_paths:
        print("[WARN] No mask PNGs found for ID visualization.")
        return mask_id_dir

    num_plotted = 0
    for mask_path in mask_paths:
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        frame_idx = int(basename)

        if frame_idx % frame_interval != 0:
            continue

        mask_arr = np.array(Image.open(mask_path))
        obj_ids = np.unique(mask_arr)
        obj_ids = obj_ids[obj_ids != 0]

        if len(obj_ids) == 0:
            continue

        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = os.path.join(images_dir, basename + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break

        if img is None:
            h, w = mask_arr.shape[:2]
            img = np.zeros((h, w, 3), dtype=np.uint8)

        canvas = img.copy()
        c2a, seg_key = _find_mapping(frame_idx, segment_id_mappings)

        for i, oid in enumerate(sorted(obj_ids)):
            color = _COLORS_BGR[i % len(_COLORS_BGR)]
            binary = (mask_arr == oid).astype(np.uint8) * 255

            overlay = np.zeros_like(canvas)
            overlay[binary > 0] = color
            cv2.addWeighted(canvas, 1.0, overlay, 0.4, 0, dst=canvas)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)

            M = cv2.moments(binary)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                coords = cv2.findNonZero(binary)
                if coords is not None:
                    bx, by, bw, bh = cv2.boundingRect(coords)
                    cx, cy = bx + bw // 2, by + bh // 2
                else:
                    continue

            tracking_id = int(oid)
            real_id = c2a.get(tracking_id, tracking_id)
            label = f"T:{tracking_id} R:{real_id}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (cx - 2, cy - th - 6), (cx + tw + 2, cy + 4), (0, 0, 0), -1)
            cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        header = f"Frame {frame_idx} | Seg: {seg_key}"
        cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(mask_id_dir, f"{basename}.jpg")
        cv2.imwrite(out_path, canvas)
        num_plotted += 1

    print(f"[INFO] Saved {num_plotted} mask ID visualization frames to: {mask_id_dir}")
    return mask_id_dir
