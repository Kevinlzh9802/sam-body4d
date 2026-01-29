from PIL import Image
from PIL import ImageDraw
import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def draw_point_marker(image: Image.Image, x: int, y: int, point_type: str) -> Image.Image:
    """
    Draw a circular marker with soft color fill:
        - Positive:  light green fill + white border + white "+"
        - Negative:  light red   fill + white border + white "-"

    Marker size auto-scales with image size.

    Args:
        image: PIL Image(RGB)
        x, y: coordinates (int)
        point_type: "positive" or "negative"

    Returns:
        PIL Image with marker drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image size
    w, h = img.size

    # ===== Auto-scale marker size =====
    base = min(w, h)
    radius = max(6, int(base * 0.015))        # Circle radius ~1.5% of the shorter side
    line_w = max(2, radius // 4)              # Stroke width for +/-
    border_w = max(2, radius // 5)            # White border thickness

    # Clamp coordinates
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    # ===== Color settings =====
    if point_type.lower() == "positive":
        fill_color = (180, 255, 180)   # Light green
    else:
        fill_color = (255, 180, 180)   # Light red

    border_color = (255, 255, 255)     # White
    sign_color = (255, 255, 255)       # White

    # ===== Draw circle (fill + white border) =====
    bbox = [x - radius, y - radius, x + radius, y + radius]

    # Fill circle
    draw.ellipse(bbox, fill=fill_color)

    # Overlaid white border
    draw.ellipse(bbox, outline=border_color, width=border_w)

    # ===== Draw plus/minus sign (white stroke) =====
    # Horizontal line
    draw.line(
        (x - radius + 3, y, x + radius - 3, y),
        fill=sign_color,
        width=line_w,
    )

    # Vertical line (only for "positive")
    if point_type.lower() == "positive":
        draw.line(
            (x, y - radius + 3, x, y + radius - 3),
            fill=sign_color,
            width=line_w,
        )

    return img

def load_bbox_kp(bbox_kp_folder: str, folder_name: str):
    """
    Load bboxes and kps from a folder.
    """
    bboxes_kps_data = None
    if len(bbox_kp_folder):
        keypoint_path = os.path.join(bbox_kp_folder, f"{folder_name}.pkl")
        with open(keypoint_path, "rb") as kp_f:
            bboxes_kps_data = pickle.load(kp_f)
    return bboxes_kps_data

def _frame_tensor_to_uint8(inference_state, frame_idx: int, out_h: int, out_w: int) -> np.ndarray:
    """
    Convert the internal SAM3 frame tensor to a uint8 RGB image, resized to (out_h, out_w).
    Mirrors the conversion logic used in `mask_generation`.
    """
    img = inference_state["images"][frame_idx].detach().float().cpu()  # (3, H, W) in [-1, 1]
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    img = F.interpolate(
        img.unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    img = img.permute(1, 2, 0)  # (H, W, 3)
    return (img.numpy() * 255).astype("uint8")


def save_sanity_first_frame_overlay(
    *,
    inference_state,
    bboxes_kps_data,
    output_dir: str,
    frame_idx: int = 0,
    filename: str = "sanity_first_frame_bbox_kps.jpg",
) -> str:
    """
    Sanity check: overlay bboxes + keypoints from PKL on the first frame that is fed to SAM3.
    Saves an image under `output_dir` and returns the saved path.

    Expected PKL entry format (per frame):
      - bboxes: (N, 4) [x1,y1,x2,y2] in pixels
      - kps:    (N, 10, 3) [x,y,label] in pixels; label == -2 means invalid
      - pids:   (N,) person ids (used for labeling)
    """
    os.makedirs(output_dir, exist_ok=True)
    if bboxes_kps_data is None:
        raise ValueError("bboxes_kps_data is None; cannot run sanity overlay.")

    out_h = int(inference_state["video_height"])
    out_w = int(inference_state["video_width"])
    img = _frame_tensor_to_uint8(inference_state, frame_idx=frame_idx, out_h=out_h, out_w=out_w)
    vis = img.copy()

    frame_rec = bboxes_kps_data[frame_idx]
    bboxes = frame_rec.get("bboxes", None)
    kps = frame_rec.get("kps", None)
    pids = frame_rec.get("pids", None)

    if bboxes is None or kps is None or pids is None:
        raise ValueError(f"Missing keys in PKL frame {frame_idx}: expected bboxes/kps/pids.")

    bboxes = np.asarray(bboxes)
    kps = np.asarray(kps)
    pids = np.asarray(pids)

    # Draw per-person overlays
    for i in range(len(pids)):
        pid = int(pids[i])
        color = (
            int((37 * pid) % 255),
            int((17 * pid) % 255),
            int((97 * pid) % 255),
        )

        # bbox
        x1, y1, x2, y2 = bboxes[i]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            vis,
            f"id={pid}",
            (int(x1), max(0, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # keypoints
        kp_i = kps[i]  # (10, 3)
        for j in range(kp_i.shape[0]):
            x, y, label = kp_i[j]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if float(label) < 0:  # -2 invalid
                continue
            cv2.circle(vis, (int(x), int(y)), 3, color, -1)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Sanity overlay saved to: {out_path}")
    return out_path