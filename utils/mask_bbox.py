"""
Utility to extract per-person bounding boxes from palette mask PNGs.

Each mask PNG is P-mode with pixel value = obj_id (0 = background).
The bbox is computed from the convex hull of each person's mask pixels.
Output format: [x, y, w, h] (top-left corner + width/height).
"""

import glob
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def bbox_from_mask_convex_hull(binary_mask: np.ndarray) -> Optional[List[int]]:
    """
    Compute [x, y, w, h] bounding box from the convex hull of a binary mask.
    Returns None if the mask is empty.
    """
    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        return None
    hull = cv2.convexHull(coords)
    x, y, w, h = cv2.boundingRect(hull)
    return [int(x), int(y), int(w), int(h)]


def _find_segment_mapping(
    frame_idx: int,
    segment_id_mappings: List[Dict],
) -> Optional[Dict[int, int]]:
    """Return the consecutive->actual mapping for the segment that contains *frame_idx*."""
    for seg in segment_id_mappings:
        if int(seg["frame_start"]) <= frame_idx <= int(seg["frame_end"]):
            return {int(k): int(v) for k, v in seg["consecutive_to_actual"].items()}
    return None


def extract_bboxes_from_masks(
    masks_dir: str,
    consecutive_to_actual: Optional[Dict[int, int]] = None,
    segment_id_mappings: Optional[List[Dict]] = None,
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Walk all mask PNGs in *masks_dir* and return::

        {
            "annotations": {
                "<image_id>": {
                    "bbox": {
                        "<person_id>": [x, y, w, h],
                        ...
                    }
                },
                ...
            }
        }

    Image IDs are the integer frame index (from the filename,
    e.g. ``00000001.png`` -> ``"1"``).

    When *segment_id_mappings* is provided (list of per-segment dicts with
    ``frame_start``, ``frame_end``, ``consecutive_to_actual``), the frame
    index is used to look up the correct per-segment mapping.

    Otherwise, *consecutive_to_actual* (a single global dict) is used.
    If neither is given, raw pixel values are used as person IDs.
    """
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    annotations: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    for mask_path in mask_paths:
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        frame_idx = int(basename)
        image_id = str(frame_idx)

        # Determine the ID mapping for this frame
        if segment_id_mappings:
            c2a = _find_segment_mapping(frame_idx, segment_id_mappings)
        else:
            c2a = consecutive_to_actual

        mask = np.array(Image.open(mask_path).convert("P"))
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # skip background

        bbox_dict: Dict[str, List[int]] = {}
        for obj_id in obj_ids:
            binary = ((mask == obj_id) * 255).astype(np.uint8)
            bb = bbox_from_mask_convex_hull(binary)
            if bb is not None:
                if c2a is not None:
                    actual_id = c2a.get(int(obj_id), int(obj_id))
                else:
                    actual_id = int(obj_id)
                bbox_dict[str(actual_id)] = bb

        if bbox_dict:
            annotations[image_id] = {"bbox": bbox_dict}

    return {"annotations": annotations}
