#!/usr/bin/env python3
"""
Utilities to sanity-check mhr.pt / raw_mhr.pt payloads.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def check_mhr(pt_path: str) -> Dict[str, Any]:
    """
    Load a raw MHR payload and print a short summary.

    Expected payload (from run_sam3d_body_raw_params.py):
      - "frames": list of dicts with "frame", "people", "obj_ids"
      - "meta": optional dict
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Missing file: {pt_path}")

    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    frames = payload.get("frames", [])
    meta = payload.get("meta", {})

    obj_ids_all = set()
    frames_with_people = 0
    for fr in frames:
        obj_ids = fr.get("obj_ids", [])
        if obj_ids:
            frames_with_people += 1
        for oid in obj_ids:
            obj_ids_all.add(int(oid))

    obj_ids_all = sorted(obj_ids_all)
    n_frames = len(frames)

    summary = {
        "path": pt_path,
        "num_frames": n_frames,
        "frames_with_people": frames_with_people,
        "num_obj_ids": len(obj_ids_all),
        "obj_ids": obj_ids_all,
        "meta": meta,
    }

    print("[MHR CHECK]")
    print(f"  path: {summary['path']}")
    print(f"  frames: {summary['num_frames']}")
    print(f"  frames_with_people: {summary['frames_with_people']}")
    print(f"  num_obj_ids: {summary['num_obj_ids']}")
    if obj_ids_all:
        print(f"  obj_ids (first 20): {obj_ids_all[:20]}")
    else:
        print("  obj_ids: []")
    if meta:
        print("  meta keys:", sorted(meta.keys()))

    return summary


def check_mask_ids(mask_dir: str, meta_json: str) -> Dict[str, Any]:
    """
    Verify that all IDs in masklets_meta.json exist in the mask PNGs.

    Args:
        mask_dir: folder containing mask PNGs (palette masks; pixel value == obj_id)
        meta_json: path to masklets_meta.json (contains expected obj_ids)
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Missing mask_dir: {mask_dir}")
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Missing meta_json: {meta_json}")

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Heuristic: expected IDs stored in meta under one of these keys
    expected_ids: Set[int] = set()
    key = "out_obj_ids"

    expected_ids = {int(x) for x in meta[key]}
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        raise FileNotFoundError(f"No .png masks found in: {mask_dir}")

    present_ids: Set[int] = set()
    for mp in mask_paths:
        mask = np.array(Image.open(mp).convert("P"))
        ids = np.unique(mask)
        # Exclude background 0
        ids = ids[ids != 0]
        present_ids.update(int(x) for x in ids.tolist())

    missing = sorted(expected_ids - present_ids)
    extra = sorted(present_ids - expected_ids)

    summary = {
        "mask_dir": mask_dir,
        "meta_json": meta_json,
        "expected_count": len(expected_ids),
        "present_count": len(present_ids),
        "missing_ids": missing,
        "extra_ids": extra,
    }

    print("[MASK ID CHECK]")
    print(f"  mask_dir: {mask_dir}")
    print(f"  meta_json: {meta_json}")
    print(f"  expected_ids: {len(expected_ids)}")
    print(f"  present_ids: {len(present_ids)}")
    if missing:
        print(f"  missing_ids: {missing}")
    else:
        print("  missing_ids: []")
    if extra:
        print(f"  extra_ids: {extra}")

    return summary


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def check_bbox_match(mask_dir: str, bbox_pkl: str, frame_idx: int = 0) -> Dict[str, Any]:
    """
    Compare bboxes derived from mask PNGs vs bboxes used for prompting (from PKL).

    Assumes PKL format:
      list of dicts per frame with keys: bboxes (Nx4), pids (N,)
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Missing mask_dir: {mask_dir}")
    if not os.path.exists(bbox_pkl):
        raise FileNotFoundError(f"Missing bbox_pkl: {bbox_pkl}")

    # Load prompt bboxes
    with open(bbox_pkl, "rb") as f:
        data = pickle.load(f)
    rec = data[frame_idx]
    prompt_bboxes = np.asarray(rec["bboxes"], dtype=np.float32)
    prompt_pids = np.asarray(rec["pids"], dtype=np.int32)
    prompt_by_id = {int(pid): prompt_bboxes[i] for i, pid in enumerate(prompt_pids)}

    # Load mask PNG for the same frame
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        raise FileNotFoundError(f"No .png masks found in: {mask_dir}")
    if frame_idx < 0 or frame_idx >= len(mask_paths):
        raise IndexError(f"frame_idx {frame_idx} out of range (0..{len(mask_paths)-1})")
    mask = np.array(Image.open(mask_paths[frame_idx]).convert("P"))

    mask_bboxes: Dict[int, np.ndarray] = {}
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        obj_id = int(obj_id)
        mask_binary = (mask == obj_id).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask_binary)
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        mask_bboxes[obj_id] = np.array([x, y, x + w, y + h], dtype=np.float32)

    common_ids = sorted(set(prompt_by_id.keys()) & set(mask_bboxes.keys()))
    missing_in_mask = sorted(set(prompt_by_id.keys()) - set(mask_bboxes.keys()))
    missing_in_prompt = sorted(set(mask_bboxes.keys()) - set(prompt_by_id.keys()))

    rows: List[Tuple[int, float]] = []
    for oid in common_ids:
        iou = _bbox_iou(prompt_by_id[oid], mask_bboxes[oid])
        rows.append((oid, iou))

    summary = {
        "mask_dir": mask_dir,
        "bbox_pkl": bbox_pkl,
        "frame_idx": frame_idx,
        "num_prompt_ids": len(prompt_by_id),
        "num_mask_ids": len(mask_bboxes),
        "num_common_ids": len(common_ids),
        "missing_in_mask": missing_in_mask,
        "missing_in_prompt": missing_in_prompt,
        "ious": rows,
    }

    print("[BBOX MATCH CHECK]")
    print(f"  frame_idx: {frame_idx}")
    print(f"  prompt ids: {summary['num_prompt_ids']}")
    print(f"  mask ids: {summary['num_mask_ids']}")
    print(f"  common ids: {summary['num_common_ids']}")
    if missing_in_mask:
        print(f"  missing_in_mask: {missing_in_mask}")
    if missing_in_prompt:
        print(f"  missing_in_prompt: {missing_in_prompt}")
    if rows:
        worst = sorted(rows, key=lambda x: x[1])[:10]
        print("  worst IoU (up to 10):", worst)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a raw MHR .pt file")
    # parser.add_argument("--mhr-pt", help="Path to raw MHR .pt (e.g., raw_mhr.pt)")
    parser.add_argument("--mask-dir", default=None, help="Path to masks/ folder (PNG masks)")
    parser.add_argument("--meta-json", default=None, help="Path to masklets_meta.json")
    parser.add_argument("--bbox-pkl", default=None, help="Path to bbox/kps pkl used for prompting")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index for bbox comparison")
    args = parser.parse_args()

    # check_mhr(args.mhr_pt)
    if args.mask_dir and args.meta_json:
        check_mask_ids(args.mask_dir, args.meta_json)
    if args.mask_dir and args.bbox_pkl:
        check_bbox_match(args.mask_dir, args.bbox_pkl, frame_idx=args.frame_idx)


if __name__ == "__main__":
    main()
