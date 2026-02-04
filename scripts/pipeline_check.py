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
    # plot the bboxes and masks on image
    img = cv2.imread("./experiments/00000000.jpg")
    for oid in common_ids:
        bbox = prompt_by_id[oid]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, str(oid), (int(bbox[0]), int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        bbox = mask_bboxes[oid]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, str(oid), (int(bbox[0]), int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("bbox and mask", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

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


def check_id_matching_stage1_stage2(
    mask_dir: str,
    meta_json: str,
    raw_mhr_pt: str,
) -> Dict[str, Any]:
    """
    Compare IDs across stage1 (masklets_meta.json) and stage2 (raw_mhr.pt).
    
    Note: Stage 1 stores consecutive IDs (1, 2, 3, ...) in out_obj_ids and masks.
    Stage 2 converts these to actual PIDs using the id_mapping.json.
    This function uses the mapping to compare properly.
    """
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Missing meta_json: {meta_json}")
    if not os.path.exists(raw_mhr_pt):
        raise FileNotFoundError(f"Missing raw_mhr_pt: {raw_mhr_pt}")

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stage1_consecutive_ids = {int(x) for x in meta.get("out_obj_ids", [])}
    
    # Load ID mapping if available (consecutive -> actual)
    consecutive_to_actual: Dict[int, int] = {}
    # Try to load from masklets_meta.json first
    if "consecutive_to_actual" in meta:
        consecutive_to_actual = {int(k): int(v) for k, v in meta["consecutive_to_actual"].items()}
    else:
        # Try to load from id_mapping.json in the same directory
        id_mapping_path = os.path.join(os.path.dirname(meta_json), "id_mapping.json")
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                id_mapping = json.load(f)
            consecutive_to_actual = {int(k): int(v) for k, v in id_mapping.get("consecutive_to_actual", {}).items()}
    
    # Convert stage1 consecutive IDs to actual PIDs for comparison
    if consecutive_to_actual:
        stage1_actual_ids = {consecutive_to_actual.get(cid, cid) for cid in stage1_consecutive_ids}
        print(f"[INFO] Using ID mapping to convert {len(stage1_consecutive_ids)} consecutive IDs to actual PIDs")
    else:
        stage1_actual_ids = stage1_consecutive_ids
        print(f"[WARN] No ID mapping found; comparing IDs directly (may be inaccurate)")

    payload = torch.load(raw_mhr_pt, map_location="cpu", weights_only=False)
    frames = payload.get("frames", [])
    stage2_ids: Set[int] = set()
    for fr in frames:
        for oid in fr.get("obj_ids", []):
            stage2_ids.add(int(oid))

    missing_in_stage2 = sorted(stage1_actual_ids - stage2_ids)
    extra_in_stage2 = sorted(stage2_ids - stage1_actual_ids)

    summary = {
        "stage1_consecutive_ids": sorted(stage1_consecutive_ids),
        "stage1_actual_ids": sorted(stage1_actual_ids),
        "stage2_ids": sorted(stage2_ids),
        "missing_in_stage2": missing_in_stage2,
        "extra_in_stage2": extra_in_stage2,
        "id_mapping_used": bool(consecutive_to_actual),
    }

    print("[STAGE1 vs STAGE2 ID CHECK]")
    print(f"  stage1 consecutive IDs: {len(stage1_consecutive_ids)} -> {sorted(stage1_consecutive_ids)[:10]}{'...' if len(stage1_consecutive_ids) > 10 else ''}")
    print(f"  stage1 actual PIDs:     {len(stage1_actual_ids)} -> {sorted(stage1_actual_ids)[:10]}{'...' if len(stage1_actual_ids) > 10 else ''}")
    print(f"  stage2 IDs:             {len(stage2_ids)} -> {sorted(stage2_ids)[:10]}{'...' if len(stage2_ids) > 10 else ''}")
    if missing_in_stage2:
        print(f"  missing_in_stage2: {missing_in_stage2}")
    else:
        print("  missing_in_stage2: [] (all stage1 IDs found in stage2)")
    if extra_in_stage2:
        print(f"  extra_in_stage2: {extra_in_stage2}")
    else:
        print("  extra_in_stage2: [] (no unexpected IDs in stage2)")

    return summary


def check_id_matching_by_iou(
    mask_dir: str,
    bbox_pkl: str,
    frame_idx: int = 0,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Build an ID mapping between prompt bboxes and mask-derived bboxes using IoU matching.
    Useful when IDs exist but ordering differs.
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Missing mask_dir: {mask_dir}")
    if not os.path.exists(bbox_pkl):
        raise FileNotFoundError(f"Missing bbox_pkl: {bbox_pkl}")

    with open(bbox_pkl, "rb") as f:
        data = pickle.load(f)
    rec = data[frame_idx]
    prompt_bboxes = np.asarray(rec["bboxes"], dtype=np.float32)
    prompt_pids = np.asarray(rec["pids"], dtype=np.int32)

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

    # IoU matching: for each prompt id, find best mask id
    matches: List[Tuple[int, int, float]] = []
    for i, pid in enumerate(prompt_pids):
        best_id = None
        best_iou = -1.0
        for mid, mb in mask_bboxes.items():
            iou = _bbox_iou(prompt_bboxes[i], mb)
            if iou > best_iou:
                best_iou = iou
                best_id = mid
        matches.append((int(pid), int(best_id) if best_id is not None else -1, float(best_iou)))

    low_iou = [m for m in matches if m[2] < iou_thresh]
    summary = {
        "frame_idx": frame_idx,
        "num_prompt_ids": len(prompt_pids),
        "num_mask_ids": len(mask_bboxes),
        "matches": matches,
        "low_iou": low_iou,
    }

    print("[IOU ID MATCH CHECK]")
    print(f"  frame_idx: {frame_idx}")
    print(f"  prompt_ids: {len(prompt_pids)}")
    print(f"  mask_ids: {len(mask_bboxes)}")
    if low_iou:
        print(f"  matches below iou<{iou_thresh}: {low_iou[:10]}")
    else:
        print("  all matches above threshold")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a raw MHR .pt file")
    parser.add_argument("--mhr-pt", help="Path to raw MHR .pt (e.g., raw_mhr.pt)")
    parser.add_argument("--mask-dir", default=None, help="Path to masks/ folder (PNG masks)")
    parser.add_argument("--meta-json", default=None, help="Path to masklets_meta.json")
    parser.add_argument("--bbox-pkl", default=None, help="Path to bbox/kps pkl used for prompting")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index for bbox comparison")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for ID matching")
    args = parser.parse_args()

    check_mhr(args.mhr_pt)
    # if args.mask_dir and args.meta_json:
    #     check_mask_ids(args.mask_dir, args.meta_json)
    # if args.mask_dir and args.bbox_pkl:
    #     check_bbox_match(args.mask_dir, args.bbox_pkl, frame_idx=args.frame_idx)
    #     check_id_matching_by_iou(args.mask_dir, args.bbox_pkl, frame_idx=args.frame_idx, iou_thresh=args.iou_thresh)
    if args.meta_json and args.mhr_pt:
        check_id_matching_stage1_stage2(args.mask_dir or "", args.meta_json, args.mhr_pt)


if __name__ == "__main__":
    main()
