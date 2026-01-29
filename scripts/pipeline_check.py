#!/usr/bin/env python3
"""
Utilities to sanity-check mhr.pt / raw_mhr.pt payloads.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import torch


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check a raw MHR .pt file")
    parser.add_argument("mhr_pt", help="Path to raw MHR .pt (e.g., raw_mhr.pt)")
    args = parser.parse_args()

    check_mhr(args.mhr_pt)


if __name__ == "__main__":
    main()
