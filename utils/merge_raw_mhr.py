"""
Merge partial ``raw_mhr_part_*.pt`` files produced by
``run_sam3d_body_raw_params.py --frame-start/--frame-end`` into a single
``raw_mhr.pt``.

The individual part files are always kept on disk after merging.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

NUM_EXPECTED_PARTS = 3


def discover_part_files(input_dir: str) -> List[str]:
    """Return sorted list of ``raw_mhr_part_*.pt`` paths inside *input_dir*/raw_mhr_parts/."""
    parts_dir = os.path.join(input_dir, "raw_mhr_parts")
    return sorted(glob.glob(os.path.join(parts_dir, "raw_mhr_part_*.pt")))


def all_parts_ready(input_dir: str, n_expected: int = NUM_EXPECTED_PARTS) -> bool:
    """Check whether *n_expected* part files are present."""
    return len(discover_part_files(input_dir)) >= n_expected


def merge_raw_mhr_parts(
    input_dir: str,
    out_path: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover ``raw_mhr_part_*.pt`` in *input_dir*/raw_mhr_parts/, sort them
    by frame name, concatenate the ``frames`` lists, and write the combined
    ``raw_mhr.pt``.

    Returns ``(out_path, all_frames, meta)`` so the caller can run any
    post-merge diagnostics (summary JSON, per-segment files, etc.).
    """
    import torch
    from models.sam_3d_body.sam_3d_body.models.meta_arch.mhr_io import (
        load_raw_mhr,
        save_raw_mhr,
    )
    from utils.id_mapping import load_segment_id_mappings

    part_files = discover_part_files(input_dir)
    if not part_files:
        parts_dir = os.path.join(input_dir, "raw_mhr_parts")
        raise FileNotFoundError(f"No raw_mhr_part_*.pt files found in {parts_dir}")

    print(f"[INFO] Merging {len(part_files)} part file(s)")

    all_frames: List[Dict[str, Any]] = []
    meta_base: Dict[str, Any] = {}
    total_corrupted = 0

    for pf in part_files:
        payload = load_raw_mhr(pf, map_location="cpu")
        part_frames = payload.get("frames", [])
        part_meta = payload.get("meta", {})
        print(f"  {os.path.basename(pf)}: {len(part_frames)} frames, "
              f"range [{part_meta.get('frame_start', '?')}, {part_meta.get('frame_end', '?')}]")
        all_frames.extend(part_frames)
        total_corrupted += part_meta.get("num_corrupted_interpolated", 0)
        if not meta_base:
            meta_base = {k: v for k, v in part_meta.items()
                         if k not in ("frame_start", "frame_end", "part_label",
                                      "num_corrupted_interpolated", "note")}

    all_frames.sort(key=lambda f: int(f["frame"]))
    print(f"[INFO] Merged total: {len(all_frames)} frames, {total_corrupted} interpolated")

    segment_id_mappings = load_segment_id_mappings(input_dir)

    meta_base.update({
        "note": f"Merged from {len(part_files)} parts with SAM3DBODY_DISABLE_TEMPORAL_SMOOTHING=1",
        "segment_id_mappings": segment_id_mappings,
        "num_corrupted_interpolated": total_corrupted,
        "num_parts_merged": len(part_files),
    })

    combined = {"frames": all_frames, "meta": meta_base}
    out_path = out_path or os.path.join(input_dir, "raw_mhr.pt")
    save_raw_mhr(out_path, combined)
    print(f"[INFO] Saved merged raw_mhr to: {out_path}")

    return out_path, all_frames, meta_base
