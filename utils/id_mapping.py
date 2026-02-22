"""
Per-segment ID mapping utilities.

Handles loading, conversion, and reverse-lookup between tracking
(consecutive) IDs stored in mask pixels and actual (real) person IDs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


SegmentMapping = Dict[str, Any]


def load_segment_id_mappings(path_or_dir: str) -> List[SegmentMapping]:
    """Load segment ID mappings from *id_mapping.json*.

    *path_or_dir* may be either the JSON file itself or a directory
    containing ``id_mapping.json``.

    Returns a list of segment dicts, each with keys
    ``segment_key``, ``frame_start``, ``frame_end``, ``consecutive_to_actual``.
    """
    if os.path.isdir(path_or_dir):
        path = os.path.join(path_or_dir, "id_mapping.json")
    else:
        path = path_or_dir
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segments" in data:
        return [
            {
                "segment_key": s.get("segment_key", ""),
                "frame_start": int(s["frame_start"]),
                "frame_end": int(s["frame_end"]),
                "consecutive_to_actual": {int(k): int(v) for k, v in s["consecutive_to_actual"].items()},
            }
            for s in data["segments"]
        ]

    if "consecutive_to_actual" in data:
        return [
            {
                "segment_key": "",
                "frame_start": 0,
                "frame_end": 999_999_999,
                "consecutive_to_actual": {int(k): int(v) for k, v in data["consecutive_to_actual"].items()},
            }
        ]

    return []


def load_segment_id_mappings_from_meta(
    meta: Dict[str, Any],
    fallback_dir: Optional[str] = None,
) -> Optional[List[SegmentMapping]]:
    """Extract segment ID mappings from a raw_mhr payload ``meta`` dict.

    Falls back to loading ``id_mapping.json`` from *fallback_dir* when the
    meta dict has no embedded mappings.
    """
    mappings = meta.get("segment_id_mappings", None)
    if mappings:
        result = []
        for s in mappings:
            c2a = s.get("consecutive_to_actual", {})
            result.append({
                "segment_key": s.get("segment_key", ""),
                "frame_start": int(s.get("frame_start", 0)),
                "frame_end": int(s.get("frame_end", 999_999_999)),
                "consecutive_to_actual": {int(k): int(v) for k, v in c2a.items()},
            })
        return result

    if fallback_dir:
        loaded = load_segment_id_mappings(fallback_dir)
        return loaded if loaded else None

    return None


def find_segment_for_frame(
    frame_idx: int,
    segment_id_mappings: List[SegmentMapping],
) -> Optional[SegmentMapping]:
    """Return the segment dict that contains *frame_idx*, or ``None``."""
    for seg in segment_id_mappings:
        if seg["frame_start"] <= frame_idx <= seg["frame_end"]:
            return seg
    return None


def to_actual_pid(
    consecutive_id: int,
    frame_idx: int,
    segment_id_mappings: List[SegmentMapping],
) -> int:
    """Convert a tracking (consecutive) ID to the actual person ID."""
    seg = find_segment_for_frame(frame_idx, segment_id_mappings)
    if seg is not None:
        mapping = seg["consecutive_to_actual"]
        if int(consecutive_id) in mapping:
            return mapping[int(consecutive_id)]
    return consecutive_id


def actual_to_consecutive(
    actual_pid: int,
    frame_idx: int,
    segment_id_mappings: List[SegmentMapping],
) -> Optional[int]:
    """Reverse-map an actual person ID to the tracking (consecutive) ID.

    Returns ``None`` if no mapping is found.
    """
    seg = find_segment_for_frame(frame_idx, segment_id_mappings)
    if seg is not None:
        for cid, aid in seg["consecutive_to_actual"].items():
            if int(aid) == int(actual_pid):
                return int(cid)
    return None
