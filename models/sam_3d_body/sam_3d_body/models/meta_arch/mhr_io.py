"""
I/O helpers for saving/loading "raw MHR" tensors for a decoupled pipeline:

- Stage2: masks/images -> raw params (no temporal smoothing), saved to disk
- Stage3: raw params -> temporal smoothing -> mhr_forward -> meshes

This module is intentionally independent of the heavy image encoder forward pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def save_raw_mhr(path: str, payload: Dict[str, Any]) -> None:
    """
    Save raw MHR payload with torch serialization.
    Expected keys (recommended):
      - "mhr": Dict[str, torch.Tensor]  (flattened B=T*N)
      - "num_frames": int
      - "frame_obj_ids": List[List[int]]  (len T; obj_ids present per frame)
      - "occ_dict": Optional[Dict[int, List[int]]]
      - "meta": Optional[Dict[str, Any]]
    """
    torch.save(payload, path)


def load_raw_mhr(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """
    Load raw MHR payload saved by save_raw_mhr().

    Note: PyTorch 2.6+ defaults `weights_only=True` in torch.load, which breaks
    loading arbitrary dict payloads that contain numpy objects. We explicitly
    set weights_only=False here to preserve the previous behavior.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def is_raw_mhr_payload(d: Dict[str, Any]) -> bool:
    return isinstance(d, dict) and "mhr" in d and "num_frames" in d and "frame_obj_ids" in d

