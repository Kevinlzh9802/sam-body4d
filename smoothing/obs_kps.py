from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ObsKps:
    # kp_idx refers to mhr70 index
    kp_idx: torch.LongTensor  # (M,)
    xy: torch.FloatTensor     # (M,2)


def load_bboxes_kps_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_obj_id_to_bbox_idx(bboxes_kps_data: Any) -> Optional[Dict[int, int]]:
    """
    Build mapping obj_id -> bbox index using frame 0 `pids`, if present.
    """
    if bboxes_kps_data is None:
        return None
    try:
        rec0 = bboxes_kps_data[0]
        if isinstance(rec0, dict) and "pids" in rec0:
            pids = rec0["pids"]
            return {int(pid): int(i) for i, pid in enumerate(pids)}
    except Exception:
        return None
    return None


def get_obs_for_person(
    bboxes_kps_data: Any,
    frame_idx: int,
    obj_id: int,
    obj_id_to_bbox_idx: Dict[int, int],
    device: torch.device,
) -> Optional[ObsKps]:
    if bboxes_kps_data is None:
        return None
    if frame_idx < 0:
        return None
    try:
        if frame_idx >= len(bboxes_kps_data):
            return None
    except Exception:
        return None
    if obj_id_to_bbox_idx is None or int(obj_id) not in obj_id_to_bbox_idx:
        return None

    rec = bboxes_kps_data[frame_idx]
    if not isinstance(rec, dict):
        return None
    kps_all = rec.get("kps", None)
    if kps_all is None:
        return None

    kps_all = np.asarray(kps_all, dtype=np.float32)
    if kps_all.ndim != 3 or kps_all.shape[-1] < 3:
        return None
    bbox_idx = int(obj_id_to_bbox_idx[int(obj_id)])
    if bbox_idx < 0 or bbox_idx >= kps_all.shape[0]:
        return None

    obs = kps_all[bbox_idx]  # (K,3): (x,y,kp_idx)
    kp_idx_list: List[int] = []
    xy_list: List[List[float]] = []
    for row in obs:
        x, y, ki = float(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(ki):
            continue
        kp_idx = int(ki)
        if kp_idx < 0:
            continue
        kp_idx_list.append(kp_idx)
        xy_list.append([x, y])

    if not kp_idx_list:
        return None

    return ObsKps(
        kp_idx=torch.tensor(kp_idx_list, dtype=torch.long, device=device),
        xy=torch.tensor(xy_list, dtype=torch.float32, device=device),
    )

