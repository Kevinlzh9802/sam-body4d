"""Stacking of Stage 2's per-frame raw params into (frames x people) tensors.

People with a negative real ID are left out: Stage 1's ID mapping marks people
outside the study with -1, -2, ... (Stage 2 runs from before skipping them still
have their params).

Also drop_mesh_vertices: the --no-vertices clean-up of a Stage 3 output.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Marker in stage3_run_options.txt of an output whose npz have no vertices
NO_VERTICES_OPT = "no_vertices"


def drop_mesh_vertices(stage3_dir: str) -> int:
    """Rewrites <stage3_dir>/meshes_4d_individual/*.npz without ``vertices`` and
    ``faces`` (the GBs; ``keypoints``, ``cam_t``, ``frame_names`` stay) and notes
    ``no_vertices: True`` in stage3_run_options.txt. Returns the bytes freed."""
    mesh_dir = os.path.join(stage3_dir, "meshes_4d_individual")
    freed = 0
    for fn in sorted(os.listdir(mesh_dir)) if os.path.isdir(mesh_dir) else []:
        if not fn.endswith(".npz") or fn.endswith(".part.npz"):
            continue
        path = os.path.join(mesh_dir, fn)
        with np.load(path) as z:
            if "vertices" not in z.files:
                continue
            keep = {k: z[k] for k in z.files if k not in ("vertices", "faces")}
        size = os.path.getsize(path)
        tmp = path[:-4] + ".part.npz"
        np.savez_compressed(tmp, **keep)
        os.replace(tmp, path)
        freed += size - os.path.getsize(path)
    opts = os.path.join(stage3_dir, "stage3_run_options.txt")
    lines = open(opts, encoding="utf-8").read().splitlines() if os.path.isfile(opts) else []
    lines = [ln for ln in lines if not ln.startswith(f"{NO_VERTICES_OPT}: ")] + [f"{NO_VERTICES_OPT}: True"]
    with open(opts, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return freed


def stack_frames_to_tensors(
    frames: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[List[int]], Dict[int, List[int]], List[str], List[int]]:
    """
    Convert raw frames payload into flattened tensors shaped (T*N, D).
    Slot order is fixed across frames: sorted(all_obj_ids).
    """
    T = len(frames)
    frame_names = [str(fr.get("frame", f"{i:08d}")) for i, fr in enumerate(frames)]
    obj_set = set()
    for fr in frames:
        for p in fr.get("people", []):
            if p is None:
                continue
            oid = p.get("obj_id", None)
            if oid is not None and int(oid) >= 0:
                obj_set.add(int(oid))
    obj_ids_all = sorted(obj_set)
    N = len(obj_ids_all)
    if N == 0:
        raise ValueError("No people found in raw payload.")

    slot_of = {oid: si for si, oid in enumerate(obj_ids_all)}

    frame_obj_ids_slots: List[List[int]] = []
    vis_flags: Dict[int, List[int]] = {oid: [0] * T for oid in obj_ids_all}
    for ti, fr in enumerate(frames):
        slots = [0] * N
        for p in fr.get("people", []):
            if p is None:
                continue
            oid = int(p.get("obj_id"))
            si = slot_of.get(oid, None)
            if si is None:  # negative ID
                continue
            slots[si] = oid
            vis_flags[oid][ti] = 1
        frame_obj_ids_slots.append(slots)

    def first_shape(key: str) -> int:
        for fr in frames:
            for p in fr.get("people", []):
                v = p.get(key, None)
                if v is not None:
                    arr = np.asarray(v)
                    return int(arr.reshape(-1).shape[0])
        return 0

    dims = {
        "global_rot": first_shape("global_rot"),
        "body_pose": first_shape("body_pose"),
        "hand": first_shape("hand"),
        "scale": first_shape("scale"),
        "shape": first_shape("shape"),
        "face": first_shape("face"),
        "pred_cam_t": first_shape("pred_cam_t"),
        "focal_length": 1,
    }

    B = T * N
    mhr: Dict[str, torch.Tensor] = {}
    for k, d in dims.items():
        if d <= 0:
            continue
        mhr[k] = torch.zeros((B, d), dtype=torch.float32, device=device)

    for ti, fr in enumerate(frames):
        for p in fr.get("people", []):
            si = slot_of.get(int(p.get("obj_id")))
            if si is None:
                continue
            bi = ti * N + si
            for k in ["global_rot", "body_pose", "hand", "scale", "shape", "face", "pred_cam_t"]:
                if k not in mhr:
                    continue
                v = p.get(k, None)
                if v is None:
                    continue
                vv = torch.from_numpy(np.asarray(v, dtype=np.float32).reshape(-1)).to(device)
                if vv.numel() == mhr[k].shape[1]:
                    mhr[k][bi] = vv
            if "focal_length" in mhr:
                fl = p.get("focal_length", None)
                if fl is not None:
                    mhr["focal_length"][bi, 0] = float(np.asarray(fl).reshape(-1)[0])

    return mhr, frame_obj_ids_slots, vis_flags, frame_names, obj_ids_all
