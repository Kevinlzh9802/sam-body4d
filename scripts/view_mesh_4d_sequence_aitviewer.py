#!/usr/bin/env python3
"""
Visualize SAM-Body4D mesh sequences stored as per-frame .ply files using aitviewer.

Expected directory structure (as produced by this repo after our object-id folder fix):
  mesh_4d_individual/
    2/
      00000000.ply
      00000001.ply
      ...
    4/
      ...

Usage:
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir /path/to/mesh_4d_individual
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --ids 2 4 6 8
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --stride 2 --max_frames 300
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _natural_sort_key(s: str) -> Tuple:
    # Works well for zero-padded frame names and also plain integers.
    base = os.path.basename(s)
    name, _ext = os.path.splitext(base)
    try:
        return (0, int(name))
    except Exception:
        return (1, base)


def _list_person_ids(mesh_dir: str) -> List[str]:
    ids = []
    for name in os.listdir(mesh_dir):
        p = os.path.join(mesh_dir, name)
        if os.path.isdir(p):
            ids.append(name)
    return sorted(ids, key=_natural_sort_key)


def _load_ply_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a .ply as (vertices, faces).
    Uses trimesh if available (already a dependency of this repo's renderer).
    """
    try:
        import trimesh  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "trimesh is required to load .ply files for this viewer script. "
            "Install it in your environment (e.g. `pip install trimesh`)."
        ) from e

    mesh = trimesh.load(path, process=False)
    # trimesh can return a Scene; handle that.
    if hasattr(mesh, "geometry") and not hasattr(mesh, "faces"):
        # Scene -> take first geometry
        geoms = list(mesh.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry found in {path}")
        mesh = geoms[0]

    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape {v.shape} in {path}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Invalid faces shape {f.shape} in {path}")
    return v, f


def _load_sequence_for_person(
    person_dir: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ply_files = [
        os.path.join(person_dir, f)
        for f in os.listdir(person_dir)
        if f.lower().endswith(".ply")
    ]
    ply_files = sorted(ply_files, key=_natural_sort_key)
    if not ply_files:
        raise ValueError(f"No .ply files found in: {person_dir}")

    if stride > 1:
        ply_files = ply_files[::stride]
    if max_frames is not None:
        ply_files = ply_files[:max_frames]

    verts_seq: List[np.ndarray] = []
    faces_ref: Optional[np.ndarray] = None
    used_files: List[str] = []

    for p in ply_files:
        v, f = _load_ply_mesh(p)
        if faces_ref is None:
            faces_ref = f
        else:
            # If faces differ, keep first faces (common for consistent topology).
            if f.shape != faces_ref.shape or not np.array_equal(f, faces_ref):
                pass
        verts_seq.append(v)
        used_files.append(os.path.basename(p))

    assert faces_ref is not None
    verts = np.stack(verts_seq, axis=0)  # (T, V, 3)
    return verts, faces_ref, used_files


def _sync_sequences(
    vertices_by_id: Dict[str, np.ndarray],
    mode: str,
) -> Dict[str, np.ndarray]:
    """
    Make all sequences the same length for a clean timeline in aitviewer.
    - truncate: cut to min length
    - pad: pad shorter sequences with last frame to max length
    """
    lengths = {k: v.shape[0] for k, v in vertices_by_id.items()}
    if not lengths:
        return vertices_by_id
    min_t = min(lengths.values())
    max_t = max(lengths.values())

    if mode == "none":
        return vertices_by_id
    if mode == "truncate":
        return {k: v[:min_t] for k, v in vertices_by_id.items()}
    if mode == "pad":
        out: Dict[str, np.ndarray] = {}
        for k, v in vertices_by_id.items():
            if v.shape[0] == max_t:
                out[k] = v
                continue
            last = v[-1:]
            pad = np.repeat(last, repeats=(max_t - v.shape[0]), axis=0)
            out[k] = np.concatenate([v, pad], axis=0)
        return out
    raise ValueError(f"Unknown sync mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True, help="Path to mesh_4d_individual directory")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional list of person/object IDs (folder names). If omitted, loads all.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames per person")
    parser.add_argument(
        "--sync",
        choices=["truncate", "pad", "none"],
        default="truncate",
        help="How to sync different sequence lengths for viewing (default: truncate)",
    )
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    if not os.path.isdir(mesh_dir):
        raise FileNotFoundError(f"--mesh_dir is not a directory: {mesh_dir}")

    person_ids = args.ids if args.ids else _list_person_ids(mesh_dir)
    if not person_ids:
        raise ValueError(f"No person folders found in {mesh_dir}")

    vertices_by_id: Dict[str, np.ndarray] = {}
    faces_by_id: Dict[str, np.ndarray] = {}

    for pid in person_ids:
        pdir = os.path.join(mesh_dir, str(pid))
        if not os.path.isdir(pdir):
            print(f"[WARN] Skipping missing folder: {pdir}")
            continue
        verts, faces, used = _load_sequence_for_person(
            pdir, stride=max(1, args.stride), max_frames=args.max_frames
        )
        vertices_by_id[str(pid)] = verts
        faces_by_id[str(pid)] = faces
        print(f"[OK] ID {pid}: {verts.shape[0]} frames, {verts.shape[1]} verts, {faces.shape[0]} faces")
        if used:
            print(f"     first={used[0]} last={used[-1]}")

    if not vertices_by_id:
        raise ValueError("No valid sequences loaded.")

    vertices_by_id = _sync_sequences(vertices_by_id, args.sync)

    # Import aitviewer last (it may initialize OpenGL context)
    from aitviewer.renderables.meshes import Meshes  # type: ignore
    from aitviewer.viewer import Viewer  # type: ignore

    v = Viewer()

    # Simple repeating color palette
    colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.8, 0.8, 0.2),
        (0.8, 0.2, 0.8),
        (0.2, 0.8, 0.8),
    ]

    for i, pid in enumerate(sorted(vertices_by_id.keys(), key=_natural_sort_key)):
        verts = np.asarray(vertices_by_id[pid], dtype=np.float32)
        faces = np.asarray(faces_by_id[pid], dtype=np.int32)

        mesh = Meshes(vertices=verts, faces=faces, name=f"Person_{pid}")
        # Color handling differs across aitviewer versions; set if present.
        try:
            mesh.material.base_color = np.array([*colors[i % len(colors)], 1.0], dtype=np.float32)
        except Exception:
            pass

        v.scene.add(mesh)

    v.scene.floor.plane = "xy"
    v.scene.floor.side_length = 20
    print("Controls: SPACE play/pause, left/right arrows step frames.")
    v.run()


if __name__ == "__main__":
    main()


