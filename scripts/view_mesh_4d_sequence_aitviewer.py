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
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    ply_files = [
        os.path.join(person_dir, f)
        for f in os.listdir(person_dir)
        if f.lower().endswith(".ply")
    ]
    ply_files = sorted(ply_files, key=_natural_sort_key)
    if not ply_files:
        print(f"[WARN] No .ply files found in: {person_dir} — skipping person.")
        return None

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

def _center_vertices_sequence(
    verts: np.ndarray,
    center_mode: str = "mean",
    center_vertex: Optional[int] = None,
) -> np.ndarray:
    """
    Center a (T, V, 3) vertex sequence by subtracting a per-frame center.
    - If center_vertex is provided, uses that vertex as the center (per frame).
    - Otherwise uses per-frame mean/median over vertices.
    """
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"Expected verts shape (T,V,3), got {verts.shape}")

    if center_vertex is not None:
        if not (0 <= center_vertex < verts.shape[1]):
            raise ValueError(f"center_vertex out of range: {center_vertex} for V={verts.shape[1]}")
        center = verts[:, center_vertex, :]  # (T, 3)
    else:
        if center_mode == "mean":
            center = verts.mean(axis=1)  # (T, 3)
        elif center_mode == "median":
            center = np.median(verts, axis=1)  # (T, 3)
        else:
            raise ValueError(f"Unknown center_mode: {center_mode} (use 'mean' or 'median')")

    return verts - center[:, None, :]


def view_single_person_centered(
    mesh_dir: str,
    person_id: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    center_mode: str = "mean",
    center_vertex: Optional[int] = None,
    show_floor: bool = True,
) -> None:
    """
    View ONE person's mesh sequence, centered per-frame so the chosen center is at the origin.
    """
    person_dir = os.path.join(mesh_dir, str(person_id))
    if not os.path.isdir(person_dir):
        raise FileNotFoundError(f"Person folder not found: {person_dir}")

    seq = _load_sequence_for_person(
        person_dir, stride=max(1, stride), max_frames=max_frames
    )
    if seq is None:
        print(f"[WARN] No meshes to view for person {person_id}.")
        return
    verts, faces, used = seq
    verts = _center_vertices_sequence(verts, center_mode=center_mode, center_vertex=center_vertex)

    from aitviewer.renderables.meshes import Meshes  # type: ignore
    from aitviewer.viewer import Viewer  # type: ignore

    v = Viewer()
    mesh = Meshes(vertices=np.asarray(verts, dtype=np.float32), faces=np.asarray(faces, dtype=np.int32),
                  name=f"Person_{person_id}_centered")
    v.scene.add(mesh)

    if show_floor:
        v.scene.floor.plane = "xy"
        v.scene.floor.side_length = 20

    if used:
        print(f"[OK] Centered ID {person_id}: {verts.shape[0]} frames; first={used[0]} last={used[-1]}")
    print("Controls: SPACE play/pause, left/right arrows step frames.")
    v.run()


def meshes_4d() -> None:
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
        seq = _load_sequence_for_person(
            pdir, stride=max(1, args.stride), max_frames=args.max_frames
        )
        if seq is None:
            continue
        verts, faces, used = seq
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


def meshes_4d_single_person() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True, help="Path to mesh_4d_individual directory")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional list of person/object IDs (folder names). If omitted, loads all.",
    )

    # >>> add these args <<<
    parser.add_argument(
        "--single_id",
        default=None,
        help="If set, view only this person, centered at origin (overrides --ids).",
    )
    parser.add_argument(
        "--center_mode",
        choices=["mean", "median"],
        default="mean",
        help="How to compute per-frame center if --center_vertex is not set (default: mean).",
    )
    parser.add_argument(
        "--center_vertex",
        type=int,
        default=None,
        help="Optional vertex index to use as center (overrides --center_mode).",
    )
    parser.add_argument("--no_floor", action="store_true", help="Disable floor in centered single-person view.")
    # <<< end add args <<<

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

    # >>> add this early-exit branch <<<
    if args.single_id is not None:
        view_single_person_centered(
            mesh_dir=mesh_dir,
            person_id=str(args.single_id),
            stride=args.stride,
            max_frames=args.max_frames,
            center_mode=args.center_mode,
            center_vertex=args.center_vertex,
            show_floor=(not args.no_floor),
        )
        return


if __name__ == "__main__":
    meshes_4d()
    # meshes_4d_single_person()


