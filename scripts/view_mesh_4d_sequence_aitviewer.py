#!/usr/bin/env python3
"""
Visualize SAM-Body4D mesh sequences using aitviewer.

Supported formats (auto-detected):

1. **Per-person NPZ** (preferred, new default from Stage 3):
     mesh_4d_individual/
       2.npz          <- vertices (T,V,3), faces (F,3), frame_names (T,)
       4.npz

2. **Legacy per-frame PLY directories**:
     mesh_4d_individual/
       2/
         00000000.ply
         ...
       4/
         ...

3. **Legacy meshes_4d_individual.zip** (old Stage 3 output):
     Automatically extracts PLY data from the zip when neither NPZ files
     nor subdirectories are present.

Usage:
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir /path/to/mesh_4d_individual
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --ids 2 4 6 8
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --stride 2 --max_frames 300
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --scale 1.0
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --swap-yz
  python scripts/view_mesh_4d_sequence_aitviewer.py --mesh_dir ... --camera top --camera-distance 5
"""

from __future__ import annotations

import argparse
import os
import zipfile
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
    """Return person IDs from NPZ files or PLY subdirectories."""
    ids: List[str] = []
    for name in os.listdir(mesh_dir):
        p = os.path.join(mesh_dir, name)
        if name.lower().endswith(".npz") and os.path.isfile(p):
            ids.append(os.path.splitext(name)[0])
        elif os.path.isdir(p):
            ids.append(name)
    return sorted(set(ids), key=_natural_sort_key)


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


def _load_sequence_from_npz(
    npz_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load a per-person NPZ archive produced by Stage 3."""
    data = np.load(npz_path, allow_pickle=False)
    verts = np.asarray(data["vertices"], dtype=np.float32)   # (T, V, 3)
    faces = np.asarray(data["faces"], dtype=np.int32)        # (F, 3)
    frame_names = list(data.get("frame_names", np.arange(verts.shape[0])))
    if stride > 1:
        verts = verts[::stride]
        frame_names = frame_names[::stride]
    if max_frames is not None:
        verts = verts[:max_frames]
        frame_names = frame_names[:max_frames]
    if verts.shape[0] == 0:
        return None
    return verts, faces, [str(n) for n in frame_names]


def _ensure_mesh_dir_from_zip(mesh_dir: str) -> str:
    """
    If *mesh_dir* is empty (no NPZ, no subdirs) but a sibling
    ``meshes_4d_individual.zip`` exists, extract PLY data from the zip into
    *mesh_dir* so the legacy PLY loader can pick it up.

    Returns *mesh_dir* unchanged.
    """
    contents = os.listdir(mesh_dir) if os.path.isdir(mesh_dir) else []
    has_npz = any(f.endswith(".npz") for f in contents)
    has_subdirs = any(os.path.isdir(os.path.join(mesh_dir, f)) for f in contents)
    if has_npz or has_subdirs:
        return mesh_dir

    zip_path = mesh_dir + ".zip"
    if not os.path.isfile(zip_path):
        return mesh_dir

    print(f"[INFO] Extracting legacy PLY data from {zip_path} ...")
    os.makedirs(mesh_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            if not entry.lower().endswith(".ply"):
                continue
            # Archive layout: meshes_4d_individual/<pid>/<frame>.ply
            parts = entry.replace("\\", "/").split("/")
            if len(parts) >= 2:
                rel = os.path.join(*parts[-2:])
            else:
                rel = parts[-1]
            dest = os.path.join(mesh_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    print(f"[INFO] Extracted PLY data into {mesh_dir}")
    return mesh_dir


def _transform_vertices(
    verts: np.ndarray,
    scale: float = 1.0,
    swap_yz: bool = False,
) -> np.ndarray:
    """
    Apply transformations to vertices.
    
    Args:
        verts: (T, V, 3) or (V, 3) vertex array
        scale: Scale factor (e.g., 0.01 to convert cm to meters)
        swap_yz: If True, swap Y and Z axes (convert Y-up to Z-up coordinate system)
    
    Returns:
        Transformed vertices with same shape as input.
    """
    v = verts * scale
    if swap_yz:
        # Swap Y and Z: (x, y, z) -> (x, z, y)
        # This converts Y-up (SMPL convention) to Z-up (XY ground plane)
        if v.ndim == 3:
            v = v[:, :, [0, 2, 1]]
        else:
            v = v[:, [0, 2, 1]]
    return v


def _load_sequence_for_person(
    person_dir: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Load mesh sequence for a single person.
    Tries NPZ first (``<person_dir>.npz``), then falls back to a directory
    of per-frame PLY files (``<person_dir>/*.ply``).
    """
    # --- NPZ path (preferred) ---
    npz_path = person_dir + ".npz"
    if not os.path.isfile(npz_path):
        npz_path = person_dir  # caller may already pass the .npz path
    if os.path.isfile(npz_path) and npz_path.lower().endswith(".npz"):
        return _load_sequence_from_npz(npz_path, stride=stride, max_frames=max_frames)

    # --- Legacy PLY directory ---
    if not os.path.isdir(person_dir):
        print(f"[WARN] Neither NPZ nor directory found for: {person_dir} — skipping.")
        return None

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

    expected_verts_shape = None
    for p in ply_files:
        v, f = _load_ply_mesh(p)
        if faces_ref is None:
            faces_ref = f
        else:
            if f.shape != faces_ref.shape or not np.array_equal(f, faces_ref):
                pass
        if expected_verts_shape is None:
            expected_verts_shape = v.shape
        elif v.shape != expected_verts_shape:
            print(f"[ERROR] Vertex shape mismatch in {p}: got {v.shape}, expected {expected_verts_shape}")
            print(f"        Skipping this file to avoid stack error.")
            continue
        verts_seq.append(v)
        used_files.append(os.path.basename(p))

    assert faces_ref is not None
    if not verts_seq:
        print(f"[ERROR] No valid PLY files could be loaded from {person_dir}")
        return None
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
    camera_preset: str = "default",
    camera_distance: float = 5.0,
    scale: float = 0.01,
    swap_yz: bool = False,
) -> None:
    """
    View ONE person's mesh sequence, centered per-frame so the chosen center is at the origin.
    """
    person_dir = os.path.join(mesh_dir, str(person_id))
    npz_path = person_dir + ".npz"
    if not os.path.isdir(person_dir) and not os.path.isfile(npz_path):
        raise FileNotFoundError(f"No NPZ or directory found for person: {person_dir}")

    seq = _load_sequence_for_person(
        person_dir, stride=max(1, stride), max_frames=max_frames
    )
    if seq is None:
        print(f"[WARN] No meshes to view for person {person_id}.")
        return
    verts, faces, used = seq
    # Apply scale and axis transformation
    verts = _transform_vertices(verts, scale=scale, swap_yz=swap_yz)
    verts = _center_vertices_sequence(verts, center_mode=center_mode, center_vertex=center_vertex)

    from aitviewer.renderables.meshes import Meshes  # type: ignore
    from aitviewer.viewer import Viewer  # type: ignore

    v = Viewer()
    mesh = Meshes(vertices=np.asarray(verts, dtype=np.float32), faces=np.asarray(faces, dtype=np.int32),
                  name=f"Person_{person_id}_centered")
    v.scene.add(mesh)

    if show_floor:
        if swap_yz:
            v.scene.floor.plane = "xy"
        else:
            v.scene.floor.plane = "xz"
        v.scene.floor.side_length = 20

    # Set camera position based on preset
    if camera_preset != "default":
        scene_center = np.array([0.0, 0.0, 0.0])  # Centered mesh is at origin
        dist = camera_distance
        
        if swap_yz:
            # Z-up
            if camera_preset == "top":
                cam_pos = scene_center + np.array([0, 0, dist])
            elif camera_preset == "front":
                cam_pos = scene_center + np.array([0, -dist, 1.0])
            elif camera_preset == "side":
                cam_pos = scene_center + np.array([dist, 0, 1.0])
        else:
            # Y-up
            if camera_preset == "top":
                cam_pos = scene_center + np.array([0, dist, 0])
            elif camera_preset == "front":
                cam_pos = scene_center + np.array([0, 1.0, -dist])
            elif camera_preset == "side":
                cam_pos = scene_center + np.array([dist, 1.0, 0])
        
        v.scene.camera.position = cam_pos
        v.scene.camera.target = scene_center
        print(f"[INFO] Camera preset: {camera_preset}, distance: {dist}")

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
        default="pad",
        help="How to sync different sequence lengths for viewing (default: pad)",
    )
    parser.add_argument(
        "--camera",
        choices=["default", "top", "front", "side"],
        default="default",
        help="Camera view preset (default: aitviewer default, top: bird's eye view)",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=10.0,
        help="Camera distance from scene center for preset views (default: 10.0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor for vertices (default: 0.01 to convert cm to meters for aitviewer)",
    )
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap Y and Z axes to convert Y-up (SMPL) to Z-up (XY ground plane)",
    )
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    if not os.path.isdir(mesh_dir):
        # Try creating the directory from a legacy zip archive
        zip_candidate = mesh_dir + ".zip" if not mesh_dir.endswith(".zip") else mesh_dir
        if os.path.isfile(zip_candidate):
            os.makedirs(mesh_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"--mesh_dir is not a directory: {mesh_dir}")

    _ensure_mesh_dir_from_zip(mesh_dir)

    person_ids = args.ids if args.ids else _list_person_ids(mesh_dir)
    if not person_ids:
        raise ValueError(f"No person IDs (NPZ files or subdirectories) found in {mesh_dir}")

    vertices_by_id: Dict[str, np.ndarray] = {}
    faces_by_id: Dict[str, np.ndarray] = {}

    for pid in person_ids:
        pdir = os.path.join(mesh_dir, str(pid))
        seq = _load_sequence_for_person(
            pdir, stride=max(1, args.stride), max_frames=args.max_frames
        )
        if seq is None:
            continue
        verts, faces, used = seq
        verts = _transform_vertices(verts, scale=args.scale, swap_yz=args.swap_yz)
        vertices_by_id[str(pid)] = verts
        faces_by_id[str(pid)] = faces
        print(f"[OK] ID {pid}: {verts.shape[0]} frames, {verts.shape[1]} verts, {faces.shape[0]} faces")
        if used:
            print(f"     first={used[0]} last={used[-1]}")

    if not vertices_by_id:
        raise ValueError("No valid sequences loaded.")

    lengths = {k: v.shape[0] for k, v in vertices_by_id.items()}
    if lengths:
        min_t, max_t = min(lengths.values()), max(lengths.values())
        if min_t != max_t:
            print(f"\n[INFO] Frame counts differ: min={min_t}, max={max_t}")
            print(f"[INFO] Syncing with mode '{args.sync}' "
                  f"({'pad shorter to max' if args.sync == 'pad' else 'truncate all to min' if args.sync == 'truncate' else 'no sync'})")
            for pid in sorted(lengths, key=_natural_sort_key):
                t = lengths[pid]
                tag = " (shortest)" if t == min_t else ""
                print(f"       ID {pid}: {t} frames{tag}")
        else:
            print(f"\n[INFO] All {len(lengths)} person(s) have {min_t} frames")

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

    # Set floor plane based on coordinate system
    if args.swap_yz:
        # Z-up: XY is ground plane
        v.scene.floor.plane = "xy"
    else:
        # Y-up (SMPL default): XZ is ground plane
        v.scene.floor.plane = "xz"
    v.scene.floor.side_length = 20

    # Set camera position based on preset
    if args.camera != "default":
        # Compute scene center from all vertices
        all_verts = np.concatenate(list(vertices_by_id.values()), axis=0)  # (total_frames, V, 3)
        scene_center = all_verts.mean(axis=(0, 1))  # (3,)
        dist = args.camera_distance
        
        if args.swap_yz:
            # Z-up coordinate system
            if args.camera == "top":
                cam_pos = scene_center + np.array([0, 0, dist])
            elif args.camera == "front":
                cam_pos = scene_center + np.array([0, -dist, 1.5])
            elif args.camera == "side":
                cam_pos = scene_center + np.array([dist, 0, 1.5])
        else:
            # Y-up coordinate system (SMPL default)
            if args.camera == "top":
                cam_pos = scene_center + np.array([0, dist, 0])
            elif args.camera == "front":
                cam_pos = scene_center + np.array([0, 1.5, -dist])
            elif args.camera == "side":
                cam_pos = scene_center + np.array([dist, 1.5, 0])
        
        v.scene.camera.position = cam_pos
        v.scene.camera.target = scene_center
        
        print(f"[INFO] Camera preset: {args.camera}, distance: {dist}")
        print(f"       Position: {v.scene.camera.position}, Target: {v.scene.camera.target}")

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
        default="pad",
        help="How to sync different sequence lengths for viewing (default: pad)",
    )
    parser.add_argument(
        "--camera",
        choices=["default", "top", "front", "side"],
        default="default",
        help="Camera view preset (default: aitviewer default, top: bird's eye view)",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=5.0,
        help="Camera distance from scene center for preset views (default: 5.0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor for vertices (default: 0.01 to convert cm to meters for aitviewer)",
    )
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap Y and Z axes to convert Y-up (SMPL) to Z-up (XY ground plane)",
    )
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    if not os.path.isdir(mesh_dir):
        zip_candidate = mesh_dir + ".zip" if not mesh_dir.endswith(".zip") else mesh_dir
        if os.path.isfile(zip_candidate):
            os.makedirs(mesh_dir, exist_ok=True)
        else:
            raise FileNotFoundError(f"--mesh_dir is not a directory: {mesh_dir}")
    _ensure_mesh_dir_from_zip(mesh_dir)

    if args.single_id is not None:
        view_single_person_centered(
            mesh_dir=mesh_dir,
            person_id=str(args.single_id),
            stride=args.stride,
            max_frames=args.max_frames,
            center_mode=args.center_mode,
            center_vertex=args.center_vertex,
            show_floor=(not args.no_floor),
            camera_preset=args.camera,
            camera_distance=args.camera_distance,
            scale=args.scale,
            swap_yz=args.swap_yz,
        )
        return


if __name__ == "__main__":
    meshes_4d()
    # meshes_4d_single_person()


