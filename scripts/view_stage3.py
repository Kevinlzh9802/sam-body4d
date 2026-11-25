#!/usr/bin/env python3
"""
Browser viewer (viser) for Stage 3 meshes: plays back every person of one or
more stage3_mhr.py outputs (meshes_4d_individual/<id>.npz).

Camera-space outputs (--export-camera-space) of several cameras of one rig are
placed with the rig's multi-camera calibration (libCalib JSON
parameters-camera-AA-to-BB.json, poses relative to camera AA), which is also
used to draw the camera frustums. Without a floor calibration the scene is
levelled from the people themselves: "up" is the mean body axis, and the floor
is at the median foot height. World-space outputs (Stage 3 with
--extrinsics-json, the default then) are shown as exported (floor z=0), with
the cameras from experiments/extrinsics/extrinsic_cam_<MM>.json.

stage3_mvfit.py outputs (the mvfit dir, or one of its cameras/<camXX_batchBB>/)
show the shared world bodies (world/<pid>.npz, floor z=0) with the cameras of
cameras/*/extrinsics/. Those bodies hold no vertices unless the fit ran with
--save-vertices; re-pose them once in the container (MHR needs torch):
  bash scripts/local_sif.sh python scripts/export_mvfit_vertices.py <mvfit dir> --stride 2
which writes experiments/mvfit_viewer/<mvfit dir name>/ (found here by default).

Only numpy and viser are needed (no torch / container): the pyproject `viz`
group (uv sync --only-group viz, once). The vertices are read straight from the
npz stream, so --frames limits memory on full sequences: about 0.22 MB per
person per shown frame.

Usage:
  uv run --no-sync python scripts/view_stage3.py experiments/stage3_local/cam06_batch01/cam06_seg010/stage3
  uv run --no-sync python scripts/view_stage3.py <cam06 out> <cam08 out> --frames 5400 6000 --stride 2
  uv run --no-sync python scripts/view_stage3.py <mvfit dir> --frames 6000 6300
Open http://localhost:8080. On a cluster node, forward the port first:
  ssh -L 8080:localhost:8080 <login node>   (then run there, or hop on to the node)
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Rig calibrations: the repo's experiments/extrinsics, else the cosilab one (mounted at the same path as on DAIC)
COSILAB_CALIB_DIR = "/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/keypoints/extrinsics"
DEFAULT_CALIB_DIR = next((d for d in (os.path.join(REPO_DIR, "experiments", "extrinsics"), COSILAB_CALIB_DIR)
                          if os.path.isdir(d)), os.path.join(REPO_DIR, "experiments", "extrinsics"))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    name: str
    R: np.ndarray       # (3, 3) rig -> camera (x_cam = R x_rig + t)
    t: np.ndarray       # (3,) metres
    f: float
    size: Tuple[int, int]  # (width, height) px


def load_rig_calibration(path: str) -> List[Camera]:
    """Cameras of a libCalib multi-camera JSON; names from AA-to-BB in the file name."""
    with open(path, "r", encoding="utf-8") as fh:
        cams = json.load(fh)["Calibration"]["cameras"]
    m = re.search(r"camera-(\d+)-to-(\d+)", os.path.basename(path))
    first = int(m.group(1)) if m and int(m.group(2)) - int(m.group(1)) + 1 == len(cams) else None
    out = []
    for i, c in enumerate(cams):
        rot, tr = c["transform"]["rotation"], c["transform"]["translation"]
        data = c["model"]["ptr_wrapper"]["data"]
        size = data["CameraModelCRT"]["CameraModelBase"]["imageSize"]
        out.append(Camera(
            name=f"cam{first + i:02d}" if first is not None else f"cam{i}",
            R=_rodrigues(np.array([rot["rx"], rot["ry"], rot["rz"]], dtype=np.float64)),
            t=np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float64),
            f=float(data["parameters"]["f"]["val"]),
            size=(int(size["width"]), int(size["height"])),
        ))
    return out


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * Kx @ Kx


def world_cameras(rig: Optional[List[Camera]], calib_dir: str, world_scale: float) -> List[Camera]:
    """Cameras posed in the world frame, from extrinsic_cam_<MM>.json (world_to_camera, as stage3_mvfit.py writes
    them): the rig's cameras, or without a rig every such file in calib_dir. Focal length and image size from the
    JSON's K and image_size (stage3_mvfit.py writes them), else from the rig."""
    if rig is None:
        names = sorted(re.sub(r"^extrinsic_cam_", "cam", os.path.basename(p))[:-5]
                       for p in glob.glob(os.path.join(calib_dir, "extrinsic_cam_[0-9][0-9].json")))
        rig_of = {}
    else:
        names = [cam.name for cam in rig]
        rig_of = {cam.name: cam for cam in rig}
    out = []
    for name in names:
        path = os.path.join(calib_dir, f"extrinsic_{name.replace('cam', 'cam_')}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if "K" in d and "image_size" in d:
            f, size = float(d["K"][0][0]), (int(d["image_size"][0]), int(d["image_size"][1]))
        elif name in rig_of:
            f, size = rig_of[name].f, rig_of[name].size
        else:
            continue
        out.append(Camera(name, np.asarray(d["rotation"], dtype=np.float64).reshape(3, 3),
                          np.asarray(d["translation"], dtype=np.float64).reshape(3) / world_scale, f, size))
    return out


def find_rig_calibration(calib_dir: str, cam_num: int) -> Optional[str]:
    for path in sorted(glob.glob(os.path.join(calib_dir, "parameters-camera-*-to-*.json"))):
        m = re.search(r"camera-(\d+)-to-(\d+)", os.path.basename(path))
        if m and int(m.group(1)) <= cam_num <= int(m.group(2)):
            return path
    return None


def read_run_options(out_dir: str) -> Dict[str, str]:
    opts = {}
    path = os.path.join(out_dir, "stage3_run_options.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.partition(": ")
                opts[k.strip()] = v.strip()
    return opts


def camera_number(out_dir: str, opts: Dict[str, str]) -> Optional[int]:
    for text in (opts.get("camera_intrinsics_json", ""), os.path.abspath(out_dir)):
        m = re.search(r"camera-(\d+)\.json", text) or re.search(r"cam(\d{2})", text)
        if m:
            return int(m.group(1))
    return None


# cosilab recording sessions. Within a session (and batch), the same frame
# number is the same moment in every camera; the two sessions are separate
# recordings. cam05 has no rig calibration yet (parameters-camera-01-to-04.json),
# so the multi-view step cannot use it.
COSILAB_SESSIONS = {1: (6, 7, 8, 9, 10), 2: (1, 2, 3, 4, 5)}


def session_of(cam_num: Optional[int]) -> Optional[int]:
    return next((s for s, cams in COSILAB_SESSIONS.items() if cam_num in cams), None)


def batch_number(out_dir: str, opts: Dict[str, str]) -> Optional[int]:
    for text in (os.path.abspath(out_dir), opts.get("raw", ""), opts.get("input_dir", ""), opts.get("input", "")):
        m = re.search(r"cam\d{2}_batch(\d+)", text)
        if m:
            return int(m.group(1))
    return None


def read_npz_rows(path: str, key: str, rows: np.ndarray, chunk: int = 256) -> np.ndarray:
    """rows (sorted) of the array *key* in an npz, decompressing as a stream so
    only the selected rows are kept in memory."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(fh)
        assert not fortran, f"{path}:{key} is Fortran-ordered"
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        out = np.empty((len(rows),) + tuple(row_shape), dtype=dtype)
        pos, k = 0, 0
        while k < len(rows):
            start = int(rows[k])
            if start > pos:
                fh.seek((start - pos) * row_bytes, 1)
                pos = start
            n = min(chunk, int(rows[-1]) - pos + 1)
            block = np.frombuffer(fh.read(n * row_bytes), dtype=dtype).reshape((n,) + tuple(row_shape))
            sel = rows[(rows >= pos) & (rows < pos + n)]
            out[k:k + len(sel)] = block[sel - pos]
            k += len(sel)
            pos += n
        return out


@dataclass
class Person:
    label: str           # e.g. "cam06 id 28"
    pid: int
    frames: np.ndarray   # frame numbers shown
    verts: np.ndarray    # (len(frames), V, 3) float32, in the rig / world frame (m)
    faces: np.ndarray
    cam: Optional[Camera]


def expand_mvfit_root(path: str) -> List[str]:
    """A stage3_mvfit.py output dir (world/ + cameras/) -> its per-camera dirs; anything else as given."""
    cams = sorted(glob.glob(os.path.join(path, "cameras", "*", "stage3_run_options.txt")))
    if os.path.isdir(os.path.join(path, "world")) and cams:
        return [os.path.dirname(p) for p in cams]
    return [path]


def has_vertices(path: str) -> bool:
    with zipfile.ZipFile(path) as zf:
        return "vertices.npy" in zf.namelist()


def default_vertices_dirs(mesh_dir: str) -> List[str]:
    """Where scripts/export_mvfit_vertices.py puts the re-posed vertices of an mvfit world/ dir."""
    root = os.path.dirname(os.path.normpath(mesh_dir))
    return [os.path.join(REPO_DIR, "experiments", "mvfit_viewer", os.path.basename(root)), os.path.join(root, "viewer")]


def mesh_sources(mesh_dir: str, ids, vertices_dir: Optional[str]) -> Dict[int, str]:
    """pid -> npz with vertices: the mesh file itself, or the exported vertices of a body stored without them."""
    paths = {int(os.path.splitext(os.path.basename(p))[0]): p for p in glob.glob(os.path.join(mesh_dir, "*.npz"))}
    out, missing = {}, []
    search = [vertices_dir] if vertices_dir else default_vertices_dirs(mesh_dir)
    for pid in sorted(paths):
        if ids and pid not in ids:
            continue
        if has_vertices(paths[pid]):
            out[pid] = paths[pid]
            continue
        found = next((os.path.join(d, f"{pid}.npz") for d in search if os.path.isfile(os.path.join(d, f"{pid}.npz"))),
                     None)
        if found is None:
            missing.append(pid)
        else:
            out[pid] = found
    if missing:
        root = os.path.dirname(os.path.normpath(mesh_dir))
        raise SystemExit(f"[ERROR] {mesh_dir}: no vertices for ids {missing} (looked in {', '.join(search)}). "
                         f"Re-pose them once in the container:\n"
                         f"  bash scripts/local_sif.sh python scripts/export_mvfit_vertices.py {root} --stride 2")
    return out


# ---------------------------------------------------------------------------
# Display decimation
# ---------------------------------------------------------------------------

def cluster_vertices(ref: np.ndarray, faces: np.ndarray, target: int) -> np.ndarray:
    """Cluster label per vertex (0..C-1), about *target* clusters: vertices in one voxel of the reference pose that
    are connected through edges inside that voxel (so a hand resting on a hip stays apart from it). Fixed for all
    frames, so the coarse mesh keeps its faces while the body moves."""
    lo = ref.min(0)

    def voxels(size: float) -> np.ndarray:
        return np.unique(np.floor((ref - lo) / size).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)

    small, big = 1e-4, float(np.ptp(ref, 0).max()) + 1e-6  # binary search on the voxel side for ~target voxels
    for _ in range(25):
        mid = (small * big) ** 0.5
        if voxels(mid).max() + 1 > target:
            small = mid
        else:
            big = mid
    vox = voxels(big)
    # Split each voxel into its connected parts: label propagation over the edges within a voxel
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = e[vox[e[:, 0]] == vox[e[:, 1]]]
    lab = np.arange(len(ref))
    while True:
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        new = lab.copy()
        np.minimum.at(new, e[:, 0], m)
        np.minimum.at(new, e[:, 1], m)
        new = new[new]  # pointer jumping
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab, return_inverse=True)[1].reshape(-1)


def decimate(verts: np.ndarray, faces: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
    """(T, V, 3) vertices and faces -> (T, C, 3) cluster means and the faces between three different clusters."""
    lab = cluster_vertices(verts[0], faces, target)
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.nonzero(np.diff(lab[order]))[0] + 1]
    counts = np.diff(np.r_[starts, len(lab)]).astype(np.float32)
    out = np.add.reduceat(verts[:, order], starts, axis=1) / counts[None, :, None]
    f = lab[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    f = f[np.sort(np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1])]
    return out.astype(np.float32), f.astype(np.int32)


def load_people(mesh_dir: str, frame_range, stride: int, ids, to_rig: Optional[Camera],
                scale: float, cam_name: str, vertices_dir: Optional[str] = None,
                target_verts: Optional[int] = None) -> List[Person]:
    people = []
    for pid, path in mesh_sources(mesh_dir, ids, vertices_dir).items():
        with np.load(path) as d:
            frames = d["frame_names"].astype(np.int64)
            faces = d["faces"]
        keep = np.ones(len(frames), dtype=bool)
        if frame_range:
            keep &= (frames >= frame_range[0]) & (frames < frame_range[1])
        keep &= (frames - (frame_range[0] if frame_range else 0)) % stride == 0
        rows = np.nonzero(keep)[0]
        if len(rows) == 0:
            continue
        v = read_npz_rows(path, "vertices", rows) / scale
        if to_rig is not None:  # x_rig = R^T (x_cam - t)
            v = (v - to_rig.t.astype(np.float32)) @ to_rig.R.astype(np.float32)
        v, faces = v.astype(np.float32), faces.astype(np.int32)
        if target_verts:
            v, faces = decimate(v, faces, target_verts)
        people.append(Person(f"{cam_name} id {pid}", pid, frames[rows], v, faces, to_rig))
    return people


# ---------------------------------------------------------------------------
# Scene levelling
# ---------------------------------------------------------------------------

def level_scene(people: List[Person]) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation A and offset b (x -> A x + b) that make the mean body axis +z,
    put the floor at the median foot height (z=0) and centre the people, so
    people whose depth is off float above or sink below it.
    The axis is signed to point towards the person's camera (cameras are
    mounted above head height; looking straight down, the image axes say
    nothing about up)."""
    axes, samples = [], []
    for p in people:
        cam_centre = np.zeros(3) if p.cam is None else -p.cam.R.T @ p.cam.t
        for v in p.verts[:: max(1, len(p.verts) // 20), ::40]:
            c = v - v.mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][0]
            axes.append(axis if axis @ (cam_centre - v.mean(0)) > 0 else -axis)
            samples.append(v)
    up = np.mean(axes, 0)
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(up, z)
    s, c = np.linalg.norm(k), float(up @ z)
    A = np.eye(3) if s < 1e-9 else _rodrigues(k / s * np.arctan2(s, c))
    samples = [v @ A.T for v in samples]
    centres = np.stack([v.mean(0) for v in samples])
    feet = np.median([v[:, 2].min() for v in samples])
    b = -np.array([np.median(centres[:, 0]), np.median(centres[:, 1]), feet])
    return A, b


def person_color(pid: int) -> Tuple[int, int, int]:
    h = (pid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", nargs="+", help="stage 3 output dirs (with meshes_4d_individual/)")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="frame numbers [START, END)")
    ap.add_argument("--stride", type=int, default=1, help="show every n-th frame")
    ap.add_argument("--ids", type=int, nargs="+", help="only these person IDs")
    ap.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR,
                    help="where to look for parameters-camera-AA-to-BB.json (default: %(default)s)")
    ap.add_argument("--calib", help="rig calibration JSON (default: found in --calib-dir)")
    ap.add_argument("--vertices-dir", help="re-posed vertices of mvfit bodies stored without them "
                    "(default: experiments/mvfit_viewer/<mvfit dir name>/, then <mvfit dir>/viewer/)")
    ap.add_argument("--no-level", action="store_true", help="keep the rig / world axes")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--verts", type=int, default=2000,
                    help="vertices per displayed body (the browser gets every body's vertices each frame; "
                         "full MHR meshes, 18439, stall playback with many people)")
    ap.add_argument("--full-res", action="store_true", help="show the full meshes (slow with many people)")
    args = ap.parse_args()

    import viser
    import viser.transforms as vtf

    out_dirs = [d for arg in args.out_dirs for d in expand_mvfit_root(arg)]
    loaded_mesh_dirs = set()
    people: List[Person] = []
    rig: Optional[List[Camera]] = None
    rig_path = None
    world_rig: Optional[List[Camera]] = None
    shown_cams = set()
    spaces = set()
    # Only cameras of one session and batch show the same moments
    recordings = {(session_of(camera_number(d, o)), batch_number(d, o))
                  for d, o in ((d, read_run_options(d)) for d in out_dirs)}
    if len(recordings) > 1:
        raise SystemExit(f"[ERROR] The outputs are from different recordings (session, batch): {sorted(recordings, key=str)}; "
                         "frame numbers only match within one session (cam06-10 / cam01-05) and batch.")
    for out_dir in out_dirs:
        opts = read_run_options(out_dir)
        cam_num = camera_number(out_dir, opts)
        cam_name = f"cam{cam_num:02d}" if cam_num is not None else os.path.basename(os.path.normpath(out_dir))
        camera_space = opts.get("export_camera_space", "True") == "True" or opts.get("extrinsics_json") in (None, "None")
        to_rig, scale = None, 1.0
        spaces.add("camera" if camera_space else "world")
        if camera_space:
            path = args.calib or (find_rig_calibration(args.calib_dir, cam_num) if cam_num is not None else None)
            if path:
                if rig_path not in (None, path):
                    raise SystemExit(f"[ERROR] {out_dir} belongs to rig {path}, others to {rig_path}; "
                                     "camera-space outputs of different rigs cannot be combined.")
                rig, rig_path = load_rig_calibration(path), path
                to_rig = next((c for c in rig if c.name == cam_name), None)
            if to_rig is None:
                print(f"[WARN] No rig calibration for {cam_name}; showing {out_dir} in its own camera frame.")
            shown_cams.add(cam_name)
        else:
            scale = float(opts.get("world_scale", 100.0))
            path = args.calib or (find_rig_calibration(args.calib_dir, cam_num) if cam_num is not None else None)
            # Multi-view outputs carry the extrinsics of their own (estimated) floor
            own = os.path.join(out_dir, "extrinsics")
            extr_dir = own if os.path.isdir(own) else args.calib_dir
            if world_rig is None:
                world_rig = world_cameras(load_rig_calibration(path) if path else None, extr_dir, scale)
            shown_cams.add(cam_name)
        # stage3_mvfit.py: one world body per person, shared by every camera (mesh_dir in the run options)
        # (prefer the world/ beside cameras/: mesh_dir is absolute on the machine that ran the fit)
        beside = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out_dir))), "world")
        mesh_dir = (beside if opts.get("mode") == "mvfit" and os.path.isdir(beside)
                    else opts.get("mesh_dir") or os.path.join(out_dir, "meshes_4d_individual"))
        if os.path.realpath(mesh_dir) in loaded_mesh_dirs:
            continue
        loaded_mesh_dirs.add(os.path.realpath(mesh_dir))
        label = "mvfit" if opts.get("mode") == "mvfit" else cam_name
        loaded = load_people(mesh_dir, args.frames, args.stride, set(args.ids or []), to_rig, scale, label,
                             args.vertices_dir, None if args.full_res else args.verts)
        n_frames = sum(len(p.frames) for p in loaded)
        per_frame = sum(p.verts.shape[1] for p in loaded) * 12 / 1e6
        print(f"[INFO] {out_dir}: {len(loaded)} people, {n_frames} person-frames "
              f"({sum(p.verts.nbytes for p in loaded) / 1e9:.2f} GB; up to {per_frame:.1f} MB to the browser per frame, "
              f"{'full meshes' if args.full_res else f'decimated to ~{args.verts} vertices, --full-res for all'})",
              flush=True)
        people += loaded
    if not people:
        raise SystemExit("[ERROR] No meshes found (check the dirs, --frames and --ids).")

    if len(spaces) > 1:
        raise SystemExit("[ERROR] Mix of camera-space and world-space outputs; show them separately.")
    world = spaces == {"world"}
    # World outputs already have the floor at z=0 and z up
    level = not args.no_level and not world
    A, b = level_scene(people) if level else (np.eye(3), np.zeros(3))
    if world:
        rig = world_rig
    for p in people:
        p.verts = p.verts @ A.T.astype(np.float32) + b.astype(np.float32)
    timeline = np.unique(np.concatenate([p.frames for p in people]))
    row_of = [{int(f): i for i, f in enumerate(p.frames)} for p in people]

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z" if level or world else "-y")
    allv = np.concatenate([p.verts[::10, ::50].reshape(-1, 3) for p in people])
    lo, hi = allv.min(0), allv.max(0)
    if level or world:
        size = float(np.ceil(max(hi[0] - lo[0], hi[1] - lo[1]) + 4))
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        server.scene.add_grid("/floor", width=size, height=size, plane="xy", cell_size=0.5,
                              position=(cx, cy, 0.0))
        # Oblique view of the whole group, from the side
        span = size - 4
        server.initial_camera.look_at = (cx, cy, 0.8)
        server.initial_camera.position = (cx - 0.5 * span, cy - 1.1 * span - 2, 0.8 * span + 2)
        server.initial_camera.up = (0.0, 0.0, 1.0)

    for cam in rig or []:
        C = A @ (-cam.R.T @ cam.t) + b
        R_scene_cam = A @ cam.R.T
        w, h = cam.size
        server.scene.add_camera_frustum(
            f"/cameras/{cam.name}", fov=float(2 * np.arctan(h / 2 / cam.f)), aspect=w / h, scale=0.3,
            color=(255, 110, 40) if cam.name in shown_cams else (30, 30, 30),
            wxyz=vtf.SO3.from_matrix(R_scene_cam).wxyz, position=C)
        server.scene.add_label(f"/cameras/{cam.name}/label", cam.name, position=(0.0, 0.0, 0.0))

    meshes, labels, trajs = [], [], []
    for i, p in enumerate(people):
        color = person_color(p.pid)
        meshes.append(server.scene.add_mesh_simple(f"/people/{i}", p.verts[0], p.faces, color=color))
        labels.append(server.scene.add_label(f"/labels/{i}", p.label, position=tuple(p.verts[0].mean(0)),
                                             visible=False))
        centre = p.verts[:, ::25].mean(1)
        if level or world:
            centre[:, 2] = 0.02  # on the floor
        steps = np.diff(p.frames)  # no line across frames the person is missing (exports can have their own stride)
        gaps = steps > max(args.stride, int(steps.min()) if len(steps) else 1)
        seg = np.stack([centre[:-1], centre[1:]], 1)[~gaps]
        trajs.append(server.scene.add_line_segments(
            f"/trajectories/{i}", seg, colors=color, thickness=0.02, visible=False) if len(seg) else None)

    with server.gui.add_folder("Playback"):
        g_step = server.gui.add_slider("Timestep", min=0, max=len(timeline) - 1, step=1, initial_value=0)
        g_frame = server.gui.add_number("Frame", initial_value=int(timeline[0]), disabled=True)
        g_play = server.gui.add_checkbox("Playing", initial_value=True)
        g_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=args.fps)
        g_traj = server.gui.add_checkbox("Show trajectories", initial_value=False)
        g_ids = server.gui.add_checkbox("Show IDs", initial_value=False)

    def show(step: int) -> None:
        f = int(timeline[step])
        with server.atomic():
            g_frame.value = f
            for i, p in enumerate(people):
                r = row_of[i].get(f)
                meshes[i].visible = r is not None
                labels[i].visible = r is not None and g_ids.value
                if r is not None:
                    meshes[i].vertices = p.verts[r]
                    if g_ids.value:
                        v = p.verts[r]
                        labels[i].position = (*v[:, :2].mean(0), float(v[:, 2].max()) + 0.15)

    g_step.on_update(lambda _: show(int(g_step.value)))
    g_ids.on_update(lambda _: show(int(g_step.value)))

    @g_traj.on_update
    def _(_) -> None:
        for t in trajs:
            if t is not None:
                t.visible = g_traj.value

    show(0)
    print(f"[INFO] {len(people)} people, {len(timeline)} frames "
          f"({timeline[0]}..{timeline[-1]}, stride {args.stride}). Open http://localhost:{args.port}", flush=True)
    while True:
        t0 = time.time()
        if g_play.value:
            g_step.value = (int(g_step.value) + 1) % len(timeline)
        time.sleep(max(0.0, 1.0 / float(g_fps.value) - (time.time() - t0)))


if __name__ == "__main__":
    main()
