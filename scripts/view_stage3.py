#!/usr/bin/env python3
"""
Browser viewer (viser) for Stage 3 meshes: plays back every person of one or
more stage3_mhr_*.py outputs (meshes_4d_individual/<id>.npz).

Camera-space outputs (--export-camera-space) of several cameras of one rig are
placed with the rig's multi-camera calibration (libCalib JSON
parameters-camera-AA-to-BB.json, poses relative to camera AA), which is also
used to draw the camera frustums. Without a floor calibration the scene is
levelled from the people themselves: "up" is the mean body axis, and the floor
is at the median foot height. World-space outputs (Stage 3 with
--extrinsics-json, the default then) are shown as exported (floor z=0), with
the cameras from experiments/extrinsics/extrinsic_cam_<MM>.json.

Only numpy and viser are needed (no torch / container). The vertices are read
straight from the npz stream, so --frames limits memory on full sequences:
about 0.22 MB per person per shown frame.

Usage:
  python scripts/view_stage3.py experiments/stage3_local/cam06_batch01/cam06_seg010/stage3_full
  python scripts/view_stage3.py <cam06 out> <cam08 out> --frames 5400 6000 --stride 2
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


def world_cameras(rig: List[Camera], calib_dir: str, world_scale: float) -> List[Camera]:
    """The rig's cameras posed in the world frame, from extrinsic_cam_<MM>.json
    (world_to_camera, as scripts/make_ground_extrinsics.py writes them)."""
    out = []
    for cam in rig:
        path = os.path.join(calib_dir, f"extrinsic_{cam.name.replace('cam', 'cam_')}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        out.append(Camera(cam.name, np.asarray(d["rotation"], dtype=np.float64).reshape(3, 3),
                          np.asarray(d["translation"], dtype=np.float64).reshape(3) / world_scale, cam.f, cam.size))
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


def load_people(out_dir: str, frame_range, stride: int, ids, to_rig: Optional[Camera],
                scale: float, cam_name: str) -> List[Person]:
    people = []
    for path in sorted(glob.glob(os.path.join(out_dir, "meshes_4d_individual", "*.npz")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
        pid = int(os.path.splitext(os.path.basename(path))[0])
        if ids and pid not in ids:
            continue
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
        people.append(Person(f"{cam_name} id {pid}", pid, frames[rows], v.astype(np.float32),
                             faces.astype(np.int32), to_rig))
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
    ap.add_argument("--calib-dir", default=os.path.join(REPO_DIR, "experiments", "extrinsics"),
                    help="where to look for parameters-camera-AA-to-BB.json")
    ap.add_argument("--calib", help="rig calibration JSON (default: found in --calib-dir)")
    ap.add_argument("--no-level", action="store_true", help="keep the rig / world axes")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import viser
    import viser.transforms as vtf

    people: List[Person] = []
    rig: Optional[List[Camera]] = None
    rig_path = None
    world_rig: Optional[List[Camera]] = None
    shown_cams = set()
    spaces = set()
    for out_dir in args.out_dirs:
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
            if path and world_rig is None:
                world_rig = world_cameras(load_rig_calibration(path), args.calib_dir, scale)
            shown_cams.add(cam_name)
        loaded = load_people(out_dir, args.frames, args.stride, set(args.ids or []), to_rig, scale, cam_name)
        n_frames = sum(len(p.frames) for p in loaded)
        print(f"[INFO] {out_dir}: {len(loaded)} people, {n_frames} person-frames "
              f"({n_frames * 18439 * 12 / 1e9:.1f} GB)", flush=True)
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
        gaps = np.diff(p.frames) > args.stride  # no line across frames the person is missing
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
