#!/usr/bin/env python3
"""
Reprojection overlay video of a Stage 3 output: every Stage 1 frame with each
person's final mesh projected onto it (filled silhouette + outline), the body
skeleton (primary joints only) and the person's IDs, and the Stage 1 mask
outlines in the same colour (grey: a mask without a mesh)
-> <stage3_dir>/reproj_overlay.mp4 (H.264, 60 fps by default).

Works on any Stage 3 output folder (stage3_height, stage3_multiview,
stage3_noground, ...): meshes_4d_individual/*.npz, back to camera space with the
extrinsics when exported in world coordinates, and stage3_run_options.txt for
the intrinsics and the Stage 1 folder (images(.zip), masks(.zip),
id_mapping.json; archives are read in place). The video covers the frames from
the first to the last frame with a mesh. A person has the same colour in every
camera's video and in scripts/view_stage3.py.

Called at the end of stage3_mhr.py and, per camera, stage3_multiview.py
(--no-video there to skip). To (re)make videos by hand:

  python scripts/render_overlay_video.py <stage3_dir> [<stage3_dir> ...]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from multiprocessing import get_context
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from scripts.view_stage3 import person_color, read_run_options
from smoothing.ground_plane_opt import load_extrinsics_json
from smoothing.reproj_overlay_plot import PRIMARY_KPS, SKELETON, _bgr, _project, _rasterize_mesh
from utils.camera_utils import adjust_K, read_camera_intrinsics_new
from utils.id_mapping import find_segment_for_frame, load_segment_id_mappings

VIDEO_NAME = "reproj_overlay.mp4"
KP_ROW = {k: i for i, k in enumerate(PRIMARY_KPS)}
# Per person and frame: silhouette contours (int16 pixels) and the primary joints'
# pixels (NaN behind the camera)
Layer = Tuple[List[np.ndarray], np.ndarray]


class Stage1Frames:
    """Stage 1 frames by name (e.g. 00005400) from <dir>/<kind>/ or <dir>/<kind>.zip."""

    def __init__(self, stage1_dir: str, kind: str, exts: Tuple[str, ...]) -> None:
        path = os.path.join(stage1_dir, kind)
        self.dir, self.zip, names = None, None, []
        if os.path.isdir(path):
            self.dir, names = path, os.listdir(path)
        elif os.path.isfile(path + ".zip"):
            self.zip = zipfile.ZipFile(path + ".zip")
            names = self.zip.namelist()
        self.names = {}
        for n in names:
            stem, ext = os.path.splitext(os.path.basename(n))
            if ext.lower() in exts:
                self.names[stem] = n

    def read(self, name: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(self.names[name])
        with open(os.path.join(self.dir, self.names[name]), "rb") as fh:
            return fh.read()


def iter_npz_rows(path: str, key: str, chunk: int) -> Iterator[np.ndarray]:
    """The array *key* of an npz in blocks of rows, decompressed as a stream."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(fh)
        assert not fortran, f"{path}:{key} is Fortran-ordered"
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - start)
            yield np.frombuffer(fh.read(n * row_bytes), dtype=dtype).reshape((n,) + tuple(shape[1:]))


def stage1_dir_of(stage3_dir: str, opts: Dict[str, str]) -> str:
    """The Stage 1 folder stage3_mhr.py read (--input-dir, else the raw file's folder)."""
    if opts.get("input_dir", "None") != "None":
        return opts["input_dir"]
    return os.path.dirname(opts.get("raw", stage3_dir))


def _person_layers(job) -> Tuple[int, Dict[str, Layer]]:
    """Pass 1, one person (worker process): silhouette and joints per frame."""
    path, pid, K, to_cam, h, w, chunk = job
    with np.load(path) as z:
        names = [str(n) for n in z["frame_names"]]
        kps = z["keypoints"][:, list(PRIMARY_KPS)].astype(np.float64) if "keypoints" in z.files else None
        faces = z["faces"]
    if to_cam is not None:  # world (world units) -> camera (m): x_cam = (R x + t) / ws
        R, t, ws = to_cam
        if kps is not None:
            kps = (kps @ R.T + t) / ws
    layers: Dict[str, Layer] = {}
    i = 0
    for block in iter_npz_rows(path, "vertices", chunk):
        block = block.astype(np.float64)
        if to_cam is not None:
            block = (block @ R.T + t) / ws
        for v in block:
            sil = _rasterize_mesh(_project(K, v), faces, v[:, 2], h, w)
            cnts, _ = cv2.findContours(sil, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            uv = np.full((len(PRIMARY_KPS), 2), np.nan, dtype=np.float32)
            if kps is not None:
                front = kps[i, :, 2] > 0.01
                uv[front] = _project(K, kps[i, front])
            layers[names[i]] = ([c.astype(np.int16) for c in cnts], uv)
            i += 1
    return pid, layers


class VideoWriter:
    """H.264 mp4 through imageio-ffmpeg (in the container), else OpenCV's mp4v."""

    def __init__(self, path: str, w: int, h: int, fps: float) -> None:
        self.gen = self.cv = None
        try:
            import imageio_ffmpeg
            self.gen = imageio_ffmpeg.write_frames(
                path, (w, h), pix_fmt_in="bgr24", fps=fps, codec="libx264", quality=None,
                macro_block_size=2, ffmpeg_log_level="error",
                output_params=["-crf", "23", "-preset", "veryfast"])
            self.gen.send(None)
        except ImportError:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, bgr: np.ndarray) -> None:
        if self.gen is not None:
            self.gen.send(np.ascontiguousarray(bgr))
        else:
            self.cv.write(bgr)

    def close(self) -> None:
        if self.gen is not None:
            self.gen.close()
        else:
            self.cv.release()


def _draw_frame(canvas: np.ndarray, mask: Optional[np.ndarray], people: Dict[int, Layer],
                c2a: Dict[int, int], header: str) -> None:
    # Stage 1 mask outlines, in the colour of the person's mesh
    if mask is not None:
        for uid in np.unique(mask):
            if uid == 0:
                continue
            pid = c2a.get(int(uid))
            col = _bgr(person_color(pid)) if pid in people else (160, 160, 160)
            cnts, _ = cv2.findContours((mask == uid).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, col, 2)
    # Mesh silhouettes (one blend for everybody), outlines, skeletons
    people = {pid: ([c.astype(np.int32) for c in cnts], uv) for pid, (cnts, uv) in people.items()}
    overlay = np.zeros_like(canvas)
    for pid, (cnts, _) in people.items():
        if cnts:
            cv2.fillPoly(overlay, cnts, _bgr(person_color(pid)))  # even-odd: holes stay open
    cv2.addWeighted(canvas, 1.0, overlay, 0.25, 0, dst=canvas)
    a2c = {a: c for c, a in c2a.items()}
    for pid, (cnts, uv) in people.items():
        col = _bgr(person_color(pid))
        cv2.drawContours(canvas, cnts, -1, col, 1)
        for a, b in SKELETON:
            pa, pb = uv[KP_ROW[a]], uv[KP_ROW[b]]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), col, 2, cv2.LINE_AA)
        for p in uv[np.isfinite(uv).all(axis=1)].astype(int):
            cv2.circle(canvas, tuple(p), 3, col, -1)
            cv2.circle(canvas, tuple(p), 3, (255, 255, 255), 1)
    for pid, (cnts, _) in people.items():  # labels last, on top of everybody
        col = _bgr(person_color(pid))
        if cnts:
            pts = np.concatenate(cnts).reshape(-1, 2)
            label = f"T:{a2c[pid]} R:{pid}" if pid in a2c else f"R:{pid}"
            lx, ly = max(0, int(pts[:, 0].min())), max(20, int(pts[:, 1].min()) - 8)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 4), (0, 0, 0), -1)
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(canvas, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def render_overlay_video(stage3_dir: str, stage1_dir: Optional[str] = None, out_path: Optional[str] = None,
                         fps: float = 60.0, workers: Optional[int] = None, chunk: int = 256) -> Optional[str]:
    """Writes the overlay video of a Stage 3 output folder; returns its path, or
    None when there is nothing to render (no frames or no meshes)."""
    t0 = time.time()
    opts = read_run_options(stage3_dir)
    stage1_dir = stage1_dir or stage1_dir_of(stage3_dir, opts)
    out_path = out_path or os.path.join(stage3_dir, VIDEO_NAME)
    images = Stage1Frames(stage1_dir, "images", (".jpg", ".jpeg", ".png", ".bmp"))
    masks = Stage1Frames(stage1_dir, "masks", (".png", ".bmp"))
    mesh_dir = os.path.join(stage3_dir, "meshes_4d_individual")
    paths = {int(fn[:-4]): os.path.join(mesh_dir, fn) for fn in sorted(os.listdir(mesh_dir))
             if fn.endswith(".npz")} if os.path.isdir(mesh_dir) else {}
    if not images.names or not paths:
        print(f"[INFO] Overlay video skipped: {'no meshes in ' + mesh_dir if images.names else 'no Stage 1 images in ' + stage1_dir}")
        return None

    K, _ = read_camera_intrinsics_new(opts["camera_intrinsics_json"])
    K = adjust_K(K, scale=float(opts.get("camera_scale", 0.5)))
    to_cam = None
    if opts.get("export_camera_space") != "True" and opts.get("extrinsics_json", "None") != "None":
        e = load_extrinsics_json(opts["extrinsics_json"], device=torch.device("cpu"))
        to_cam = (e.R.numpy().astype(np.float64), e.t.numpy().reshape(1, 3).astype(np.float64),
                  float(opts.get("world_scale", 100.0)))
    first = cv2.imdecode(np.frombuffer(images.read(next(iter(images.names))), np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]

    # Pass 1: per person, silhouettes and joints (in parallel over the people)
    jobs = [(p, pid, K, to_cam, h, w, chunk) for pid, p in paths.items()]
    workers = max(1, min(len(jobs), workers or len(os.sched_getaffinity(0))))
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            per_person = dict(pool.imap_unordered(_person_layers, jobs))
    else:
        per_person = dict(map(_person_layers, jobs))
    by_frame: Dict[str, Dict[int, Layer]] = {}
    for pid in sorted(per_person):
        for name, layer in per_person[pid].items():
            by_frame.setdefault(name, {})[pid] = layer
    del per_person
    lo, hi = min(map(int, by_frame)), max(map(int, by_frame))
    frames = sorted((n for n in images.names if lo <= int(n) <= hi), key=int)
    t1 = time.time()

    # Pass 2: draw and encode
    id_maps = load_segment_id_mappings(stage1_dir)
    title = "/".join(os.path.normpath(os.path.abspath(stage3_dir)).split(os.sep)[-2:])  # <seq>/<label>
    tmp = out_path[:-4] + ".part.mp4"
    out = VideoWriter(tmp, w, h, fps)
    try:
        for name in frames:
            canvas = cv2.imdecode(np.frombuffer(images.read(name), np.uint8), cv2.IMREAD_COLOR)
            mask = None
            if name in masks.names:
                img = Image.open(io.BytesIO(masks.read(name)))
                mask = np.array(img)  # palette PNG: the tracking IDs
                if mask.ndim == 3:
                    mask = np.array(img.convert("L"))
            seg = find_segment_for_frame(int(name), id_maps) if id_maps else None
            c2a = seg["consecutive_to_actual"] if seg else {}
            _draw_frame(canvas, mask, by_frame.get(name, {}), c2a, f"{title}  |  Frame {name}")
            out.write(canvas)
    finally:
        out.close()
    os.replace(tmp, out_path)
    print(f"[INFO] Overlay video: {len(frames)} frames ({frames[0]}-{frames[-1]}), {len(paths)} people -> "
          f"{out_path} ({t1 - t0:.0f} s meshes on {workers} processes, {time.time() - t1:.0f} s drawing)",
          flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage3_dirs", nargs="+", help="Stage 3 output folders")
    ap.add_argument("--stage1-dir", default=None,
                    help="Stage 1 folder (default: from stage3_run_options.txt); one input folder only")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=None, help="processes for the meshes (default: all CPUs)")
    args = ap.parse_args()
    if args.stage1_dir and len(args.stage3_dirs) > 1:
        raise SystemExit("[ERROR] --stage1-dir goes with one Stage 3 folder")
    for d in args.stage3_dirs:
        render_overlay_video(d, stage1_dir=args.stage1_dir, fps=args.fps, workers=args.workers)


if __name__ == "__main__":
    main()
