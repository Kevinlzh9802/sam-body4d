#!/usr/bin/env python3
"""
Stage 1 input for a cosilab camera without a session video (top 02, 04, 07, 09; side 12-15): 10 s segments of a
batch, taken from the raw GoPro files at the session frames (utils/cosilab_camera.py raw_frame_table: world time from
the raw MP4 timecode tags, 59.94 fps drift; never processed_data/gopro_data/timecode_fixed, whose timecodes are wrong).

Frame f of segment s of batch b is session frame (b - 1) * 18000 + s * 600 + f, the same moment as in every other
camera of the session, so the Stage 1-3 frame numbers line up with the annotated cameras.

Output: <out-root>/camXX_batchBB/camXX_segNNN.mp4 (960x540 H.264 by default, as the annotated cameras' segments) and
frame_table.json (raw file / frame and world-time error per segment). Needs ffmpeg (cosilab_camera.ffmpeg_exe).

python dataset/make_raw_segments.py --cam 13 --batch 1 [--segments 10-12] [--out-root ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from utils import cosilab_camera as cc  # noqa: E402

SEG_FRAMES = 600


def parse_range(spec: str, n: int):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return [s for s in out if 1 <= s <= n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cam', type=int, required=True)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--segments', default=None, help='1-based segment numbers, e.g. 11 or 10-12 (default: all 30)')
    ap.add_argument('--out-root', default=os.path.join(cc.OUTPUT_ROOT, 'video_segs_10s'))
    ap.add_argument('--scale', type=float, default=0.5, help='output size / 1920x1080')
    ap.add_argument('--crf', type=int, default=18)
    args = ap.parse_args()

    seq = f'cam{args.cam:02d}_batch{args.batch:02d}'
    out_dir = os.path.join(args.out_root, seq)
    os.makedirs(out_dir, exist_ok=True)
    table = cc.raw_frame_table(args.cam)
    n_seg = cc.BATCH_FRAMES // SEG_FRAMES
    segs = parse_range(args.segments, n_seg) if args.segments else list(range(1, n_seg + 1))
    first = (args.batch - 1) * cc.BATCH_FRAMES
    rows = []
    for s in segs:
        start = first + (s - 1) * SEG_FRAMES
        reader = cc.RawVideoReader(args.cam, start, SEG_FRAMES, table=table)
        w, h = round(reader.width * args.scale), round(reader.height * args.scale)
        path = os.path.join(out_dir, f'cam{args.cam:02d}_seg{s:03d}.mp4')
        cmd = [cc.ffmpeg_exe(), '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
               '-r', f'{cc.FPS:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
               path + '.part.mp4']
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = 0
        for frame in reader:
            if (w, h) != (reader.width, reader.height):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(frame.tobytes())
            n += 1
        proc.stdin.close()
        if proc.wait() != 0 or n != SEG_FRAMES:
            raise SystemExit(f'[ERROR] {path}: ffmpeg exit {proc.returncode}, {n}/{SEG_FRAMES} frames')
        os.replace(path + '.part.mp4', path)
        sl = slice(start, start + SEG_FRAMES)
        dt = table['dt'][sl]
        rows.append({'segment': s, 'session_frames': [start, start + SEG_FRAMES - 1],
                     'raw_files': sorted({table['files'][i] for i in table['file'][sl]}),
                     'max_abs_dt_s': float(np.abs(dt).max())})
        print(f'[INFO] {path}: session frames {start}-{start + SEG_FRAMES - 1}, raw {rows[-1]["raw_files"]}, '
              f'world-time error up to {rows[-1]["max_abs_dt_s"] * 1000:.1f} ms', flush=True)
    meta_path = os.path.join(out_dir, 'frame_table.json')
    old = json.load(open(meta_path)) if os.path.exists(meta_path) else {'segments': []}
    keep = [r for r in old['segments'] if r['segment'] not in set(segs)]
    with open(meta_path, 'w') as fh:
        json.dump({'cam': args.cam, 'batch': args.batch, 'fps': cc.FPS, 'scale': args.scale,
                   'segments': sorted(keep + rows, key=lambda r: r['segment'])}, fh, indent=1)


if __name__ == '__main__':
    main()
