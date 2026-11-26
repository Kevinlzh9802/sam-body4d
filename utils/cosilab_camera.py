"""
COSI-Lab cameras for the multi-view fit: fisheye intrinsics and session extrinsics of the 4TU idiap_params, side
cameras included, plus session-frame access to raw GoPro footage for cameras without a session video.

Ported from Video-OnlineHMR lib/utils/cosilab.py (load_camera, raw_frame_table, RawVideoReader), so that both repos use
the same calibration and world frame. World frame = the frame of camera 06 (session 1) or 01 (session 2), metres,
OpenCV world->camera: X_c = R X_w + t. Side cameras 12-15 are processed for session 2 only (SIDE_SESSION).

Frames: a sequence camXX_batchYY covers session frames (YY - 1) * 18000 + [0, 18000); Stage 1 names frames by the
batch frame (00000000.jpg ...). Stage 1 images are 960x540, half the 1920x1080 calibration resolution: use
Camera.scaled(0.5) for them.

Paths: SB4D_COSILAB_PROJECT overrides the cosilab_project root (default the DAIC host path), SB4D_SIDE_TIMESTAMPS the
side-camera timestamp CSVs (Video-OnlineHMR scripts/cosilab_side_timestamps.py, from the raw timecode tags; never
processed_data/gopro_data/timecode_fixed, whose timecodes are wrong).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from utils import fisheye

PROJECT_ROOT = os.environ.get('SB4D_COSILAB_PROJECT', '/tudelft.net/staff-umbrella/neon/cosilab_project')
# The 4TU upload moved out of cosilab_project on 2026-11-25 (and raw_sensor_data became raw_data)
UPLOAD_ROOT = os.environ.get('SB4D_COSILAB_4TU', '/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload')
DATA_ROOT = os.path.join(UPLOAD_ROOT, 'processed')
OLD_UPLOAD_ROOTS = (os.path.join(PROJECT_ROOT, '4TU_upload'), '/mnt/cosilab_4tu')
SIDE_TIMESTAMP_ROOT = os.environ.get(
    'SB4D_SIDE_TIMESTAMPS', '/tudelft.net/staff-umbrella/neon/experiments/Video-OnlineHMR/cosilab_side_timestamps')
# The only place on /tudelft.net sam-body4d writes to (the neon umbrella is full)
OUTPUT_ROOT = os.environ.get('SB4D_OUTPUT_ROOT', '/tudelft.net/staff-umbrella/CaptureLab/neon/sam-body4d')

FPS = 60000 / 1001
BATCH_FRAMES = 18000  # each batch is 5 min of the session video
FULL_SIZE = (1920, 1080)
SIDE_CAMS = (12, 13, 14, 15)
SIDE_SESSION = 2
# Top cameras per session; cam05 has no extrinsics (and is a bad camera), so it never takes part.
SESSION_TOP_CAMS = {1: (6, 7, 8, 9, 10), 2: (1, 2, 3, 4)}
REFERENCE_CAM = {1: 6, 2: 1}  # world frame, and the world_time.csv that maps session frames to world time


def cam_session(cam: int) -> int:
    """Session of a camera: top 01-05 -> 2, top 06-10 -> 1, side 12-15 -> SIDE_SESSION."""
    if cam in SIDE_CAMS:
        return SIDE_SESSION
    return 1 if 6 <= cam <= 10 else 2


def session_cameras(session: int) -> tuple:
    """Every camera of a session that can take part (top cameras with extrinsics, plus the side cameras)."""
    return SESSION_TOP_CAMS[session] + (SIDE_CAMS if session == SIDE_SESSION else ())


def parse_seq(seq: str):
    """'cam01_batch03' -> (cam=1, batch=3, session=2)."""
    m = re.fullmatch(r'cam(\d+)_batch(\d+)', seq)
    if m is None:
        raise ValueError(f'Expected camXX_batchYY, got {seq}')
    cam, batch = int(m.group(1)), int(m.group(2))
    return cam, batch, cam_session(cam)


def session_frame(batch: int, frame) -> np.ndarray:
    """Session frame index of batch frame(s) `frame`."""
    return (batch - 1) * BATCH_FRAMES + np.asarray(frame)


@dataclass
class Camera:
    """One calibrated fisheye camera. size = (width, height) of the images it describes."""
    name: str                   # cam01
    cam: int
    session: int
    K: np.ndarray               # (3, 3)
    D: np.ndarray               # (4,) cv2.fisheye
    R: np.ndarray               # (3, 3) world -> camera
    t: np.ndarray               # (3,)
    size: tuple = FULL_SIZE
    extrinsics_file: str = ''

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def scaled(self, s: float) -> 'Camera':
        """The same camera for images scaled by s (K only; D acts on angles)."""
        K = self.K.copy()
        K[:2] *= s
        return replace(self, K=K, size=(round(self.size[0] * s), round(self.size[1] * s)))

    def to_cam(self, X_world: np.ndarray) -> np.ndarray:
        return X_world @ self.R.T + self.t

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixels (..., 2)."""
        return fisheye.project_np(self.to_cam(X_world), self.K, self.D)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels (..., 2) -> unit rays (..., 3) in the world frame (from self.centre)."""
        shape = np.shape(pixels)[:-1]
        r = fisheye.unproject(np.reshape(pixels, (-1, 2)), self.K, self.D)
        return (r @ self.R).reshape(shape + (3,))

    def torch(self, device, dtype=torch.float32) -> Dict[str, torch.Tensor]:
        return {k: torch.as_tensor(getattr(self, k), dtype=dtype, device=device) for k in ('K', 'D', 'R', 't')}


def calib_root(cam: int) -> str:
    if cam in SIDE_CAMS:
        return os.path.join(PROJECT_ROOT, 'processed_data', 'gopro_data', 'side_view_camera_calibration',
                            'idiap_params')
    return os.path.join(DATA_ROOT, 'videos', 'camera_calibration', 'idiap_params')


def extrinsics_path(cam: int) -> str:
    session = cam_session(cam)
    sep = '-' if session == 1 else '_'  # the session 1 folders use 'camera-XX', the session 2 folders 'camera_XX'
    view = 'side' if cam in SIDE_CAMS else 'top'
    return os.path.join(calib_root(cam), f'mingling_{session}_{view}', f'camera{sep}{cam:02d}', 'extrinsic.json')


def load_camera(cam: int, extrinsics_file: Optional[str] = None) -> Camera:
    """
    Fisheye intrinsics and session extrinsics (world->camera, rvec/tvec) of a camera at 1920x1080.
    extrinsics_file replaces the stored extrinsics (same rvec/tvec JSON), e.g. a refit of side cams 12/14.
    """
    with open(os.path.join(calib_root(cam), 'intrinsics', f'camera_{cam:02d}', 'intrinsic.json')) as f:
        intr = json.load(f)
    path = extrinsics_file or extrinsics_path(cam)
    if not os.path.exists(path):
        raise FileNotFoundError(f'cam{cam:02d}: no extrinsics at {path}')
    with open(path) as f:
        extr = json.load(f)
    R = cv2.Rodrigues(np.array(extr['rvec'], dtype=np.float64).reshape(3))[0]
    return Camera(name=f'cam{cam:02d}', cam=cam, session=cam_session(cam),
                  K=np.array(intr['intrinsic'], dtype=np.float64),
                  D=np.array(intr['distortion_coefficients'], dtype=np.float64).reshape(4),
                  R=R, t=np.array(extr['tvec'], dtype=np.float64).reshape(3), extrinsics_file=path)


def parse_overrides(items) -> Dict[int, str]:
    """['cam12=path', ...] -> {12: path}."""
    out = {}
    for item in items or []:
        name, path = item.split('=', 1)
        out[int(re.sub(r'\D', '', name))] = path
    return out


# ---------------------------------------------------------------------------
# Raw GoPro footage per session frame (cameras without a session video: 02, 04, 07, 09, side cams)
# ---------------------------------------------------------------------------

def current_path(path: str) -> str:
    """A path recorded before the 4TU upload moved (e.g. a Stage 2 meta input_dir), mapped to its current place."""
    for old in OLD_UPLOAD_ROOTS:
        if path == old or path.startswith(old + os.sep):
            return UPLOAD_ROOT + path[len(old):]
    return path


def ffmpeg_exe() -> str:
    """ffmpeg on the PATH, else the one bundled with imageio-ffmpeg (the apptainer image has only that one)."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_size(path: str) -> tuple:
    cap = cv2.VideoCapture(path)
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f'{path}: cannot read the frame size')
    return w, h


def video_path(cam: int) -> str:
    return os.path.join(DATA_ROOT, 'videos', 'mingling_session_videos', f'cam{cam:02d}.mp4')


def raw_video_dir(cam: int) -> str:
    """Folder of the raw GoPro files GH0*.MP4 of a camera (raw/ is next to processed/; side cams not in the 4TU upload)."""
    session = cam_session(cam)
    if cam in SIDE_CAMS:
        return os.path.join(PROJECT_ROOT, 'raw_data', 'gopro_data',
                            f'side_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')
    return os.path.join(UPLOAD_ROOT, 'raw', 'video',
                        f'top_view_camera-mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')


def timestamp_dir(cam: int) -> str:
    session = cam_session(cam)
    sub = os.path.join(f'mingle_session_{session}', f'camera_{cam:02d}-mingle_session_{session}')
    if cam in SIDE_CAMS:
        return os.path.join(SIDE_TIMESTAMP_ROOT, sub)
    return os.path.join(DATA_ROOT, 'videos', 'timestamps', sub)


def _seconds(stamps):
    """'HH:MM:SS.ffffff' strings -> seconds of the day."""
    return np.array([int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:]) for s in stamps])


def raw_frame_table(cam: int) -> dict:
    """
    For every session frame, the raw GoPro frame of camera `cam` nearest in world time.
    Session frame -> world time from the reference camera's keypoints/vitpose_dataframe/camXX_world_time.csv,
    world time -> raw frame from the camera's per-file timestamp CSVs (one row per frame).
    Returns {'files': [GH0*.MP4], 'file': (N,) index into files, 'raw_frame': (N,), 'dt': (N,) s}.
    """
    session = cam_session(cam)
    ref = os.path.join(DATA_ROOT, 'keypoints', 'vitpose_dataframe', f'cam{REFERENCE_CAM[session]:02d}_world_time.csv')
    with open(ref) as f:
        header = f.readline().strip().split(',')
        rows = [line.rstrip('\n').split(',') for line in f]
    assert [int(r[0]) for r in rows[:2]] == [0, 1], ref
    session_time = _seconds([r[header.index('world_timestamp')] for r in rows])

    ts_dir = timestamp_dir(cam)
    names = sorted(n for n in os.listdir(ts_dir) if re.fullmatch(r'GH\d+\.csv', n))
    times, file_idx, frame_idx = [], [], []
    for i, name in enumerate(names):
        with open(os.path.join(ts_dir, name)) as f:
            col = f.readline().strip().split(',').index('world_timestamp')
            t = _seconds([line.rstrip('\n').split(',')[col] for line in f if line.strip()])
        times.append(t)
        file_idx.append(np.full(len(t), i))
        frame_idx.append(np.arange(len(t)))
    times, file_idx, frame_idx = map(np.concatenate, (times, file_idx, frame_idx))
    order = np.argsort(times, kind='stable')
    times, file_idx, frame_idx = times[order], file_idx[order], frame_idx[order]

    j = np.searchsorted(times, session_time).clip(1, len(times) - 1)
    j = np.where(np.abs(times[j - 1] - session_time) <= np.abs(times[j] - session_time), j - 1, j)
    return {'files': [n.replace('.csv', '.MP4') for n in names], 'file': file_idx[j], 'raw_frame': frame_idx[j],
            'dt': times[j] - session_time}


class RawVideoReader:
    """
    Session frames [start_frame, start_frame + num_frames) of a camera, read from its raw GoPro files (see
    raw_frame_table). Yields BGR uint8 (H, W, 3); raw frames are repeated or skipped where the world clock has gaps.
    Needs ffmpeg (ffmpeg_exe()).
    """

    def __init__(self, cam: int, start_frame: int, num_frames: int, table: Optional[dict] = None):
        self.cam, self.start_frame, self.num_frames = cam, start_frame, num_frames
        self.dir = raw_video_dir(cam)
        table = table or raw_frame_table(cam)
        sl = slice(start_frame, start_frame + num_frames)
        self.files = table['files']
        self.file, self.raw_frame, self.dt = table['file'][sl], table['raw_frame'][sl], table['dt'][sl]
        if len(self.file) < num_frames:
            raise ValueError(f'cam{cam:02d}: the session has {len(table["file"])} frames, asked up to {sl.stop}')
        self.width, self.height = video_size(os.path.join(self.dir, self.files[self.file[0]]))

    def __len__(self):
        return self.num_frames

    def segments(self):
        """(file name, first raw frame, raw frames to decode, wanted raw frames per session frame) per file."""
        out = []
        breaks = np.nonzero(np.diff(self.file) != 0)[0] + 1
        for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(self.file)]):
            raw = self.raw_frame[s:e]
            if np.any(np.diff(raw) < 0):
                raise RuntimeError(f'cam{self.cam:02d}: raw frames go backwards in {self.files[self.file[s]]}')
            out.append((self.files[self.file[s]], int(raw[0]), int(raw[-1] - raw[0] + 1), raw - raw[0]))
        return out

    def __iter__(self):
        frame_bytes = self.width * self.height * 3
        for name, first, count, wanted in self.segments():
            cmd = [ffmpeg_exe(), '-nostdin', '-v', 'error', '-ss', f'{first / FPS:.6f}',
                   '-i', os.path.join(self.dir, name), '-frames:v', str(count),
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
            try:
                pos = -1
                frame = None
                for w in wanted:
                    while pos < w:
                        buf = bytearray(frame_bytes)
                        if proc.stdout.readinto(buf) < frame_bytes:
                            return
                        frame, pos = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3), pos + 1
                    yield frame
            finally:
                proc.stdout.close()
                proc.kill()
                proc.wait()
