"""Video metadata utilities."""
from __future__ import annotations

from typing import Tuple

import cv2


def read_video_metadata(path: str) -> Tuple[float, int, int, int]:
    """Return ``(fps, total_frames, width, height)`` for the video at *path*."""
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, width, height
