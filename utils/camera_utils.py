"""
Canonical camera intrinsics utilities.

Consolidates ``adjust_K``, ``read_camera_intrinsics`` (old JSON format),
and ``read_camera_intrinsics_new`` (Calibration JSON plus compatible old
JSON fallback) so every stage imports from one place.
"""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np


def adjust_K(K: np.ndarray, scale: float) -> np.ndarray:
    """Scale a 3x3 camera intrinsics matrix by *scale*."""
    return np.array(
        [
            [K[0, 0] * scale, 0, K[0, 2] * scale],
            [0, K[1, 1] * scale, K[1, 2] * scale],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def read_camera_intrinsics(intrinsic_file: str, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load old-format JSON with ``intrinsic`` and ``distortion_coefficients``.

    Returns ``(K_scaled, dist_coeffs)``.
    """
    with open(intrinsic_file, "r", encoding="utf-8") as f:
        intrinsic_data = json.load(f)
    K = np.array(intrinsic_data["intrinsic"], dtype=np.float32)
    dist_coeffs = np.array(intrinsic_data.get("distortion_coefficients", []), dtype=np.float32)
    K = adjust_K(K, scale=scale)
    return K, dist_coeffs


def read_camera_intrinsics_new(intrinsic_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load intrinsics from either supported camera JSON schema.

    Supported schemas:
    - new Calibration JSON: ``Calibration.cameras[0].model...parameters``
    - simple JSON: ``{"intrinsic": [[...]], "distortion_coefficients": [...]}``

    Returns ``(K, dist_coeffs)``. K is not pre-scaled.
    """
    with open(intrinsic_file, "r", encoding="utf-8") as fh:
        intrinsic_data = json.load(fh)

    if "intrinsic" in intrinsic_data:
        K = np.asarray(intrinsic_data["intrinsic"], dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(
            intrinsic_data.get("distortion_coefficients", []),
            dtype=np.float64,
        ).reshape(-1)
        return K, dist_coeffs

    params = intrinsic_data["Calibration"]["cameras"][0]["model"]["ptr_wrapper"]["data"]["parameters"]

    f = params["f"]["val"]
    cx = params["cx"]["val"]
    cy = params["cy"]["val"]

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    ks = [params[f"k{i}"]["val"] for i in range(1, 5)]
    dist_coeffs = np.array(ks, dtype=np.float64)
    return K, dist_coeffs
