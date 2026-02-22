"""
Shared model construction helpers.

All heavy imports (``models.*``) are deferred to inside each function so that
importing this module is cheap.
"""
from __future__ import annotations

import os
import sys

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
_sam3d_body_pkg = os.path.join(REPO_DIR, "models", "sam_3d_body")
if _sam3d_body_pkg not in sys.path:
    sys.path.insert(0, _sam3d_body_pkg)


def build_sam3_from_config(cfg):
    """Build the SAM-3 video model and return ``(sam3_model, predictor)``."""
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3["ckpt_path"])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return sam3_model, predictor


def build_sam3d_body_from_config(cfg, device: torch.device):
    """Build a :class:`SAM3DBodyEstimator` from *cfg* on *device*."""
    from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

    mhr_path = cfg.sam_3d_body.get("mhr_path", "")
    fov_path = cfg.sam_3d_body.get("fov_path", "")
    model, model_cfg = load_sam_3d_body(cfg.sam_3d_body["ckpt_path"], device=device, mhr_path=mhr_path)

    fov_estimator = None
    if fov_path:
        fov_estimator = FOVEstimator(name="moge2", device=device, path=fov_path)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
