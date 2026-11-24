"""
The MHR body model alone (SAM-3D-Body's head_pose with its checkpoint buffers), without the image backbone and decoders.
Stage 3 only needs MHR's forward pass, so this keeps the multi-view fit's GPU memory to the body model.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


def load_mhr_head(cfg, device: torch.device) -> Tuple[torch.nn.Module, np.ndarray]:
    """cfg: the body4d.yaml OmegaConf (sam_3d_body.ckpt_path, mhr_path). Returns (head_pose, faces (F, 3) int32)."""
    from models.sam_3d_body.sam_3d_body.models.heads import build_head
    from models.sam_3d_body.sam_3d_body.utils.config import get_config

    ckpt = cfg.sam_3d_body['ckpt_path']
    model_cfg = get_config(os.path.join(os.path.dirname(ckpt), 'model_config.yaml'))
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = cfg.sam_3d_body.get('mhr_path', '')
    model_cfg.freeze()
    head = build_head(model_cfg, model_cfg.MODEL.PERSON_HEAD.POSE_TYPE)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    prefix = 'head_pose.'
    own = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    del state
    missing, unexpected = head.load_state_dict(own, strict=False)
    missing = [k for k in missing if not k.startswith(('mhr.', 'proj.'))]
    if missing or unexpected:
        raise RuntimeError(f'MHR head: missing {missing[:5]}, unexpected {unexpected[:5]} checkpoint keys')
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, head.faces.cpu().numpy().astype(np.int32)
