#!/usr/bin/env python3
"""
Minimal SAM3-only smoke test.

Goal: confirm that SAM3 can be imported and a model can be constructed from a checkpoint,
without running the full SAM-Body4D pipeline.

Usage:
  python scripts/test_sam3_only.py --ckpt /path/to/sam3.pt --device cuda --video /path/to/video.mp4

Notes:
- This is intentionally light: it builds the model + initializes predictor state.
- It does NOT run full propagation (which can be slow); it can optionally run 1-step propagate.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to SAM3 checkpoint (sam3.pt)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for torch")
    parser.add_argument("--video", default=None, help="Optional video path to init tracker state")
    parser.add_argument("--one_step", action="store_true", help="If set, run a tiny one-step propagate")
    args = parser.parse_args()

    print("[INFO] python:", sys.executable)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] cuda available:", torch.cuda.is_available())

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; continuing on CPU may be very slow.")

    try:
        from sam3.model_builder import build_sam3_video_model
    except Exception:
        print("[FAIL] Could not import sam3. If you see ModuleNotFoundError, ensure sam3 is installed in the env.")
        traceback.print_exc()
        return 2

    print("[INFO] Building SAM3 video model ...")
    sam3_model = build_sam3_video_model(checkpoint_path=args.ckpt)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    sam3_model.eval()
    print("[OK] Built SAM3 model. Tracker:", type(predictor).__name__)

    if args.video is None:
        print("[INFO] No --video provided; stopping after build.")
        return 0

    print("[INFO] Initializing tracker state ...")
    state = predictor.init_state(video_path=args.video)
    predictor.clear_all_points_in_video(state)
    print("[OK] Initialized inference state.")

    if args.one_step:
        print("[INFO] Running one-step propagate (may still take a moment) ...")
        # NOTE: Without prompts this isn't meaningful; we just check the call path doesn't crash immediately.
        it = predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            reverse=False,
            propagate_preflight=True,
        )
        try:
            _ = next(iter(it))
            print("[OK] propagate_in_video returned at least one step.")
        except StopIteration:
            print("[OK] propagate_in_video produced no steps (likely due to no prompts), but call path is OK.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


