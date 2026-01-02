#!/usr/bin/env python3
"""
Minimal repro for: RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'

This isolates the failing component (MHR TorchScript / Momentum MHR) without
loading SAM3, Diffusion-VAS, or the full SAM-3D-Body pipeline.

Usage (on GPU node):
  python scripts/test_mhr_bf16_issue.py --mhr /path/to/mhr_model.pt --device cuda

If you run inside Apptainer, remember --nv.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch


def _load_mhr(mhr_path: str, device: torch.device) -> torch.nn.Module:
    # Prefer Momentum if available (same as repo logic), else fall back to torchscript.
    momentum_enabled = os.environ.get("MOMENTUM_ENABLED") is None
    if momentum_enabled:
        try:
            from mhr.mhr import MHR  # type: ignore

            print("[INFO] Using Momentum MHR (mhr.mhr.MHR.from_files)")
            mhr = MHR.from_files(device=device, lod=1)
            return mhr
        except Exception:
            print("[WARN] Momentum import failed; falling back to torch.jit.load")
            traceback.print_exc()

    print("[INFO] Using torch.jit.load on:", mhr_path)
    return torch.jit.load(mhr_path, map_location=device.type)


def _run_once(mhr: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    # The exact shapes aren’t critical to reproduce the BF16 sparse op failure.
    # We choose small batch sizes to keep it fast.
    B = 1
    # NOTE: the TorchScript MHR model expects specific feature lengths.
    # For the public MHR model shipped with SAM-3D-Body, the common expected sizes are:
    # - identity_coeffs (shape): 45
    # - model_parameters: 204  (so cat([model_parameters, zeros_like(identity)]) => 249)
    # - expression_coeffs: 72
    #
    # If these defaults don't match your checkpoint, pass --shape-dim/--model-dim/--expr-dim.
    shape_params = torch.zeros(B, _run_once.shape_dim, device=device, dtype=dtype)
    model_params = torch.zeros(B, _run_once.model_dim, device=device, dtype=dtype)
    expr_params = torch.zeros(B, _run_once.expr_dim, device=device, dtype=dtype)

    print(f"[INFO] Forward with dtype={dtype} ...")
    with torch.no_grad():
        out = mhr(shape_params, model_params, expr_params)
    if isinstance(out, (tuple, list)):
        shapes = [getattr(x, "shape", None) for x in out]
    else:
        shapes = getattr(out, "shape", None)
    print("[OK] Forward succeeded. Output shapes:", shapes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhr", required=True, help="Path to mhr_model.pt (TorchScript)")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (cuda recommended to reproduce the error)",
    )
    parser.add_argument("--shape-dim", type=int, default=45, help="Identity/shape coeff dim (default: 45)")
    parser.add_argument("--model-dim", type=int, default=204, help="Model parameters dim (default: 204)")
    parser.add_argument("--expr-dim", type=int, default=72, help="Expression coeff dim (default: 72)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("[INFO] python:", sys.executable)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] device:", device)
    print("[INFO] torch.cuda.is_available:", torch.cuda.is_available())
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[FAIL] CUDA requested but not available. Did you forget apptainer --nv or request a GPU node?")
        return 2

    mhr = _load_mhr(args.mhr, device)
    mhr.eval()

    # Thread dims into _run_once without changing call sites too much.
    _run_once.shape_dim = args.shape_dim  # type: ignore[attr-defined]
    _run_once.model_dim = args.model_dim  # type: ignore[attr-defined]
    _run_once.expr_dim = args.expr_dim  # type: ignore[attr-defined]

    # 1) Prove FP32 works (baseline)
    try:
        _run_once(mhr, device, torch.float32)
    except Exception:
        print("[FAIL] FP32 forward failed (unexpected).")
        traceback.print_exc()
        print(
            "[HINT] If you see an einsum size mismatch like 252 vs 249, your --model-dim is wrong.\n"
            "       For example, if expected_n=249 and shape_dim=45, use --model-dim 204."
        )
        return 3

    # 2) Try BF16 (expected failure on CUDA due to sparse op)
    if device.type == "cuda":
        try:
            _run_once(mhr, device, torch.bfloat16)
            print("[WARN] BF16 forward succeeded on this setup; you may not hit the original error.")
        except Exception as e:
            print("[EXPECTED FAIL] BF16 forward failed.")
            print("Exception:", type(e).__name__, e)
            traceback.print_exc()
            return 0
    else:
        print("[INFO] CPU selected; BF16 sparse CUDA error won't reproduce on CPU.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


