# gpu_profiler.py
#
# Unified profiler for PyTorch:
# - Measures time (auto formats: seconds / minutes / hours)
# - Measures GPU memory usage (alloc / reserved / peak / delta)
#
# Usage:
#   from gpu_profiler import gpu_profile
#   on_mask_generation = gpu_profile(on_mask_generation)
#   on_4d_generation   = gpu_profile(on_4d_generation)

"""
Includes:
- `gpu_profile` decorator for quick profiling of function runtime + CUDA memory.
- Lightweight helpers for scripts to log peak CUDA memory usage to JSON.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import torch


def _fmt_mem(bytes_val: int) -> str:
    """Format bytes to human-readable GB."""
    return f"{bytes_val / (1024 ** 3):.2f} GB"


def _fmt_time(sec: float) -> str:
    """
    Smart human-readable time formatter:
    - < 60 sec → "xx.xx s"
    - < 1 hour → "Xm Ys"
    - ≥ 1 hour → "Xh Ym Zs"
    """
    if sec < 60:
        return f"{sec:.2f} s"
    elif sec < 3600:
        m = int(sec // 60)
        s = sec % 60
        return f"{m:d}m {s:.1f}s"
    else:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:d}h {m:d}m {s:.1f}s"


def cuda_reset_peak_memory_stats() -> None:
    """Reset CUDA peak memory stats if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_mem_snapshot() -> Dict[str, Any]:
    """
    Return a JSON-serializable snapshot of CUDA peak memory usage.

    Keys match what our stage scripts historically wrote (bytes).
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "device": str(torch.cuda.get_device_name(0)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write dict as pretty JSON, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def gpu_profile(fn):
    """
    Decorator: Profile both GPU memory usage and runtime.
    Works even if CUDA is not available (time only).
    """
    def wrapped(*args, **kwargs):
        # ------------------ CPU ONLY MODE ------------------
        if not torch.cuda.is_available():
            t0 = time.time()
            out = fn(*args, **kwargs)
            t1 = time.time()
            print(f"[TIME PROF][{fn.__name__}] runtime={_fmt_time(t1 - t0)}")
            return out

        # ------------------ GPU MEMORY BASELINE ------------------
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        base_reserved = torch.cuda.memory_reserved()

        torch.cuda.reset_peak_memory_stats()

        # ------------------ RUN FUNCTION ------------------
        t0 = time.time()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.time()

        # ------------------ AFTER METRICS ------------------
        end_alloc = torch.cuda.memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        delta_peak = peak_alloc - base_alloc

        # ------------------ REPORT ------------------
        print(
            f"[PROF][{fn.__name__}] "
            f"time={_fmt_time(t1 - t0)},  "
            f"alloc={_fmt_mem(end_alloc)}, "
            f"reserved={_fmt_mem(end_reserved)}, "
            f"peak={_fmt_mem(peak_alloc)}, "
            f"delta_peak={_fmt_mem(delta_peak)}"
        )

        return out

    wrapped.__name__ = f"{fn.__name__}_gpu_profiled"
    return wrapped
