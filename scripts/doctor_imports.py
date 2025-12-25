#!/usr/bin/env python3
"""
Small diagnostics helper for container / cluster runs.

This is intentionally dependency-light and safe to run during Apptainer builds.
"""

from __future__ import annotations

import os
import sys
import traceback


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        mod = sys.modules[name]
        path = getattr(mod, "__file__", None)
        print(f"[OK] import {name} ({path})")
        return True
    except Exception as e:
        print(f"[FAIL] import {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    print("[INFO] python:", sys.executable)
    print("[INFO] cwd:", os.getcwd())
    print("[INFO] sys.path:")
    for p in sys.path:
        print("  -", p)

    ok = True
    ok &= _try_import("sam3")
    ok &= _try_import("models")
    ok &= _try_import("models.sam3.sam3.model_builder")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


