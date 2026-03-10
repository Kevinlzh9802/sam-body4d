#!/usr/bin/env python3
"""Download keypoint data from a remote server via SSH using rsync.

This copies the full directory structure under `REMOTE_DATA_PATH` into
`LOCAL_DATA_PATH`, but skips any directory named `meshes_4d_individual`.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

LOCAL_DATA_PATH = "/home/zonghuan/tudelft/projects/datasets/conflab/bbox_kp"
REMOTE_DATA_PATH = "/scratch/zli33/data/conflab/bbox_kp"

HOST = "login.delftblue.tudelft.nl"
USER = "zli33"
PORT = 22

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download remote keypoint folders over SSH, excluding "
            "'meshes_4d_individual'."
        )
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"Remote SSH host (default: {HOST})",
    )
    parser.add_argument(
        "--user",
        default=USER,
        help=f"Remote SSH username (default: {USER})",
    )
    parser.add_argument("--port", type=int, default=PORT, help="SSH port")
    parser.add_argument(
        "--remote-path",
        default=REMOTE_DATA_PATH,
        help=f"Remote root path (default: {REMOTE_DATA_PATH})",
    )
    parser.add_argument(
        "--local-path",
        default=LOCAL_DATA_PATH,
        help=f"Local destination root path (default: {LOCAL_DATA_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without downloading",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    if shutil.which("rsync") is None:
        raise RuntimeError("`rsync` is required but was not found in PATH.")
    if shutil.which("ssh") is None:
        raise RuntimeError("`ssh` is required but was not found in PATH.")


def build_rsync_command(
    host: str,
    user: str,
    port: int,
    remote_path: str,
    local_path: str,
    dry_run: bool,
) -> list[str]:
    ssh_parts = ["ssh"]
    if port != 22:
        ssh_parts += ["-p", str(port)]
    ssh_cmd = shlex.join(ssh_parts)

    remote_spec = f"{user}@{host}:{remote_path.rstrip('/')}/"
    local_spec = f"{local_path.rstrip('/')}/"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "--exclude",
        "meshes_4d_individual/",
        "-e",
        ssh_cmd,
        remote_spec,
        local_spec,
    ]
    if dry_run:
        cmd.insert(1, "--dry-run")
    return cmd


def main() -> int:
    args = parse_args()

    try:
        ensure_dependencies()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    local_root = Path(args.local_path)
    local_root.mkdir(parents=True, exist_ok=True)

    cmd = build_rsync_command(
        host=args.host,
        user=args.user,
        port=args.port,
        remote_path=args.remote_path,
        local_path=args.local_path,
        dry_run=args.dry_run,
    )

    print("Running command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print("\nSSH will prompt for password if key-based auth is not set up.\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"rsync failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
