#!/bin/bash
# Builds the body4d apptainer image from a clean export of the committed repo
# (git archive HEAD), so untracked data such as experiments/ is never copied in
# by the def's `%files . /opt/sam-body4d`. The def itself is taken from the
# working tree.
#
# Usage: bash apptainer/build.sh <cu118|cu128> [out.sif]
#   cu118  torch 2.7.1+cu118 -> body4d_osmesa.sif        A40 / L40 / A100
#   cu128  torch 2.7.1+cu128 -> body4d_osmesa_cu128.sif  also RTX Pro 6000 (Blackwell, sm_120)
# Default output is apptainer/<name>.sif (gitignored). Needs apptainer >= 1.2
# (--build-arg) and fakeroot (an /etc/subuid entry).
set -euo pipefail

TORCH_CUDA="${1:?usage: bash apptainer/build.sh <cu118|cu128> [out.sif]}"
case "$TORCH_CUDA" in
  cu118) name=body4d_osmesa.sif ;;
  cu128) name=body4d_osmesa_cu128.sif ;;
  *) echo "[ERROR] Unknown CUDA tag: $TORCH_CUDA (expected cu118 or cu128)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${2:-$REPO/apptainer/$name}")"
DEF="$REPO/apptainer/body4d.def"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git -C "$REPO" archive HEAD | tar -x -C "$CTX"

echo "[INFO] Building $OUT (torch $TORCH_CUDA) from commit $(git -C "$REPO" rev-parse --short HEAD)"
cd "$CTX"
apptainer build --fakeroot --build-arg TORCH_CUDA="$TORCH_CUDA" "$OUT" "$DEF"
