#!/bin/bash
# Runs a command of this repo in the cluster's apptainer image on this machine (GPU if there is one), e.g. the
# multi-view pipeline on a development clip:
#   bash scripts/local_sif.sh python stage2_sam3body.py --input <Stage 1 dir> --output-dir experiments/... --fisheye
#   bash scripts/local_sif.sh python stage3_mvfit.py <raw dirs> --stage1-root ... --out-dir experiments/...
# Mounts: the repo at /mnt/sam-body4d (working dir), the checkpoints (SB4D_CKPT) at /mnt/sam4d_checkpoints, and the
# cosilab project and Video-OnlineHMR side-camera timestamps read-only at their own paths (utils/cosilab_camera.py),
# and the multi-view root (SB4D_MV_ROOT, the DAIC results) read-only at its own path, e.g. for
# scripts/export_mvfit_vertices.py.
# Write local results under experiments/ (gitignored); on /tudelft.net only under the multi-view root (MV_ROOT in
# job_scripts/lib/dataset.sh).
# The host's GL libraries that --nv mounts can need a newer glibc than the image has, so the image's own are put first.
# Env: SB4D_SIF (default: the cu118 image, which still supports older GPUs), SB4D_CKPT.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEON=/tudelft.net/staff-umbrella/neon
SIF="${SB4D_SIF:-$NEON/apptainer/body4d_osmesa.sif}"
CKPT="${SB4D_CKPT:-$NEON/models/sam4d_checkpoints}"
MV_ROOT="${SB4D_MV_ROOT:-/tudelft.net/staff-umbrella/CaptureLab/neon/sam-body4d}"
MV_BIND=()
[ -d "$MV_ROOT" ] && MV_BIND=(--bind "$MV_ROOT:$MV_ROOT:ro")
NV=()
nvidia-smi -L >/dev/null 2>&1 && NV=(--nv)
mkdir -p "$REPO/experiments/tmp"
cd "$REPO"
exec apptainer exec ${NV[@]+"${NV[@]}"} --pwd /mnt/sam-body4d --bind "$REPO:/mnt/sam-body4d" \
  --bind "$CKPT:/mnt/sam4d_checkpoints:ro" ${MV_BIND[@]+"${MV_BIND[@]}"} \
  --bind "$NEON/cosilab_project:$NEON/cosilab_project:ro" \
  --bind "/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload:/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload:ro" \
  --bind "$NEON/experiments/Video-OnlineHMR/cosilab_side_timestamps:$NEON/experiments/Video-OnlineHMR/cosilab_side_timestamps:ro" \
  --env TMPDIR=/mnt/sam-body4d/experiments/tmp --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
  --env PYTHONPATH=/mnt/sam-body4d/models/sam3:/mnt/sam-body4d --env PYOPENGL_PLATFORM=osmesa \
  "$SIF" "$@"
