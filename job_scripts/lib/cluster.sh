#!/bin/bash
# Cluster detection and host paths. Sourced by job_scripts/lib/common.sh.
#
# DelftBlue is detected by the presence of /scratch/zli33; anything else is DAIC.
# Override with SB4D_CLUSTER=delftblue|daic, and the dataset base dir with
# SB4D_DATA_BASE (e.g. for dry runs against a local copy). On DAIC, cosilab
# Stage 2/3 folders live under SB4D_COSILAB_PROJECT (see lib/dataset.sh).
#
# Jobs always run the single checkout at CODE_HOST (no code snapshots); the
# launchers refuse to start from any other copy. SB4D_CODE_HOST overrides it
# (e.g. for dry runs elsewhere). On DAIC it depends on the user: zli33's home, or
# ~/linuxhome/sam-body4d for everyone else.
#
# Container mount points are identical on both clusters so paths stored in
# outputs (e.g. raw_mhr.pt meta.input_dir) stay valid across clusters:
#   checkpoints -> /mnt/sam4d_checkpoints   (matches ckpt_root in configs/body4d.yaml)
#   dataset     -> /mnt/data/<dataset>
#   code        -> /mnt/sam-body4d

detect_cluster() {
  if [ -d /scratch/zli33 ]; then
    echo delftblue
  else
    echo daic
  fi
}

CLUSTER="${SB4D_CLUSTER:-$(detect_cluster)}"

# Multi-view pipeline output root on DAIC (lib/dataset.sh).
MV_ROOT="${SB4D_MV_ROOT:-/tudelft.net/staff-umbrella/CaptureLab/neon/sam-body4d}"

case "$CLUSTER" in
  delftblue)
    CODE_HOST=/home/zli33/projects/sam-body4d
    DATA_BASE_HOST=/scratch/zli33/data
    MODEL_HOST=/scratch/zli33/models/sam4d_checkpoints
    SIF_PATH=/scratch/zli33/apptainers/body4d_osmesa.sif
    LOG_DIR=/scratch/zli33/slurm_outputs/sam_4d_body
    ;;
  daic)
    NEON_HOST=/tudelft.net/staff-umbrella/neon
    if [ "${USER:-}" = zli33 ]; then
      CODE_HOST=/home/zli33/linuxhome/projects/sam-body4d
      LOG_DIR=/home/zli33/linuxhome/slurm_outputs/sam_4d_body
    else
      # e.g. edortaperez: checkout in the user's linuxhome, logs in its gitignored logs/ (set below)
      CODE_HOST=$HOME/linuxhome/sam-body4d
      LOG_DIR=""
    fi
    DATA_BASE_HOST=$NEON_HOST/zonghuan/data
    MODEL_HOST=$NEON_HOST/models/sam4d_checkpoints
    SIF_PATH=$NEON_HOST/apptainer/body4d_osmesa.sif
    # torch cu128 build (apptainer/build.sh cu128) for Blackwell GPUs (RTX Pro 6000)
    SIF_PATH_CU128=$NEON_HOST/apptainer/body4d_osmesa_cu128.sif
    COSILAB_PROJECT_HOST="${SB4D_COSILAB_PROJECT:-$NEON_HOST/cosilab_project}"
    # /home/zli33 is ~5 MB (config files only), so the container gets a job-local
    # home on the node's /tmp for matplotlib/torch/CUDA caches (removed at exit).
    CONTAINER_HOME_HOST="${TMPDIR:-/tmp/$USER}/sb4d_home_${SLURM_JOB_ID:-$$}"
    ;;
  *)
    echo "[ERROR] Unknown cluster: $CLUSTER (expected delftblue or daic)" >&2
    exit 2
    ;;
esac

DATA_BASE_HOST="${SB4D_DATA_BASE:-$DATA_BASE_HOST}"
CODE_HOST="${SB4D_CODE_HOST:-$CODE_HOST}"
LOG_DIR="${LOG_DIR:-$CODE_HOST/logs}"

BIND_CODE=/mnt/sam-body4d
BIND_MODEL=/mnt/sam4d_checkpoints
