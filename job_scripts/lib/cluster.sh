#!/bin/bash
# Cluster detection and host paths. Sourced by job_scripts/lib/common.sh.
#
# DelftBlue is detected by the presence of /scratch/zli33; anything else is DAIC.
# Override with SB4D_CLUSTER=delftblue|daic, and the dataset base dir with
# SB4D_DATA_BASE (e.g. for dry runs against a local copy). On DAIC, cosilab
# Stage 2/3 folders live under SB4D_COSILAB_PROJECT (see lib/dataset.sh).
#
# Container mount points are identical on both clusters so paths stored in
# outputs (e.g. raw_mhr.pt meta.input_dir) stay valid across clusters:
#   checkpoints -> /mnt/sam4d_checkpoints   (matches ckpt_root in configs/body4d.yaml)
#   dataset     -> /mnt/data/<dataset>
#   home        -> /mnt/home/zli33

detect_cluster() {
  if [ -d /scratch/zli33 ]; then
    echo delftblue
  else
    echo daic
  fi
}

CLUSTER="${SB4D_CLUSTER:-$(detect_cluster)}"

case "$CLUSTER" in
  delftblue)
    HOME_HOST=/home/zli33
    DATA_BASE_HOST=/scratch/zli33/data
    MODEL_HOST=/scratch/zli33/models/sam4d_checkpoints
    SIF_PATH=/scratch/zli33/apptainers/body4d_osmesa.sif
    LOG_DIR=/scratch/zli33/slurm_outputs/sam_4d_body
    ;;
  daic)
    NEON_HOST=/tudelft.net/staff-umbrella/neon
    HOME_HOST=/home/nfs/zli33
    DATA_BASE_HOST=$NEON_HOST/zonghuan/data
    MODEL_HOST=$NEON_HOST/models/sam4d_checkpoints
    SIF_PATH=$NEON_HOST/apptainer/body4d_osmesa.sif
    LOG_DIR=$HOME_HOST/slurm_outputs/sam_4d_body
    COSILAB_PROJECT_HOST="${SB4D_COSILAB_PROJECT:-$NEON_HOST/cosilab_project}"
    ;;
  *)
    echo "[ERROR] Unknown cluster: $CLUSTER (expected delftblue or daic)" >&2
    exit 2
    ;;
esac

DATA_BASE_HOST="${SB4D_DATA_BASE:-$DATA_BASE_HOST}"

BIND_HOME=/mnt/home/zli33
BIND_MODEL=/mnt/sam4d_checkpoints
