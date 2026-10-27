#!/bin/bash
# Per-stage SLURM resources and self-submission. Sourced by job_scripts/lib/common.sh.
#
# #SBATCH headers are parsed before a script runs, so they cannot depend on the
# cluster. Instead, stage scripts are started with `bash` on a login node and
# re-submit themselves via `self_submit` with the flags built here.

# Builds SBATCH_FLAGS for <stage> on $CLUSTER.
sbatch_flags() {
  local stage="$1" time partition cpus mem_per_cpu
  case "$stage" in
    stage1) time=6:00:00;  partition=gpu-a100; cpus=16; mem_per_cpu=8000M ;;
    stage2) time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M ;;
    stage3) time=18:00:00; partition=gpu;      cpus=24; mem_per_cpu=4000M ;;
    *) echo "[ERROR] Unknown stage: $stage" >&2; exit 2 ;;
  esac

  SBATCH_FLAGS=(
    --job-name="sb4d_${stage}_${DATASET}"
    --time="$time"
    --ntasks=1
    --mail-type=END
    --output="$LOG_DIR/slurm_%j.out"
    --error="$LOG_DIR/slurm_%j.err"
  )

  if [ "$CLUSTER" = delftblue ]; then
    SBATCH_FLAGS+=(
      --account=research-eemcs-insy
      --partition="$partition"
      --cpus-per-task="$cpus"
      --mem-per-cpu="$mem_per_cpu"
      --gpus-per-task=1
    )
  else
    SBATCH_FLAGS+=(
      --partition=insy,general
      --qos="$(daic_qos "$time")"
      --cpus-per-task=8
      --mem=64G
      --gres=gpu:a40:1
    )
  fi
}

# DAIC QoS from a H:MM:SS limit: short <=4h, medium <=36h, long otherwise.
daic_qos() {
  local hours="${1%%:*}"
  if (( 10#$hours < 4 )) || [ "$1" = "4:00:00" ]; then
    echo short
  elif (( 10#$hours < 36 )) || [ "$1" = "36:00:00" ]; then
    echo medium
  else
    echo long
  fi
}

# On a login node: snapshot the code, submit <script> with <stage> flags and the
# original args, then exit. Inside a SLURM job (or an interactive allocation)
# it returns so the caller does the work.
# With SB4D_DRY_RUN=1 it prints the sbatch command and returns, so the caller
# prints the container commands too.
self_submit() {
  local stage="$1" script="$2"
  shift 2
  [ -n "${SLURM_JOB_ID:-}" ] && return 0

  sbatch_flags "$stage"
  make_snapshot

  local extra=()
  [ -n "$EXTRA_SBATCH" ] && read -r -a extra <<< "$EXTRA_SBATCH"

  local cmd=(sbatch "${SBATCH_FLAGS[@]}" ${extra[@]+"${extra[@]}"}
    --export=ALL,SB4D_CODE="$SB4D_CODE",SB4D_CLUSTER="$CLUSTER"
    "$script" "$@")

  if [ "${SB4D_DRY_RUN:-0}" = 1 ]; then
    echo "[DRY-RUN] $(printf '%q ' "${cmd[@]}")"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  Sequences: ${SEQS[*]}"
  "${cmd[@]}"
  exit 0
}
