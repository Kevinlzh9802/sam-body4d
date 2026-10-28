#!/bin/bash
# Per-stage SLURM resources and self-submission. Sourced by job_scripts/lib/common.sh.
#
# #SBATCH headers are parsed before a script runs, so they cannot depend on the
# cluster. Instead, stage scripts are started with `bash` on a login node and
# re-submit themselves via `self_submit` with the flags built here.

# Builds SBATCH_FLAGS for <stage> on $CLUSTER.
# Stages: stage1, stage2 (full sequence), stage2_part (one -part), stage2_merge, stage3.
#
# DAIC follows https://doc.daic.tudelft.nl/tutorials/slurm/: --account is
# required, --partition=all, GPUs by full type name. The container's PyTorch
# (2.7.1+cu118) supports A40/L40 but not the RTX Pro 6000 (Blackwell) nodes, so
# an A40 (48 GB) is requested; override with -sbatch "--gres=gpu:nvidia_l40:1".
# DAIC time/memory are estimates from DelftBlue runs of a cosilab batch
# (960x540, 18000 frames, ~22 people/frame); check `seff <jobid>` and adjust.
sbatch_flags() {
  local stage="$1" time partition cpus mem_per_cpu daic_time daic_cpus daic_mem
  case "$stage" in
    #             DelftBlue                                                     DAIC
    stage1)       time=6:00:00;  partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=""; daic_cpus=""; daic_mem="" ;;
    stage2)       time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=7-00:00:00; daic_cpus=4; daic_mem=32G ;;
    stage2_part)  time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=3-00:00:00; daic_cpus=4; daic_mem=32G ;;
    stage2_merge) time=2:00:00;  partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=2:00:00;    daic_cpus=2; daic_mem=32G ;;
    stage3)       time=18:00:00; partition=gpu;      cpus=24; mem_per_cpu=4000M; daic_time=1-00:00:00; daic_cpus=8; daic_mem=250G ;;
    *) echo "[ERROR] Unknown stage: $stage" >&2; exit 2 ;;
  esac

  SBATCH_FLAGS=(
    --job-name="sb4d_${stage}_${DATASET}"
    --ntasks=1
    --mail-type=END
    --output="$LOG_DIR/slurm_%j.out"
    --error="$LOG_DIR/slurm_%j.err"
  )

  if [ "$CLUSTER" = delftblue ]; then
    SBATCH_FLAGS+=(
      --account=research-eemcs-insy
      --time="$time"
      --partition="$partition"
      --cpus-per-task="$cpus"
      --mem-per-cpu="$mem_per_cpu"
      --gpus-per-task=1
    )
  else
    [ -n "$daic_time" ] || { echo "[ERROR] $stage has no DAIC resources" >&2; exit 2; }
    local account
    account="$(daic_account)" || exit 2
    SBATCH_FLAGS+=(
      --account="$account"
      --time="$daic_time"
      --partition=all
      --cpus-per-task="$daic_cpus"
      --mem="$daic_mem"
      --gres=gpu:nvidia_a40:1
    )
  fi
}

# DAIC billing account: $SB4D_ACCOUNT, else the first of the user's Slurm
# associations (`sacctmgr show associations user=$USER`).
daic_account() {
  local account="${SB4D_ACCOUNT:-}"
  if [ -z "$account" ] && command -v sacctmgr >/dev/null 2>&1; then
    account="$(sacctmgr -n -P show associations user="$USER" format=Account 2>/dev/null | head -n1)"
  fi
  if [ -z "$account" ]; then
    if [ "${SB4D_DRY_RUN:-0}" = 1 ]; then
      echo "<account>"
      return 0
    fi
    echo "[ERROR] No DAIC account found; set SB4D_ACCOUNT=<account>" >&2
    return 1
  fi
  echo "$account"
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
