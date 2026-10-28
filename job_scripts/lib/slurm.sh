#!/bin/bash
# Per-stage SLURM resources and self-submission. Sourced by job_scripts/lib/common.sh.
#
# #SBATCH headers are parsed before a script runs, so they cannot depend on the
# cluster. Instead, stage scripts are started with `bash` on a login node and
# re-submit themselves via `self_submit` with the flags built here.

# Builds SBATCH_FLAGS for <stage> on $CLUSTER.
# Stages: stage1, stage2 (full sequence), stage2_part (one -part), stage2_merge, stage3.
#
# DAIC (new environment, https://doc.daic.tudelft.nl/tutorials/slurm/): GPUs by
# full type name. Stage 2 uses an RTX Pro 6000 (96 GB, Blackwell), which needs
# the CUDA 12.8 image (SIF_PATH_CU128, selected in stage2_sam3body.sh). Stage 3
# stays on an A40 (48 GB) with the cu118 image. Override the GPU with e.g.
# -sbatch "--gres=gpu:nvidia_l40:1". Memory is estimated for a cosilab batch
# (960x540, 18000 frames, 23 people): stage 3 keeps one T*N x 18439 x 3 float32
# vertex tensor (~92 GB) in RAM. Check `seff <jobid>` and adjust.
# Partition "all" (the department partition ewi-insy is not open to this user).
# Account: $SB4D_ACCOUNT, default daic-eemcs-insy-prb (verified with sbatch --test-only).
sbatch_flags() {
  local stage="$1" time partition cpus mem_per_cpu daic_time daic_cpus daic_mem daic_gpu
  case "$stage" in
    #             DelftBlue                                                     DAIC
    stage1)       time=6:00:00;  partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=""; daic_cpus=""; daic_mem=""; daic_gpu="" ;;
    stage2)       time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=3-00:00:00; daic_cpus=4; daic_mem=32G;  daic_gpu=nvidia_rtx_pro_6000 ;;
    stage2_part)  time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=1-00:00:00; daic_cpus=4; daic_mem=32G;  daic_gpu=nvidia_rtx_pro_6000 ;;
    stage2_merge) time=2:00:00;  partition=gpu-a100; cpus=16; mem_per_cpu=8000M; daic_time=2:00:00;    daic_cpus=2; daic_mem=32G;  daic_gpu=nvidia_rtx_pro_6000 ;;
    stage3)       time=18:00:00; partition=gpu;      cpus=24; mem_per_cpu=4000M; daic_time=1-00:00:00; daic_cpus=8; daic_mem=128G; daic_gpu=nvidia_a40 ;;
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
    SBATCH_FLAGS+=(
      --account="${SB4D_ACCOUNT:-daic-eemcs-insy-prb}"
      --time="$daic_time"
      --partition=all
      --cpus-per-task="$daic_cpus"
      --mem="$daic_mem"
      --gres="gpu:$daic_gpu:1"
    )
  fi
}

# On a login node: submit <script> with <stage> flags and the original args,
# then exit. Inside a SLURM job (or an interactive allocation)
# it returns so the caller does the work.
# With SB4D_DRY_RUN=1 it prints the sbatch command and returns, so the caller
# prints the container commands too.
self_submit() {
  local stage="$1" script="$2"
  shift 2
  [ -n "${SLURM_JOB_ID:-}" ] && return 0

  sbatch_flags "$stage"

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
