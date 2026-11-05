#!/bin/bash
# Per-stage SLURM resources and self-submission. Sourced by job_scripts/lib/common.sh.
#
# #SBATCH headers are parsed before a script runs, so they cannot depend on the
# cluster. Instead, stage scripts are started with `bash` on a login node and
# re-submit themselves via `self_submit` with the flags built here.

# Builds SBATCH_FLAGS for <stage> on $CLUSTER.
# Stages: stage1, stage2 (full sequence), stage2_part (one -part), stage2_merge, stage3,
# stage3_multiview.
#
# DAIC (new environment, https://doc.daic.tudelft.nl/tutorials/slurm/): GPUs by
# full type name. Stage 2 uses an RTX Pro 6000 (96 GB, Blackwell), which needs
# the CUDA 12.8 image (SIF_PATH_CU128, selected in stage2_sam3body.sh). Override
# the GPU with e.g. -sbatch "--gres=gpu:nvidia_l40:1".
# stage2_merge and stage3 are CPU only (empty daic_gpu; DelftBlue partition
# "compute"): the merge only concatenates part files, and stage 3 runs the MHR
# forward pass and the small cam_t optimizations on CPU. Their launchers set
# CONTAINER_GPU=0. Memory is estimated for a cosilab batch (960x540, 18000
# frames, 23 people): stage 3 works one person at a time; its peak is the
# sequence's masks (~8 GB) + one person's soft masks (~8 GB) and meshes
# (~4 GB, x2 while exporting). Check `seff <jobid>` and adjust.
# Time limits: one Stage 1 job and one Stage 2 part (6000 frames) finish within
# 20 h; Stage 3 on CPU took ~70 s for 3000 person-frames locally, ~1-2 h
# extrapolated to a full cosilab batch (~200k person-frames), so 8 h leaves room.
# stage3_multiview runs Stage 3 (camera space) for up to 5 cameras one after the
# other, then the multi-view step (reads/writes every mesh once): 24 h.
# Partition "all" (the department partition ewi-insy is not open to this user).
# Account: $SB4D_ACCOUNT, default daic-eemcs-insy-prb (verified with sbatch --test-only).
sbatch_flags() {
  local stage="$1" time partition cpus mem_per_cpu daic_time daic_cpus daic_mem daic_gpu
  case "$stage" in
    #             DelftBlue                                                      DAIC
    stage1)       time=20:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M;  daic_time=""; daic_cpus=""; daic_mem=""; daic_gpu="" ;;
    stage2)       time=28:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M;  daic_time=3-00:00:00; daic_cpus=4;  daic_mem=32G;  daic_gpu=nvidia_rtx_pro_6000 ;;
    stage2_part)  time=20:00:00; partition=gpu-a100; cpus=16; mem_per_cpu=8000M;  daic_time=20:00:00;   daic_cpus=4;  daic_mem=32G;  daic_gpu=nvidia_rtx_pro_6000 ;;
    stage2_merge) time=0:30:00;  partition=compute;  cpus=2;  mem_per_cpu=16000M; daic_time=0:30:00;    daic_cpus=2;  daic_mem=32G;  daic_gpu="" ;;
    stage3)       time=8:00:00;  partition=compute;  cpus=16; mem_per_cpu=4000M;  daic_time=8:00:00;    daic_cpus=16; daic_mem=64G;  daic_gpu="" ;;
    stage3_multiview) time=24:00:00; partition=compute; cpus=16; mem_per_cpu=4000M; daic_time=24:00:00; daic_cpus=16; daic_mem=64G; daic_gpu="" ;;
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
    )
    [ "$partition" = compute ] || SBATCH_FLAGS+=(--gpus-per-task=1)
  else
    [ -n "$daic_time" ] || { echo "[ERROR] $stage has no DAIC resources" >&2; exit 2; }
    SBATCH_FLAGS+=(
      --account="${SB4D_ACCOUNT:-daic-eemcs-insy-prb}"
      --time="$daic_time"
      --partition=all
      --cpus-per-task="$daic_cpus"
      --mem="$daic_mem"
    )
    [ -z "$daic_gpu" ] || SBATCH_FLAGS+=(--gres="gpu:$daic_gpu:1")
  fi
}

# Slurm --time value ([D-]HH:MM:SS, [D-]HH:MM, MM:SS, MM, D-HH) -> seconds
time_to_seconds() {
  local t="$1" d=0 h=0 m=0 s=0 parts
  if [[ "$t" == *-* ]]; then
    d="${t%%-*}"; t="${t#*-}"
    IFS=: read -r -a parts <<< "$t"
    h=${parts[0]}; m=${parts[1]:-0}; s=${parts[2]:-0}
  else
    IFS=: read -r -a parts <<< "$t"
    case ${#parts[@]} in
      3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
      2) m=${parts[0]}; s=${parts[1]} ;;
      *) m=${parts[0]} ;;
    esac
  fi
  echo $(( ((10#$d * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s ))
}

# DAIC: jobs reach /tudelft.net and ~/linuxhome (data, outputs and the SLURM log
# files) only through the Kerberos ticket that AUKS takes at submission and
# renews while the job is queued or running, up to the ticket's "renew until"
# time. Without a valid ticket a job fails within seconds and leaves no log
# files; once the renewal runs out, it dies mid-run without a trace (its log
# files are on that storage too). https://doc.daic.tudelft.nl/docs/storage/storage/
# Refuses to submit unless the ticket is renewable for the job's time limit + 1 day of queueing.
check_kerberos() {
  local time_s="$1" renew renew_s left
  command -v klist >/dev/null 2>&1 || return 0
  klist -s || die "No valid Kerberos ticket. Run kinit (NetID password), then submit again."
  renew="$(klist | awk '/renew until/ { sub(/.*renew until */, ""); print; exit }')"
  [ -n "$renew" ] || die "Kerberos ticket is not renewable (no 'renew until' in klist), so jobs lose storage access within 10 h. Run kinit (NetID password), then submit again."
  if renew_s="$(date -d "$renew" +%s 2>/dev/null)"; then
    left=$(( renew_s - $(date +%s) ))
    if (( left < time_s + 86400 )); then
      die "Kerberos ticket is renewable only until $renew; the job may need $((time_s / 3600)) h + queueing. Run kinit (NetID password) for a new 7-day ticket, then submit again."
    fi
    echo "[INFO] Kerberos ticket renewable until $renew"
  else
    echo "[WARN] Could not read the Kerberos 'renew until' time; check klist (jobs lose storage access when it passes)" >&2
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
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[INFO] Job $SLURM_JOB_ID on $(hostname -s), started $(date '+%F %T')"
    if [ "$CLUSTER" = daic ] && command -v klist >/dev/null 2>&1; then
      local ticket
      ticket="$(klist 2>/dev/null | awk '/krbtgt/ { e = $3 " " $4 } /renew until/ { sub(/.*renew until */, ""); r = $0 }
                                     END { if (e != "") print "expires " e ", renewable until " r }' || true)"
      echo "[INFO] Kerberos ticket: ${ticket:-none (network storage will fail)}"
    fi
    return 0
  fi

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
  if [ "$CLUSTER" = daic ]; then
    # The last --time wins (-sbatch flags come after the stage's)
    local f time_limit=""
    for f in "${SBATCH_FLAGS[@]}" ${extra[@]+"${extra[@]}"}; do
      [[ "$f" == --time=* ]] && time_limit="${f#--time=}"
    done
    check_kerberos "$(time_to_seconds "$time_limit")"
  fi
  mkdir -p "$LOG_DIR"
  echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  Sequences: ${SEQS[*]}"
  "${cmd[@]}"
  exit 0
}
