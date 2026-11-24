#!/bin/bash
# Stage 3, multi-view (cosilab, DAIC): one world-frame body per person fitted to all cameras of a session and batch
# (stage3_mvfit.py) -> $MVFIT_DIR/session<S>_batch<BB>[_f<a>-<b>]/ under the multi-view root (lib/dataset.sh
# MV_ROOT): world/<pid>.npz, cameras/<seq>/ (run options, extrinsics, reproj_overlay.mp4), summary.json.
#
# Needs Stage 2 -fisheye output ($FISHEYE_RAW_DIR/<seq>/raw_mhr.pt; job_scripts/stage2_sam3body.sh -fisheye) of every
# camera taking part. Cameras: session 1 = cam06-10, session 2 = cam01-04 + side cams 12-15 (cam05 has no
# extrinsics). By default every camera of the session with Stage 2 -fisheye output takes part, except cameras whose
# masklet ID check (scripts/check_masklet_ids.py, id_check.json in their Stage 1 folder) rejected them.
#
# Run with bash on a DAIC login node; the script submits itself to SLURM (one GPU job per batch and session).
#
# Usage:
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2 -cams 01,03 -start 6000 -num_frames 600
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2 -cams 01,02,03,04 -holdout 02     # leave-one-camera-out
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2 -- --iters 300 --w-fill 5        # stage3_mvfit.py args
# Full batches (~2 GPU-h per person on a GTX 1080 Ti): split the people over an array job, then merge:
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2 -parts 8                           # -> job <id>, 8 tasks
#   bash job_scripts/stage3_mvfit.sh -b 1 -session 2 -merge -sbatch "--dependency=afterok:<id>"
#
# Options:
#   -b <spec>              batches: 1 | 1,3,5 | 3-5 (required)
#   -session <1|2>         session (required)
#   -cams <MM,MM,...>      cameras (default: all of the session with Stage 2 -fisheye output and not rejected)
#   -holdout <MM,...>      cameras evaluated but not fitted
#   -start <f> -num_frames <n>   batch frames [f, f + n) only (development clips)
#   -label <name>          output folder name (default session<S>_batch<BB>[_f<a>-<b>])
#   -parts <N>             array job of N tasks, task K fits every N-th person from K (stage3_mvfit.py --part K/N);
#                          one batch only. Then -merge (same other options) collects them, fits the floor, exports.
#   -merge                 the merge step of -parts (no fitting)
#   -video                 also render the overlay videos in the fit job (default: no; slow, more than an hour per
#                          camera for 600 frames on 8 CPUs)
#   -render                CPU job: only the overlay videos of an existing fit (cameras/<seq>/reproj_overlay.mp4)
#   -sbatch "<flags>"      extra sbatch flags
#   -- <args>              passed on to stage3_mvfit.py
set -euo pipefail

LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
SB4D_CODE="${SB4D_CODE:-$(cd "$(dirname "$LAUNCHER")/.." && pwd)}"
if [ ! -f "$SB4D_CODE/job_scripts/lib/common.sh" ]; then
  echo "[ERROR] job_scripts/lib not found under $SB4D_CODE. Start this script with bash, not sbatch." >&2
  exit 2
fi
source "$SB4D_CODE/job_scripts/lib/common.sh"

ORIG_ARGS=("$@")
SESSION=""
CAMS=""
HOLDOUT=""
START=""
NUM_FRAMES=""
LABEL=""
PARTS=""
MERGE=0
VIDEO=0
RENDER=0
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -session|--session)       SESSION="${2:-}"; shift 2 ;;
    -cams|--cams)             CAMS="${2:-}"; shift 2 ;;
    -holdout|--holdout)       HOLDOUT="${2:-}"; shift 2 ;;
    -start|--start)           START="${2:-}"; shift 2 ;;
    -num_frames|--num_frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    -label|--label)           LABEL="${2:-}"; shift 2 ;;
    -parts|--parts)           PARTS="${2:-}"; shift 2 ;;
    -merge|--merge)           MERGE=1; shift ;;
    -video|--video)           VIDEO=1; shift ;;
    -render|--render)         RENDER=1; shift ;;
    --)                       shift; PY_ARGS=("$@"); break ;;
    *)                        parse_common_args "$@"; shift "$COMMON_SHIFT" ;;
  esac
done

load_dataset "$DATASET_ARG"
load_mv_dirs
[ -n "$BATCH_SPEC" ] || die "-b <spec> is required (see -h)"
[[ "$SESSION" =~ ^[12]$ ]] || die "-session 1|2 is required"
[ -z "$CAM_NUM$ALL_PREFIX" ] && [ ${#POS_SEQS[@]} -eq 0 ] || die "Give batches with -b and cameras with -cams"
[ -z "$START$NUM_FRAMES" ] || { [[ "$START" =~ ^[0-9]+$ ]] && [[ "$NUM_FRAMES" =~ ^[1-9][0-9]*$ ]]; } \
  || die "-start and -num_frames go together (non-negative integers)"
expand_batch_spec "$BATCH_SPEC"
if [ "$SESSION" = 1 ]; then SESSION_CAMS="06 07 08 09 10"; else SESSION_CAMS="01 02 03 04 12 13 14 15"; fi
if [ -n "$CAMS" ]; then SESSION_CAMS="${CAMS//,/ }"; fi

SEQS=()
for bn in "${BATCH_NUMS[@]}"; do SEQS+=("$(printf 'session%d_batch%02d' "$SESSION" "$bn")"); done
if [ -n "$PARTS" ]; then
  [[ "$PARTS" =~ ^[1-9][0-9]*$ ]] || die "-parts must be a positive integer"
  [ "$MERGE" = 0 ] || die "-parts and -merge are separate submissions"
  [ ${#BATCH_NUMS[@]} -eq 1 ] || die "-parts needs exactly one batch"
  EXTRA_SBATCH="--array=0-$((PARTS - 1)) $EXTRA_SBATCH"
fi
STAGE=stage3_mvfit
if [ "$RENDER" = 1 ]; then
  STAGE=mv_render
  CONTAINER_GPU=0
else
  use_gpu_image  # CUDA 12.8 image for the RTX Pro 6000, 11.8 for SB4D_GPU=nvidia_a40/l40
fi
self_submit "$STAGE" "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  code: $SB4D_CODE  session $SESSION  batches ${BATCH_NUMS[*]}"
echo "[INFO] Stage 2 -fisheye: $FISHEYE_RAW_DIR   output: $MVFIT_DIR"
failed=0
for bn in "${BATCH_NUMS[@]}"; do
  inputs=()
  holdout=()
  for c in $SESSION_CAMS; do
    cam=$(printf '%02d' "$((10#$c))")
    seq="cam${cam}_batch$(printf %02d "$bn")"
    raw="$FISHEYE_RAW_DIR/$seq/raw_mhr.pt"
    if [ ! -f "$raw" ]; then
      echo "[INFO] $seq: no Stage 2 -fisheye output, left out"
      continue
    fi
    check="$(seq_stage1_dir "$seq")/id_check.json"
    if [ -f "$check" ] && grep -q '"accepted": false' "$check"; then
      echo "[WARN] $seq: rejected by the masklet ID check ($check), left out" >&2
      continue
    fi
    inputs+=("$(host_to_container "$FISHEYE_RAW_DIR/$seq")")
    for h in ${HOLDOUT//,/ }; do
      [ "$((10#$h))" = "$((10#$cam))" ] && holdout+=("cam$cam")
    done
  done
  if [ ${#inputs[@]} -lt 1 ]; then
    echo "[WARN] batch $bn: no camera can take part" >&2
    failed=$((failed + 1))
    continue
  fi
  label="${LABEL:-$(printf 'session%d_batch%02d' "$SESSION" "$bn")}"
  CLIP=()
  if [ -n "$START" ]; then
    CLIP=(--start "$START" --num-frames "$NUM_FRAMES")
    [ -n "$LABEL" ] || label+="_f${START}-$((START + NUM_FRAMES))"
  fi
  out="$MVFIT_DIR/$label"
  guard_mv_output "$out"
  [ "${SB4D_DRY_RUN:-0}" = 1 ] || mkdir -p "$out"
  echo "=================================================================="
  echo "[INFO] batch $bn: ${#inputs[@]} cameras -> $out"
  echo "=================================================================="
  HOLD=()
  [ ${#holdout[@]} -eq 0 ] || HOLD=(--holdout "${holdout[@]}")
  if [ -n "$PARTS" ]; then
    HOLD+=(--part "${SLURM_ARRAY_TASK_ID:-0}/$PARTS")
  elif [ "$MERGE" = 1 ]; then
    HOLD+=(--merge)
  fi
  if [ "$RENDER" = 1 ]; then
    dirs=()
    for d in "$out"/cameras/*/; do dirs+=("$(host_to_container "${d%/}")"); done
    [ ${#dirs[@]} -gt 0 ] || { echo "[ERROR] no cameras/ in $out" >&2; failed=$((failed + 1)); continue; }
    if run_in_container scripts/render_overlay_video.py "${dirs[@]}"; then echo "[INFO] videos done."
    else echo "[ERROR] videos FAILED." >&2; failed=$((failed + 1)); fi
    continue
  fi
  [ "$VIDEO" = 1 ] || HOLD+=(--no-video)
  if run_in_container stage3_mvfit.py "${inputs[@]}" --out-dir "$(host_to_container "$out")" --config "$CONFIG" \
      ${CLIP[@]+"${CLIP[@]}"} ${HOLD[@]+"${HOLD[@]}"} ${PY_ARGS[@]+"${PY_ARGS[@]}"}; then
    echo "[INFO] batch $bn done."
  else
    echo "[ERROR] batch $bn FAILED." >&2
    failed=$((failed + 1))
  fi
done
echo "[INFO] All done, $failed failure(s)."
[ "$failed" -eq 0 ]
