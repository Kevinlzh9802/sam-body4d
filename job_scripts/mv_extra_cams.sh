#!/bin/bash
# Multi-view pipeline (cosilab, DAIC): Stage 1 input for cameras without annotations (top 02, 04, 07, 09; side cams
# 12-15), from a stage3_mvfit.py fit of the session's annotated cameras. One step per call, each its own SLURM job:
#
#   segments  dataset/make_raw_segments.py: 10 s segments of the batch from the raw GoPro files at the session
#             frames (raw timecode tags) -> $MV_VIDEO_DIR/<seq>/                                      (CPU)
#   prompts   scripts/project_prompts.py: the fit's bodies projected into the camera at every segment start, people
#             mostly hidden by a nearer person left out -> $MV_ANNOT_DIR/<seq>/<cam>_segNNN.json         (GPU)
#   stage1    stage1_masklet.py with those prompts -> $MV_STAGE1_DIR/<seq>/                              (GPU)
#   idcheck   scripts/check_masklet_ids.py: masklets vs the projected bodies; relabels swaps, drops non-matches,
#             accepts or rejects the camera (id_check.json; stage3_mvfit.sh leaves rejected cameras out)  (GPU)
# Then: bash job_scripts/stage2_sam3body.sh -fisheye <seq> and bash job_scripts/stage3_mvfit.sh with all cameras.
#
# Run with bash on a DAIC login node; the script submits itself to SLURM. Chain the steps with
# -sbatch "--dependency=afterok:<jobid>".
#
# Usage:
#   bash job_scripts/mv_extra_cams.sh -step segments -cam 13 -b 1
#   bash job_scripts/mv_extra_cams.sh -step prompts  -cam 13 -b 1 -mvfit session2_batch01
#   bash job_scripts/mv_extra_cams.sh -step stage1   -cam 13 -b 1
#   bash job_scripts/mv_extra_cams.sh -step idcheck  -cam 13 -b 1 -mvfit session2_batch01
#
# Options:
#   -step <segments|prompts|stage1|idcheck>
#   -cam <num> -b <spec>       camera and batches (one job for all batches)
#   -mvfit <label>             fit folder under $MVFIT_DIR (prompts, idcheck); default session<S>_batch<BB>
#   -segments <spec>           1-based segment numbers (segments, prompts), e.g. 11 or 10-12; default all
#   -extrinsics <json>         replace the camera's extrinsics (prompts, idcheck), e.g. a refit of cam12/14
#   -sbatch "<flags>"          extra sbatch flags
set -euo pipefail

LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
SB4D_CODE="${SB4D_CODE:-$(cd "$(dirname "$LAUNCHER")/.." && pwd)}"
if [ ! -f "$SB4D_CODE/job_scripts/lib/common.sh" ]; then
  echo "[ERROR] job_scripts/lib not found under $SB4D_CODE. Start this script with bash, not sbatch." >&2
  exit 2
fi
source "$SB4D_CODE/job_scripts/lib/common.sh"

ORIG_ARGS=("$@")
STEP=""
MVFIT_LABEL=""
SEG_SPEC=""
EXTR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -step|--step)             STEP="${2:-}"; shift 2 ;;
    -mvfit|--mvfit)           MVFIT_LABEL="${2:-}"; shift 2 ;;
    -segments|--segments)     SEG_SPEC="${2:-}"; shift 2 ;;
    -extrinsics|--extrinsics) EXTR="${2:-}"; shift 2 ;;
    *)                        parse_common_args "$@"; shift "$COMMON_SHIFT" ;;
  esac
done

load_dataset "$DATASET_ARG"
load_mv_dirs
[ -n "$CAM_NUM" ] && [ -n "$BATCH_SPEC" ] || die "-cam <num> -b <spec> are required"
CAM=$(printf '%02d' "$((10#$CAM_NUM))")
case "$CAM" in
  01|03|05|06|08|10) die "cam$CAM has annotations; use stage1_masklet.sh" ;;
  02|04|12|13|14|15) SESSION=2 ;;
  07|09)             SESSION=1 ;;
  *)                 die "cam$CAM is not a cosilab camera without annotations" ;;
esac
case "$STEP" in
  segments) STAGE=mv_segments; CONTAINER_GPU=0 ;;
  prompts|idcheck) STAGE=mv_prompts ;;
  stage1)   STAGE=mv_stage1 ;;
  *)        die "-step must be segments, prompts, stage1 or idcheck" ;;
esac
resolve_seqs "$STAGE1_DIR"
[ "$STAGE" = mv_segments ] || SIF_PATH=$SIF_PATH_CU128  # RTX Pro 6000 (Blackwell)
self_submit "$STAGE" "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  code: $SB4D_CODE  step $STEP  sequences ${SEQS[*]}"
failed=0
EXTR_ARGS=()
[ -z "$EXTR" ] || EXTR_ARGS=(--extrinsics "cam$CAM=$(host_to_container "$EXTR")")
SEG_ARGS=()
[ -z "$SEG_SPEC" ] || SEG_ARGS=(--segments "$SEG_SPEC")
for seq in "${SEQS[@]}"; do
  bn=$((10#${seq##*batch}))
  fit="$MVFIT_DIR/${MVFIT_LABEL:-$(printf 'session%d_batch%02d' "$SESSION" "$bn")}"
  echo "=================================================================="
  echo "[INFO] $seq: $STEP"
  echo "=================================================================="
  case "$STEP" in
    segments)
      guard_mv_output "$MV_VIDEO_DIR/$seq"
      cmd=(dataset/make_raw_segments.py --cam "$CAM" --batch "$bn" --out-root "$(host_to_container "$MV_VIDEO_DIR")"
           ${SEG_ARGS[@]+"${SEG_ARGS[@]}"}) ;;
    prompts)
      guard_mv_output "$MV_ANNOT_DIR/$seq"
      cmd=(scripts/project_prompts.py --mvfit "$(host_to_container "$fit")" --cam "$CAM" --batch "$bn"
           --out-root "$(host_to_container "$MV_ANNOT_DIR")" --config "$CONFIG"
           ${SEG_ARGS[@]+"${SEG_ARGS[@]}"} ${EXTR_ARGS[@]+"${EXTR_ARGS[@]}"}) ;;
    stage1)
      guard_mv_output "$MV_STAGE1_DIR/$seq"
      [ "${SB4D_DRY_RUN:-0}" = 1 ] || mkdir -p "$MV_STAGE1_DIR/$seq"
      cmd=(stage1_masklet.py --input-folder "$(host_to_container "$MV_VIDEO_DIR/$seq")"
           --annotation-folder "$(host_to_container "$MV_ANNOT_DIR/$seq")" --config "$CONFIG"
           --cam-res-scale 0.5 --output "$(host_to_container "$MV_STAGE1_DIR/$seq")") ;;
    idcheck)
      guard_mv_output "$MV_STAGE1_DIR/$seq"
      cmd=(scripts/check_masklet_ids.py --mvfit "$(host_to_container "$fit")"
           --stage1 "$(host_to_container "$MV_STAGE1_DIR/$seq")" --config "$CONFIG" ${EXTR_ARGS[@]+"${EXTR_ARGS[@]}"}) ;;
  esac
  if run_in_container "${cmd[@]}"; then
    echo "[INFO] $seq: $STEP done."
  else
    echo "[ERROR] $seq: $STEP FAILED." >&2
    failed=$((failed + 1))
  fi
done
echo "[INFO] All done, $failed failure(s)."
[ "$failed" -eq 0 ]
