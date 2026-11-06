#!/bin/bash
# Stage 3, multi-view option (cosilab): no assumed camera height. Metric scale
# from triangulating each person across the cameras of a recording session,
# the floor from their feet (stage3_multiview.py)
# -> $SMOOTH_DIR/<seq>/stage3_multiview/ for every camera taking part, each with
# its reprojection overlay video reproj_overlay.mp4.
#
# cosilab sessions (lib/dataset.sh), two separate recordings: session 1 =
# cam06-10, session 2 = cam01-05. Within a session and batch the same frame
# number is the same moment in every camera, so cameras are only combined with
# the other cameras of their session and batch. A camera takes part if it has
# Stage 2 output and is in the session's rig calibration
# (parameters-camera-AA-to-BB.json in the extrinsics folder; cam05 is not in
# parameters-camera-01-to-04.json, so it is left out with a warning).
#
# Per batch and session:
#   1. every camera taking part: stage3_mhr.py in camera space without the
#      ground step -> <seq>/stage3_camera (reused if already there; the same as
#      `stage3_mhr.sh -camera <seq>`, which can make them in parallel first)
#   2. stage3_multiview.py over those cameras -> <seq>/stage3_multiview, incl.
#      reproj_overlay.mp4 per camera
# The height option (job_scripts/stage3_mhr.sh -> <seq>/stage3_height) is separate.
#
# Run with bash on a login node (DelftBlue or DAIC, auto-detected); the script
# submits itself to SLURM (CPU job).
#
# Usage:
#   bash job_scripts/stage3_multiview.sh -b 1                # batch 01, both sessions
#   bash job_scripts/stage3_multiview.sh -b 1-7 -session 1   # cam06-10 only
#
# Options:
#   -b <spec>                    batches: 1 | 1,3,5 | 3-5 | 1,3-5 (required)
#   -session <1|2>               only this session (default: both)
#   -sigma <frames> | -no-smooth | -no-mask   step 1 options (folder names get
#                                the same suffixes as stage3_mhr.sh's)
#   -redo                        run step 1 again even if stage3_camera exists
#   -no-video                    no reproj_overlay.mp4
#   -sbatch "<flags>"            extra sbatch flags
set -euo pipefail

LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
SB4D_CODE="${SB4D_CODE:-$(cd "$(dirname "$LAUNCHER")/.." && pwd)}"
if [ ! -f "$SB4D_CODE/job_scripts/lib/common.sh" ]; then
  echo "[ERROR] job_scripts/lib not found under $SB4D_CODE. Start this script with bash, not sbatch." >&2
  exit 2
fi
source "$SB4D_CODE/job_scripts/lib/common.sh"

ORIG_ARGS=("$@")
STAGE3_ARGS=()
SUFFIX=""
SESSION=""
REDO=0
MV_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -sigma|--sigma)     [ $# -ge 2 ] || die "$1 needs a value"
                        STAGE3_ARGS+=(--smooth-sigma "$2"); SUFFIX+="_s$2"; shift 2 ;;
    -no-smooth)         STAGE3_ARGS+=(--no-smooth); SUFFIX+="_nosmooth"; shift ;;
    -no-mask)           STAGE3_ARGS+=(--no-mask-reproj); SUFFIX+="_nomask"; shift ;;
    -session|--session) [ $# -ge 2 ] || die "$1 needs a value"; SESSION="$2"; shift 2 ;;
    -redo)              REDO=1; shift ;;
    -no-video)          MV_ARGS+=(--no-video); shift ;;
    *)
      parse_common_args "$@"
      shift "$COMMON_SHIFT"
      ;;
  esac
done
CAMERA_LABEL="stage3_camera$SUFFIX"
MV_LABEL="stage3_multiview$SUFFIX"

load_dataset "$DATASET_ARG"
[ "$DATASET" = cosilab ] || die "The multi-view option needs a calibrated camera rig (cosilab)"
[ -n "$BATCH_SPEC" ] || die "-b <spec> is required (see -h)"
[ -z "$CAM_NUM$ALL_PREFIX" ] && [ ${#POS_SEQS[@]} -eq 0 ] || die "Give batches with -b; the cameras come from the sessions"
SESSIONS=("${!COSILAB_SESSIONS[@]}")
if [ -n "$SESSION" ]; then
  [ -n "${COSILAB_SESSIONS[$SESSION]+x}" ] || die "-session must be one of ${SESSIONS[*]}, got: $SESSION"
  SESSIONS=("$SESSION")
fi
expand_batch_spec "$BATCH_SPEC"
EXTR_DIR="$CAMERA_DIR/extrinsics"

SEQS=()
for bn in "${BATCH_NUMS[@]}"; do SEQS+=("$(printf 'batch%02d' "$bn")"); done
CONTAINER_GPU=0
self_submit stage3_multiview "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Batches: ${SEQS[*]}  Sessions: ${SESSIONS[*]}  Rig calibrations: $EXTR_DIR"
echo "[INFO] Step 1: stage3_mhr.py --no-ground --export-camera-space --no-video ${STAGE3_ARGS[*]} -> <seq>/$CAMERA_LABEL"
echo "[INFO] Step 2: stage3_multiview.py -> <seq>/$MV_LABEL"
echo ""

failed=0
groups=0
for bn in "${BATCH_NUMS[@]}"; do
  for session in "${SESSIONS[@]}"; do
    groups=$((groups + 1))
    echo "=================================================================="
    echo "[INFO] batch $(printf %02d "$bn"), session $session (cam${COSILAB_SESSIONS[$session]// /, cam})"
    echo "=================================================================="
    calib=""
    cam_dirs=()
    for cam in ${COSILAB_SESSIONS[$session]}; do
      seq="cam${cam}_batch$(printf %02d "$bn")"
      raw="$RAW_DIR/$seq/raw_mhr.pt"
      if [ ! -f "$raw" ]; then
        echo "[INFO] $seq: no raw_mhr.pt, left out"
        continue
      fi
      cam_calib="$(rig_calibration_for_cam "$EXTR_DIR" "$cam")"
      if [ -z "$cam_calib" ]; then
        echo "[WARN] $seq: cam$cam is in no rig calibration in $EXTR_DIR, left out" >&2
        continue
      fi
      calib="${calib:-$cam_calib}"
      if [ "$cam_calib" != "$calib" ]; then
        echo "[WARN] $seq: calibrated in $(basename "$cam_calib"), the session's other cameras in" \
             "$(basename "$calib"); left out" >&2
        continue
      fi
      out_dir="$SMOOTH_DIR/$seq/$CAMERA_LABEL"
      if [ "$REDO" = 0 ] && [ -f "$out_dir/stage3_opt_summary.json" ]; then
        echo "[INFO] $seq: reusing $out_dir"
        cam_dirs+=("$(host_to_container "$out_dir")")
        continue
      fi
      intrinsics="$(seq_intrinsics "$seq")"
      if [ ! -f "$intrinsics" ]; then
        echo "[WARN] $seq: intrinsics not found: $intrinsics, left out" >&2
        continue
      fi
      DIR_ARGS=(--out "$(host_to_container "$out_dir")")
      [ "$STAGE1_DIR" = "$RAW_DIR" ] || DIR_ARGS+=(--input-dir "$(host_to_container "$STAGE1_DIR/$seq")")
      echo "[INFO] $seq: step 1 -> $out_dir"
      if run_in_container stage3_mhr.py \
        --raw "$(host_to_container "$raw")" \
        --camera-intrinsics-json "$(host_to_container "$intrinsics")" \
        ${DIR_ARGS[@]+"${DIR_ARGS[@]}"} \
        --config "$CONFIG" \
        --no-ground --export-camera-space --no-video \
        ${STAGE3_ARGS[@]+"${STAGE3_ARGS[@]}"}; then
        cam_dirs+=("$(host_to_container "$out_dir")")
      else
        echo "[ERROR] $seq: step 1 FAILED, left out" >&2
        failed=$((failed + 1))
      fi
    done
    if [ ${#cam_dirs[@]} -lt 2 ]; then
      echo "[WARN] Fewer than two cameras of this session and batch can take part; skipping the multi-view step" >&2
      echo ""
      continue
    fi
    if run_in_container stage3_multiview.py "${cam_dirs[@]}" \
      --calib "$(host_to_container "$calib")" --label "$MV_LABEL" ${MV_ARGS[@]+"${MV_ARGS[@]}"}; then
      echo "[INFO] multi-view step completed."
    else
      echo "[ERROR] multi-view step FAILED." >&2
      failed=$((failed + 1))
    fi
    echo ""
  done
done

echo "=================================================================="
echo "[INFO] All done. $groups batch/session group(s), $failed failure(s)."
echo "=================================================================="
[ "$failed" -eq 0 ]
