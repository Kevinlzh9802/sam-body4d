#!/bin/bash
# Stage 3, multi-view option (cosilab): no assumed camera height. Metric scale
# from triangulating each person across the cameras of a calibrated rig, the
# floor from their feet (stage3_multiview.py)
# -> $SMOOTH_DIR/<seq>/stage3_multiview/ for every camera of the rig.
#
# Per batch and rig (parameters-camera-AA-to-BB.json in the extrinsics folder):
#   1. every camera with Stage 2 output: stage3_mhr.py in camera space without
#      the ground step -> <seq>/stage3_camera (reused if already there; the same
#      as `stage3_mhr.sh -camera <seq>`, which can make them in parallel first)
#   2. stage3_multiview.py over those cameras -> <seq>/stage3_multiview
# The height option (job_scripts/stage3_mhr.sh -> <seq>/stage3_height) is separate.
#
# Run with bash on a login node (DelftBlue or DAIC, auto-detected); the script
# submits itself to SLURM (CPU job).
#
# Usage:
#   bash job_scripts/stage3_multiview.sh -b 1              # batch 01, every rig
#   bash job_scripts/stage3_multiview.sh -b 1-7 -rig 06-10
#
# Options:
#   -b <spec>                    batches: 1 | 1,3,5 | 3-5 | 1,3-5 (required)
#   -rig <AA-BB>                 only this rig (default: every rig calibration found)
#   -sigma <frames> | -no-smooth | -no-mask   step 1 options (folder names get
#                                the same suffixes as stage3_mhr.sh's)
#   -redo                        run step 1 again even if stage3_camera exists
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
RIG=""
REDO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -sigma|--sigma)     [ $# -ge 2 ] || die "$1 needs a value"
                        STAGE3_ARGS+=(--smooth-sigma "$2"); SUFFIX+="_s$2"; shift 2 ;;
    -no-smooth)         STAGE3_ARGS+=(--no-smooth); SUFFIX+="_nosmooth"; shift ;;
    -no-mask)           STAGE3_ARGS+=(--no-mask-reproj); SUFFIX+="_nomask"; shift ;;
    -rig|--rig)         [ $# -ge 2 ] || die "$1 needs a value"; RIG="$2"; shift 2 ;;
    -redo)              REDO=1; shift ;;
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
[ -z "$CAM_NUM$ALL_PREFIX" ] && [ ${#POS_SEQS[@]} -eq 0 ] || die "Give batches with -b; the cameras come from the rig"
[ -z "$RIG" ] || [[ "$RIG" =~ ^[0-9]{2}-[0-9]{2}$ ]] || die "-rig must look like 06-10, got: $RIG"
expand_batch_spec "$BATCH_SPEC"

EXTR_DIR="$CAMERA_DIR/extrinsics"
shopt -s nullglob
RIG_FILES=("$EXTR_DIR"/parameters-camera-${RIG:-*-to-*}.json)
shopt -u nullglob
[ -n "$RIG" ] && RIG_FILES=("$EXTR_DIR/parameters-camera-${RIG/-/-to-}.json")
[ ${#RIG_FILES[@]} -gt 0 ] && [ -f "${RIG_FILES[0]}" ] || die "No rig calibration (parameters-camera-AA-to-BB.json) in $EXTR_DIR"

SEQS=()
for bn in "${BATCH_NUMS[@]}"; do SEQS+=("$(printf 'batch%02d' "$bn")"); done
CONTAINER_GPU=0
self_submit stage3_multiview "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Batches: ${SEQS[*]}  Rigs: ${RIG_FILES[*]##*/}"
echo "[INFO] Step 1: stage3_mhr.py --no-ground --export-camera-space ${STAGE3_ARGS[*]} -> <seq>/$CAMERA_LABEL"
echo "[INFO] Step 2: stage3_multiview.py -> <seq>/$MV_LABEL"
echo ""

failed=0
groups=0
for bn in "${BATCH_NUMS[@]}"; do
  for calib in "${RIG_FILES[@]}"; do
    [[ "$(basename "$calib")" =~ camera-([0-9]{2})-to-([0-9]{2}) ]] || continue
    first=$((10#${BASH_REMATCH[1]})); last=$((10#${BASH_REMATCH[2]}))
    groups=$((groups + 1))
    echo "=================================================================="
    echo "[INFO] batch $(printf %02d "$bn"), rig $(basename "$calib")"
    echo "=================================================================="
    cam_dirs=()
    for ((cn = first; cn <= last; cn++)); do
      seq="$(printf 'cam%02d_batch%02d' "$cn" "$bn")"
      raw="$RAW_DIR/$seq/raw_mhr.pt"
      if [ ! -f "$raw" ]; then
        echo "[INFO] $seq: no raw_mhr.pt, left out"
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
        --no-ground --export-camera-space \
        ${STAGE3_ARGS[@]+"${STAGE3_ARGS[@]}"}; then
        cam_dirs+=("$(host_to_container "$out_dir")")
      else
        echo "[ERROR] $seq: step 1 FAILED, left out" >&2
        failed=$((failed + 1))
      fi
    done
    if [ ${#cam_dirs[@]} -lt 2 ]; then
      echo "[WARN] Fewer than two cameras of this rig have output; skipping the multi-view step" >&2
      echo ""
      continue
    fi
    if run_in_container stage3_multiview.py "${cam_dirs[@]}" \
      --calib "$(host_to_container "$calib")" --label "$MV_LABEL"; then
      echo "[INFO] multi-view step completed."
    else
      echo "[ERROR] multi-view step FAILED." >&2
      failed=$((failed + 1))
    fi
    echo ""
  done
done

echo "=================================================================="
echo "[INFO] All done. $groups batch/rig group(s), $failed failure(s)."
echo "=================================================================="
[ "$failed" -eq 0 ]
