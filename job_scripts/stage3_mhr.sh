#!/bin/bash
# Stage 3, height option: raw MHR params -> smoothed, mask-aligned, grounded
# meshes (stage3_mhr.py) -> $SMOOTH_DIR/<seq>/stage3_height/meshes_4d_individual/,
# plus reproj_overlay.mp4 there (the meshes on every Stage 1 frame)
#
# Steps (stage3_mhr.py, one person at a time): temporal smoothing -> MHR forward ->
# mask reprojection -> ground (each person scaled about the camera centre +
# shifted along the viewing ray so the feet are on the floor of the extrinsics,
# the image stays put) -> export in world coordinates.
# The floor comes from the camera's extrinsics JSON: for cosilab an assumed
# height of cam01/cam06 (scripts/make_ground_extrinsics.py --height 3.5). The
# multi-view option estimates the floor instead: job_scripts/stage3_multiview.sh.
#
# Needs $RAW_DIR/<seq>/raw_mhr.pt (Stage 2), $STAGE1_DIR/<seq>/masks.zip and
# images.zip (Stage 1), the sequence's intrinsics JSON and its extrinsics JSON
# (not needed with -no-ground / -camera).
# All three dirs are <dataset>/bbox_kp/ except cosilab on DAIC (see lib/dataset.sh),
# where the Stage 1 folder is only read.
#
# Run with bash on a login node (DelftBlue or DAIC, auto-detected); the script
# submits itself to SLURM (CPU job), running the code checkout at CODE_HOST (lib/cluster.sh).
#
# Usage:
#   bash job_scripts/stage3_mhr.sh cam06_batch01
#   bash job_scripts/stage3_mhr.sh -all cam06
#   bash job_scripts/stage3_mhr.sh -no-ground cam06_batch01     # -> <seq>/stage3_noground
#   bash job_scripts/stage3_mhr.sh -camera cam06_batch01        # -> <seq>/stage3_camera
#   bash job_scripts/stage3_mhr.sh -dataset conflab 428
#
# Options:
#   -dataset <cosilab|conflab>   default: cosilab
#   -cam <num> -b <spec> | -all <prefix> | <seq> ...   (see lib/common.sh)
#   -sigma <frames>              smoothing width (default: stage3_mhr.py's)
#   -no-smooth | -no-mask | -no-ground   leave a step out (to compare)
#   -no-video                    no reproj_overlay.mp4
#   -camera-space                export in camera space even with extrinsics
#   -camera                      = -no-ground -camera-space -no-video, into
#                                stage3_camera: the per-camera input of the multi-view
#                                option (its launcher makes these itself if missing)
#   -label <name>                output folder name (default: stage3_height, or
#                                stage3_noground / stage3_camera; plus _s<sigma> and
#                                _no<step> for the other options, _cam for -camera-space)
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
LABEL=""
SUFFIX=""
BASE=stage3_height
NO_GROUND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -sigma|--sigma)     [ $# -ge 2 ] || die "$1 needs a value"
                        STAGE3_ARGS+=(--smooth-sigma "$2"); SUFFIX+="_s$2"; shift 2 ;;
    -no-smooth)         STAGE3_ARGS+=(--no-smooth); SUFFIX+="_nosmooth"; shift ;;
    -no-mask)           STAGE3_ARGS+=(--no-mask-reproj); SUFFIX+="_nomask"; shift ;;
    -no-ground)         STAGE3_ARGS+=(--no-ground); BASE=stage3_noground; NO_GROUND=1; shift ;;
    -camera-space)      STAGE3_ARGS+=(--export-camera-space); SUFFIX+="_cam"; shift ;;
    -no-video)          STAGE3_ARGS+=(--no-video); shift ;;
    -camera)            STAGE3_ARGS+=(--no-ground --export-camera-space --no-video); BASE=stage3_camera; NO_GROUND=1; shift ;;
    -label|--label)     [ $# -ge 2 ] || die "$1 needs a value"; LABEL="$2"; shift 2 ;;
    -segment|-simple|--segment|--simple)
      die "$1 was removed: Stage 3 is one pipeline now (stage3_mhr.py), see -h" ;;
    *)
      parse_common_args "$@"
      shift "$COMMON_SHIFT"
      ;;
  esac
done
[ "$BASE" != stage3_camera ] || SUFFIX="${SUFFIX//_cam/}"
LABEL="${LABEL:-$BASE$SUFFIX}"

STAGE3_SCRIPT="stage3_mhr.py"
# CPU job (lib/slurm.sh); the stage 3 scripts fall back to --device cpu
CONTAINER_GPU=0

load_dataset "$DATASET_ARG"
resolve_seqs "$RAW_DIR"
self_submit stage3 "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Sequences: ${SEQS[*]}"
echo "[INFO] Stage 3: $STAGE3_SCRIPT ${STAGE3_ARGS[*]}"
echo "[INFO] Stage 1 dir: $STAGE1_DIR"
echo "[INFO] Raw dir:     $RAW_DIR"
echo "[INFO] Output dir:  $SMOOTH_DIR/<seq>/$LABEL"
echo ""

total=${#SEQS[@]}
idx=0
failed=0

for seq in "${SEQS[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing folder=$seq"
  echo "=================================================================="

  raw="$RAW_DIR/$seq/raw_mhr.pt"
  if [ ! -f "$raw" ]; then
    echo "[WARN] raw_mhr.pt not found: $raw. Skipping." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  intrinsics="$(seq_intrinsics "$seq")"
  if [ ! -f "$intrinsics" ]; then
    echo "[WARN] Intrinsics not found: $intrinsics. Skipping folder $seq." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  extrinsics="$(seq_extrinsics "$seq")"
  EXTRINSICS_ARGS=()
  if [ -f "$extrinsics" ]; then
    EXTRINSICS_ARGS=(--extrinsics-json "$(host_to_container "$extrinsics")")
    echo "[INFO] extrinsics-json=$extrinsics"
  elif [ "$EXTRINSICS_REQUIRED" = 1 ] || [ "$NO_GROUND" = 0 ]; then
    echo "[WARN] Extrinsics not found: $extrinsics (the height option needs them; -no-ground or" \
         "-camera export in camera space without). Skipping folder $seq." >&2
    failed=$((failed + 1))
    echo ""
    continue
  else
    echo "[INFO] extrinsics-json=(not available: $extrinsics) -> camera-space export"
  fi

  # One output folder per label, so variants don't overwrite each other
  out_dir="$SMOOTH_DIR/$seq/$LABEL"
  DIR_ARGS=(--out "$(host_to_container "$out_dir")")
  if [ "$STAGE1_DIR" != "$RAW_DIR" ]; then
    DIR_ARGS+=(--input-dir "$(host_to_container "$STAGE1_DIR/$seq")")
  fi

  echo "[INFO] raw=$raw"
  echo "[INFO] output=$out_dir"
  echo "[INFO] camera-intrinsics-json=$intrinsics"

  if run_in_container "$STAGE3_SCRIPT" \
    --raw "$(host_to_container "$raw")" \
    --camera-intrinsics-json "$(host_to_container "$intrinsics")" \
    ${DIR_ARGS[@]+"${DIR_ARGS[@]}"} \
    --config "$CONFIG" \
    ${EXTRINSICS_ARGS[@]+"${EXTRINSICS_ARGS[@]}"} \
    ${STAGE3_ARGS[@]+"${STAGE3_ARGS[@]}"}; then
    echo "[INFO] [$idx/$total] folder=$seq completed successfully."
  else
    echo "[ERROR] [$idx/$total] folder=$seq FAILED." >&2
    failed=$((failed + 1))
  fi
  echo ""
done

echo "=================================================================="
echo "[INFO] All done. Processed $total folder(s), $failed failure(s)."
echo "=================================================================="
[ "$failed" -eq 0 ]
