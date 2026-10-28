#!/bin/bash
# Stage 3: smooth raw MHR params and export meshes -> $SMOOTH_DIR/<seq>/meshes_4d_individual/
#
# Needs $RAW_DIR/<seq>/raw_mhr.pt (Stage 2), $STAGE1_DIR/<seq>/masks.zip and
# images.zip (Stage 1), the sequence's intrinsics JSON and, if available, its
# extrinsics JSON (required for conflab; without it cosilab exports in camera
# space and skips ground optimization). All three dirs are <dataset>/bbox_kp/
# except cosilab on DAIC (see lib/dataset.sh), where the Stage 1 folder is only read.
#
# Run with bash on a login node (DelftBlue or DAIC, auto-detected); the script
# snapshots the code and submits itself to SLURM.
#
# Usage:
#   bash job_scripts/stage3_mhr.sh cam06_batch01
#   bash job_scripts/stage3_mhr.sh -segment cam06_batch01
#   bash job_scripts/stage3_mhr.sh -simple -all cam06
#   bash job_scripts/stage3_mhr.sh -dataset conflab 428
#   bash job_scripts/stage3_mhr.sh -dataset conflab -all -segment 4
#
# Modes:
#   (default)  stage3_mhr_full.py    - whole sequence as one trajectory
#   -segment   stage3_mhr_segment.py - each video segment independently, then concatenated
#   -simple    stage3_mhr_simple.py  - no optimization; mask centroids + fixed world height
#
# Options:
#   -dataset <cosilab|conflab>   default: cosilab
#   -cam <num> -b <spec> | -all <prefix> | <seq> ...   (see lib/common.sh)
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
MODE=full

while [[ $# -gt 0 ]]; do
  case "$1" in
    -segment|--segment) MODE=segment; shift ;;
    -simple|--simple)   MODE=simple; shift ;;
    *)
      parse_common_args "$@"
      shift "$COMMON_SHIFT"
      ;;
  esac
done

STAGE3_SCRIPT="stage3_mhr_${MODE}.py"

load_dataset "$DATASET_ARG"
resolve_seqs "$RAW_DIR"
self_submit stage3 "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Sequences: ${SEQS[*]}"
echo "[INFO] Stage 3 mode: $MODE ($STAGE3_SCRIPT)"
echo "[INFO] Stage 1 dir: $STAGE1_DIR"
echo "[INFO] Raw dir:     $RAW_DIR"
echo "[INFO] Output dir:  $SMOOTH_DIR"
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
  elif [ "$EXTRINSICS_REQUIRED" = 1 ]; then
    echo "[WARN] Extrinsics not found: $extrinsics. Skipping folder $seq." >&2
    failed=$((failed + 1))
    echo ""
    continue
  else
    echo "[INFO] extrinsics-json=(not available: $extrinsics) -> camera-space export, no ground opt"
  fi

  # Separate Stage 1 / output folders (default: next to raw_mhr.pt)
  DIR_ARGS=()
  if [ "$STAGE1_DIR" != "$RAW_DIR" ]; then
    DIR_ARGS+=(--input-dir "$(host_to_container "$STAGE1_DIR/$seq")")
  fi
  if [ "$SMOOTH_DIR" != "$RAW_DIR" ]; then
    DIR_ARGS+=(--out "$(host_to_container "$SMOOTH_DIR/$seq")")
  fi

  echo "[INFO] raw=$raw"
  echo "[INFO] output=$SMOOTH_DIR/$seq"
  echo "[INFO] camera-intrinsics-json=$intrinsics"

  if [ "$MODE" = simple ]; then
    MODE_ARGS=()
  else
    MODE_ARGS=(--config "$CONFIG" --enable-ground)
  fi

  if run_in_container "$STAGE3_SCRIPT" \
    --raw "$(host_to_container "$raw")" \
    --camera-intrinsics-json "$(host_to_container "$intrinsics")" \
    ${DIR_ARGS[@]+"${DIR_ARGS[@]}"} \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
    ${EXTRINSICS_ARGS[@]+"${EXTRINSICS_ARGS[@]}"}; then
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
