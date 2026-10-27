#!/bin/bash
# Stage 3: smooth raw MHR params and export meshes -> <dataset>/bbox_kp/<seq>/meshes_4d_individual/
#
# Needs bbox_kp/<seq>/raw_mhr.pt and masks.zip (Stage 1/2), the sequence's
# intrinsics JSON and, if available, its extrinsics JSON (required for conflab;
# without it cosilab exports in camera space and skips ground optimization).
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
resolve_seqs "$BBOX_DIR"
self_submit stage3 "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Sequences: ${SEQS[*]}"
echo "[INFO] Stage 3 mode: $MODE ($STAGE3_SCRIPT)"
echo ""

total=${#SEQS[@]}
idx=0
failed=0

for seq in "${SEQS[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing folder=$seq"
  echo "=================================================================="

  raw="$BBOX_DIR/$seq/raw_mhr.pt"
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

  echo "[INFO] raw=$raw"
  echo "[INFO] camera-intrinsics-json=$intrinsics"

  if [ "$MODE" = simple ]; then
    MODE_ARGS=()
  else
    MODE_ARGS=(--config "$CONFIG" --enable-ground)
  fi

  if run_in_container "$STAGE3_SCRIPT" \
    --raw "$(host_to_container "$raw")" \
    --camera-intrinsics-json "$(host_to_container "$intrinsics")" \
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
