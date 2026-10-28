#!/bin/bash
# Stage 2: SAM-3D-Body raw MHR params (stage2_sam3body.py) -> $RAW_DIR/<seq>/raw_mhr.pt
#
# Needs Stage 1 output in $STAGE1_DIR/<seq>/ (images.zip, masks.zip, id_mapping.json)
# and the sequence's intrinsics JSON. Both dirs are <dataset>/bbox_kp/ except
# cosilab on DAIC (see lib/dataset.sh), where the Stage 1 folder is only read.
#
# Run with bash on a login node (DelftBlue or DAIC, auto-detected); the script
# snapshots the code and submits itself to SLURM.
#
# Usage:
#   bash job_scripts/stage2_sam3body.sh cam06_batch01                # full run, one job
#   bash job_scripts/stage2_sam3body.sh -cam 6 -b 1-3                # several sequences, one job
#   bash job_scripts/stage2_sam3body.sh -all cam06
#   bash job_scripts/stage2_sam3body.sh -dataset conflab 228
#   bash job_scripts/stage2_sam3body.sh -dataset conflab -all 2
#
# Split one sequence into 3 parallel jobs of 6000 frames each (frames
# [0,6000), [6000,12000), [12000,18000)); the last part to finish merges them
# into raw_mhr.pt. `-part merge` merges by hand.
#   bash job_scripts/stage2_sam3body.sh cam06_batch01 -part p1
#   bash job_scripts/stage2_sam3body.sh cam06_batch01 -part p2
#   bash job_scripts/stage2_sam3body.sh cam06_batch01 -part p3
#   bash job_scripts/stage2_sam3body.sh cam06_batch01 -part merge
#
# Options:
#   -dataset <cosilab|conflab>   default: cosilab
#   -cam <num> -b <spec> | -all <prefix> | <seq> ...   (see lib/common.sh)
#   -part <p1|p2|p3|merge>       split run; needs exactly one sequence
#   -batch_size <n>              frames per inference call (default 16, peaks at ~56 GB
#                                GPU memory; use 8 on a 48 GB A40/L40)
#
# On DAIC this runs on an RTX Pro 6000 (Blackwell) with the CUDA 12.8 image
# ($SIF_PATH_CU128, built with apptainer/build.sh cu128).
#   -sbatch "<flags>"            extra sbatch flags, e.g. "--time=10:00:00" for a part
set -euo pipefail

LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
SB4D_CODE="${SB4D_CODE:-$(cd "$(dirname "$LAUNCHER")/.." && pwd)}"
if [ ! -f "$SB4D_CODE/job_scripts/lib/common.sh" ]; then
  echo "[ERROR] job_scripts/lib not found under $SB4D_CODE. Start this script with bash, not sbatch." >&2
  exit 2
fi
source "$SB4D_CODE/job_scripts/lib/common.sh"

ORIG_ARGS=("$@")
PART=""
BATCH_SIZE=""
# Must match NUM_EXPECTED_PARTS=3 in utils/merge_raw_mhr.py
FRAMES_PER_PART=6000

while [[ $# -gt 0 ]]; do
  case "$1" in
    -part|--part)
      PART="${2:-}"
      shift 2
      ;;
    -batch_size|--batch_size|--batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    *)
      parse_common_args "$@"
      shift "$COMMON_SHIFT"
      ;;
  esac
done

[ -z "$BATCH_SIZE" ] || [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "-batch_size must be a positive integer, got: $BATCH_SIZE"

STAGE=stage2
PART_ARGS=()
case "$PART" in
  "")    ;;
  p1)    PART_ARGS=(--frame-start 0 --frame-end "$FRAMES_PER_PART" --part-label p1) ;;
  p2)    PART_ARGS=(--frame-start "$FRAMES_PER_PART" --frame-end "$((FRAMES_PER_PART * 2))" --part-label p2) ;;
  p3)    PART_ARGS=(--frame-start "$((FRAMES_PER_PART * 2))" --frame-end "$((FRAMES_PER_PART * 3))" --part-label p3) ;;
  merge) PART_ARGS=(--merge) ;;
  *)     die "Unknown -part: $PART (use p1, p2, p3 or merge)" ;;
esac
case "$PART" in
  "")    ;;
  merge) STAGE=stage2_merge ;;
  *)     STAGE=stage2_part ;;
esac

BATCH_ARGS=()
[ -z "$BATCH_SIZE" ] || BATCH_ARGS=(--batch-size "$BATCH_SIZE")

# Blackwell GPUs need torch built for CUDA >= 12.8
[ "$CLUSTER" != daic ] || SIF_PATH=$SIF_PATH_CU128

load_dataset "$DATASET_ARG"
resolve_seqs "$STAGE1_DIR"
if [ -n "$PART" ] && [ ${#SEQS[@]} -ne 1 ]; then
  die "-part needs exactly one sequence, got ${#SEQS[@]}: ${SEQS[*]}"
fi
self_submit "$STAGE" "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Sequences: ${SEQS[*]}  Part: ${PART:-full}  batch size: ${BATCH_SIZE:-default}"
echo "[INFO] Stage 1 dir: $STAGE1_DIR"
echo "[INFO] Output dir:  $RAW_DIR"
echo "[INFO] Container:   $SIF_PATH"
echo ""

total=${#SEQS[@]}
idx=0
failed=0

for seq in "${SEQS[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing folder=$seq  part=${PART:-full}"
  echo "=================================================================="

  folder="$STAGE1_DIR/$seq"
  if [ ! -d "$folder" ]; then
    echo "[ERROR] Folder does not exist: $folder" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  intrinsics="$(seq_intrinsics "$seq")"
  if [ ! -f "$intrinsics" ]; then
    echo "[ERROR] Intrinsics file not found: $intrinsics" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  # A separate output dir makes stage2_sam3body.py treat the input as read-only
  OUTPUT_ARGS=()
  out_folder="$RAW_DIR/$seq"
  if [ "$out_folder" != "$folder" ]; then
    [ "${SB4D_DRY_RUN:-0}" = 1 ] || mkdir -p "$out_folder"
    OUTPUT_ARGS=(--output-dir "$(host_to_container "$out_folder")")
  fi

  echo "[INFO] input=$folder"
  echo "[INFO] output=$out_folder"
  echo "[INFO] intrinsics=$intrinsics"

  # No --out: raw_mhr.pt (full/merge) or raw_mhr_parts/raw_mhr_part_<label>.pt (parts)
  if run_in_container stage2_sam3body.py \
    --input "$(host_to_container "$folder")" \
    ${OUTPUT_ARGS[@]+"${OUTPUT_ARGS[@]}"} \
    --intrinsic-path "$(host_to_container "$intrinsics")" \
    --config "$CONFIG" \
    ${BATCH_ARGS[@]+"${BATCH_ARGS[@]}"} \
    ${PART_ARGS[@]+"${PART_ARGS[@]}"}; then
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
