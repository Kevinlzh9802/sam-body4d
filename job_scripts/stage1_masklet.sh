#!/bin/bash
# Stage 1: SAM-3 masklets (stage1_masklet.py) -> <dataset>/bbox_kp/<seq>/
#   images.zip, masks.zip, id_mapping.json, masklets_meta.json, video_mask.mp4, ...
#
# Run with bash on a DelftBlue login node; the script submits
# itself to SLURM, running the code checkout at CODE_HOST. Not implemented on DAIC.
#
# Usage:
#   bash job_scripts/stage1_masklet.sh -cam 2 -b 1                 # cosilab cam02_batch01
#   bash job_scripts/stage1_masklet.sh -cam 2 -b 1,3-5 -cam_res_scale 0.5
#   bash job_scripts/stage1_masklet.sh cam06_batch01 cam06_batch02
#   bash job_scripts/stage1_masklet.sh -all cam06                  # every cam06_* video folder
#   bash job_scripts/stage1_masklet.sh -dataset conflab 428
#   bash job_scripts/stage1_masklet.sh -dataset conflab -all 4     # every 4xx video folder
#
# Options:
#   -dataset <cosilab|conflab>   default: cosilab
#   -cam <num> -b <spec>         cosilab batches (spec: 1 | 1,3,5 | 3-5 | 1,3-5)
#   -all <prefix>                all sequences in the video folder starting with <prefix>
#   -cam_res_scale <float>       scale annotation [x,y,w,h] to video resolution (default: 0.5)
#   -sbatch "<flags>"            extra sbatch flags, e.g. "--time=2:00:00"
#
# Inputs (host, under <data>/<dataset>/):
#   cosilab: video_segs_10s/<seq>/*_seg*.mp4, video_frame_annotations/<seq>*/
#   conflab: video_segs_20s/<seq>/*_seg*.mp4, video_frame_annotations/
set -euo pipefail

LAUNCHER="$(realpath "${BASH_SOURCE[0]}")"
SB4D_CODE="${SB4D_CODE:-$(cd "$(dirname "$LAUNCHER")/.." && pwd)}"
if [ ! -f "$SB4D_CODE/job_scripts/lib/common.sh" ]; then
  echo "[ERROR] job_scripts/lib not found under $SB4D_CODE. Start this script with bash, not sbatch." >&2
  exit 2
fi
source "$SB4D_CODE/job_scripts/lib/common.sh"

ORIG_ARGS=("$@")
CAM_RES_SCALE=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    -cam_res_scale|--cam_res_scale|--cam-res-scale)
      CAM_RES_SCALE="${2:-}"
      shift 2
      ;;
    *)
      parse_common_args "$@"
      shift "$COMMON_SHIFT"
      ;;
  esac
done

[[ "$CAM_RES_SCALE" =~ ^[0-9]*\.?[0-9]+$ ]] || die "-cam_res_scale must be a positive number, got: $CAM_RES_SCALE"
[ "$CLUSTER" != daic ] || die "NotImplementedError: Stage 1 is not implemented on DAIC. Run it on DelftBlue."

load_dataset "$DATASET_ARG"
resolve_seqs "$VIDEO_DIR"
self_submit stage1 "$LAUNCHER" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}

echo "[INFO] Cluster: $CLUSTER  Dataset: $DATASET  code: $SB4D_CODE"
echo "[INFO] Sequences: ${SEQS[*]}"
echo "[INFO] cam_res_scale: $CAM_RES_SCALE"
echo ""

total=${#SEQS[@]}
idx=0
failed=0

for seq in "${SEQS[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing sequence: $seq"
  echo "=================================================================="

  video_dir="$(seq_video_dir "$seq")"
  if [ ! -d "$video_dir" ]; then
    echo "[ERROR] Input folder not found: $video_dir" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  if ! annotation_dir="$(seq_annotation_dir "$seq")"; then
    echo "[ERROR] No annotation folder found for $seq under $DATA_ROOT/video_frame_annotations" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  output_dir="$BBOX_DIR/$seq"
  [ "${SB4D_DRY_RUN:-0}" = 1 ] || mkdir -p "$output_dir"

  echo "[INFO] input-folder=$video_dir"
  echo "[INFO] annotation-folder=$annotation_dir"
  echo "[INFO] output=$output_dir"

  if run_in_container stage1_masklet.py \
    --input-folder "$(host_to_container "$video_dir")" \
    --annotation-folder "$(host_to_container "$annotation_dir")" \
    --config "$CONFIG" \
    --cam-res-scale "$CAM_RES_SCALE" \
    --output "$(host_to_container "$output_dir")"; then
    echo "[INFO] [$idx/$total] Completed sequence: $seq"
  else
    echo "[ERROR] [$idx/$total] FAILED sequence: $seq" >&2
    failed=$((failed + 1))
  fi
  echo ""
done

echo "=================================================================="
echo "[INFO] All done. Processed $total sequence(s), $failed failure(s)."
echo "=================================================================="
[ "$failed" -eq 0 ]
