#!/bin/bash
# Local Stage 3, multi-view option, on one 10 s segment of several cameras of one
# recording session (no SLURM): metric scale from triangulating the people across
# the cameras and the floor from their feet, no assumed camera height
# (stage3_multiview.py). Sessions: 1 = cam06-10, 2 = cam01-05 (cam05 has no rig
# calibration yet); frame numbers of one batch are the same moments only within
# a session, so the cameras must all be from one session.
#
# Per camera, scripts/local_stage3.sh --camera makes the camera-space input
# (experiments/stage3_local/<seq>/<segment>/stage3_camera, reused if present),
# then stage3_multiview.py writes .../stage3_multiview next to it (no overlay
# videos locally). Needs what
# local_stage3.sh needs for each camera, plus the rig calibration
# experiments/extrinsics/parameters-camera-AA-to-BB.json.
#
# Usage:
#   bash scripts/local_stage3_multiview.sh <batch> <segment> <cam> <cam> [<cam> ...]
#   bash scripts/local_stage3_multiview.sh 01 seg010 06 08
#   bash scripts/local_stage3_multiview.sh 01 seg010 06 08 --redo   # remake stage3_camera
#   bash scripts/local_stage3_multiview.sh 01 seg010 06 08 -- --max-ray-gap 0.2  # extra args
# Env: SB4D_SIF (default apptainer/body4d_osmesa_cu128.sif)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ $# -ge 4 ] || { sed -n '2,/^set -/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2; }
BATCH="$(printf '%02d' "$((10#$1))")"; SEG="${2#seg}"; shift 2
CAMS=()
REDO=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --redo) REDO=1; shift ;;
    --)     shift; EXTRA=("$@"); break ;;
    [0-9]|[0-9][0-9]) CAMS+=("$(printf '%02d' "$((10#$1))")"); shift ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ ${#CAMS[@]} -ge 2 ] || { echo "[ERROR] Needs at least two cameras" >&2; exit 2; }

# Sessions (separate recordings; frame numbers match only within one):
# session 1 = cam06-10, session 2 = cam01-05
session_of() { if (( 10#$1 >= 6 && 10#$1 <= 10 )); then echo 1; elif (( 10#$1 >= 1 && 10#$1 <= 5 )); then echo 2; fi; }
CAM_LIST="$(printf 'cam%s ' "${CAMS[@]}")"
SESSION="$(session_of "${CAMS[0]}")"
for c in "${CAMS[@]}"; do
  [ "$(session_of "$c")" = "$SESSION" ] && [ -n "$SESSION" ] || {
    echo "[ERROR] ${CAM_LIST% } are not all in one session (1 = cam06-10, 2 = cam01-05)" >&2; exit 1; }
done

# The session's rig calibration, which must have every camera (cam05 is in none yet)
CALIB=""
for f in experiments/extrinsics/parameters-camera-*-to-*.json; do
  [[ "$(basename "$f")" =~ camera-([0-9]{2})-to-([0-9]{2}) ]] || continue
  ok=1
  for c in "${CAMS[@]}"; do
    (( 10#$c >= 10#${BASH_REMATCH[1]} && 10#$c <= 10#${BASH_REMATCH[2]} )) || ok=0
  done
  [ "$ok" = 0 ] || CALIB="$f"
done
[ -n "$CALIB" ] || { echo "[ERROR] No rig calibration in experiments/extrinsics/ has all of ${CAM_LIST% }" >&2; exit 1; }

DIRS=()
for c in "${CAMS[@]}"; do
  seq="cam${c}_batch${BATCH}"
  seg_key="$(printf 'cam%s_seg%03d' "$c" "$((10#$SEG))")"
  dir="experiments/stage3_local/$seq/$seg_key/stage3_camera"
  if [ "$REDO" = 1 ] || [ ! -f "$dir/stage3_opt_summary.json" ]; then
    bash scripts/local_stage3.sh "$seq" "$SEG" --camera --no-jitter
  else
    echo "[INFO] Reusing $dir"
  fi
  DIRS+=("$dir")
done

SIF="${SB4D_SIF:-apptainer/body4d_osmesa_cu128.sif}"
echo "[INFO] Multi-view step: ${DIRS[*]} with $CALIB"
apptainer exec --pwd /mnt/sam-body4d --bind "$REPO:/mnt/sam-body4d" \
  --env "PYTHONPATH=/mnt/sam-body4d" "$SIF" \
  python stage3_multiview.py "${DIRS[@]}" --calib "$CALIB" --no-video ${EXTRA[@]+"${EXTRA[@]}"}

echo "[INFO] View the meshes: python scripts/view_stage3.py ${DIRS[*]//stage3_camera/stage3_multiview}  (needs viser)"
