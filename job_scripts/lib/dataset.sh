#!/bin/bash
# Per-dataset folder layout. Sourced by job_scripts/lib/common.sh.
#
# Every dataset lives under $DATA_BASE_HOST/<dataset>/ with:
#   bbox_kp/<seq>/            Stage 1-3 outputs (one folder per sequence)
#   intrinsics/, extrinsics/  camera parameters
#   video_frame_annotations/  bbox annotations for Stage 1
#
#                 cosilab                                   conflab
#   seq name      cam{MM}_batch{NN}                         <NUM> (e.g. 428)
#   videos        video_segs_10s/<seq>                      video_segs_20s/<seq>
#   annotations   video_frame_annotations/<seq>* (first)    video_frame_annotations (flat)
#   intrinsics    intrinsics/parameters-camera-{MM}.json    intrinsics/parameters-camera-04.json
#   extrinsics    extrinsics/extrinsic_cam_{MM}.json        extrinsics/extrinsic_zh_<first digit>.json
#                 (optional)                                (required by Stage 3)
#
# Per-sequence stage folders (<dir>/<seq>/) default to bbox_kp/ for all stages,
# and intrinsics/, extrinsics/ to the dataset root (CAMERA_DIR).
# cosilab on DAIC reads Stage 1 output from the 4TU upload (mounted read-only,
# never written to) and writes Stage 2/3 to data_temp:
#   STAGE1_DIR  $COSILAB_4TU_HOST/processed/keypoints/sam3_output
#               (/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload/...)
#   RAW_DIR     $COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_raw
#   SMOOTH_DIR  /tudelft.net/staff-umbrella/DEEF/COSILab/keypoints (for now;
#               was $COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_smoothed;
#               SB4D_STAGE3_OUT overrides it)
#   CAMERA_DIR  $COSILAB_PROJECT_HOST/data_temp/keypoints

load_dataset() {
  DATASET="$1"
  case "$DATASET" in
    cosilab)
      SEQ_REGEX='^cam([0-9]{2})_batch[0-9]{2}$'
      VIDEO_SUBDIR=video_segs_10s
      EXTRINSICS_REQUIRED=0
      ;;
    conflab)
      SEQ_REGEX='^[0-9]+$'
      VIDEO_SUBDIR=video_segs_20s
      EXTRINSICS_REQUIRED=1
      ;;
    *)
      echo "[ERROR] Unknown dataset: $DATASET (expected cosilab or conflab)" >&2
      exit 2
      ;;
  esac

  DATA_ROOT=$DATA_BASE_HOST/$DATASET
  BIND_DATA=/mnt/data/$DATASET
  BBOX_DIR=$DATA_ROOT/bbox_kp
  VIDEO_DIR=$DATA_ROOT/$VIDEO_SUBDIR

  STAGE1_DIR=$BBOX_DIR
  RAW_DIR=$BBOX_DIR
  SMOOTH_DIR=$BBOX_DIR
  CAMERA_DIR=$DATA_ROOT
  # Extra host:container[:ro] mounts for run_in_container
  EXTRA_BINDS=()
  if [ "$CLUSTER" = daic ] && [ "$DATASET" = cosilab ]; then
    STAGE1_DIR=$COSILAB_4TU_HOST/processed/keypoints/sam3_output
    RAW_DIR=$COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_raw
    SMOOTH_DIR="${SB4D_STAGE3_OUT:-/tudelft.net/staff-umbrella/DEEF/COSILab/keypoints}"
    CAMERA_DIR=$COSILAB_PROJECT_HOST/data_temp/keypoints
    EXTRA_BINDS=(
      "$COSILAB_4TU_HOST:/mnt/cosilab_4tu:ro"
      "$COSILAB_PROJECT_HOST/data_temp:/mnt/cosilab_temp"
    )
    # Stage 3 output outside data_temp gets its own mount
    case "$SMOOTH_DIR" in
      "$COSILAB_PROJECT_HOST/data_temp"|"$COSILAB_PROJECT_HOST/data_temp"/*) ;;
      *) EXTRA_BINDS+=("$SMOOTH_DIR:/mnt/cosilab_stage3") ;;
    esac
  fi
}

# cosilab recording sessions (two separate recordings). Within a session and
# batch, the same frame number is the same moment in every camera.
COSILAB_SESSIONS=([1]="06 07 08 09 10" [2]="01 02 03 04 05")

# The rig calibration (parameters-camera-AA-to-BB.json in <dir>) that has camera
# <MM>; prints nothing if none has it (cam05 for now).
rig_calibration_for_cam() {
  local dir="$1" cam=$((10#$2)) f
  for f in "$dir"/parameters-camera-*-to-*.json; do
    [[ "$(basename "$f")" =~ camera-([0-9]{2})-to-([0-9]{2}) ]] || continue
    if (( cam >= 10#${BASH_REMATCH[1]} && cam <= 10#${BASH_REMATCH[2]} )); then
      echo "$f"
      return 0
    fi
  done
}

# Two-digit camera number of a cosilab sequence (cam06_batch01 -> 06).
seq_cam() {
  [[ "$1" =~ $SEQ_REGEX ]] && echo "${BASH_REMATCH[1]}"
}

seq_video_dir() {
  echo "$VIDEO_DIR/$1"
}

# Prints the annotation folder for a sequence; returns 1 if none exists.
seq_annotation_dir() {
  local seq="$1"
  if [ "$DATASET" = conflab ]; then
    echo "$DATA_ROOT/video_frame_annotations"
    return 0
  fi
  local matches
  shopt -s nullglob
  matches=("$DATA_ROOT/video_frame_annotations/${seq}"*)
  shopt -u nullglob
  [ ${#matches[@]} -gt 0 ] || return 1
  IFS=$'\n' matches=($(printf '%s\n' "${matches[@]}" | sort)); unset IFS
  if [ ${#matches[@]} -gt 1 ]; then
    echo "[WARN] Multiple annotation folders matched ${seq}*. Using: ${matches[0]}" >&2
  fi
  echo "${matches[0]}"
}

seq_intrinsics() {
  if [ "$DATASET" = conflab ]; then
    echo "$CAMERA_DIR/intrinsics/parameters-camera-04.json"
  else
    echo "$CAMERA_DIR/intrinsics/parameters-camera-$(seq_cam "$1").json"
  fi
}

seq_extrinsics() {
  if [ "$DATASET" = conflab ]; then
    echo "$CAMERA_DIR/extrinsics/extrinsic_zh_${1:0:1}.json"
  else
    echo "$CAMERA_DIR/extrinsics/extrinsic_cam_$(seq_cam "$1").json"
  fi
}
