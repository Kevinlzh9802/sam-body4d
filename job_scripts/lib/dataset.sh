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
#   STAGE1_DIR  $COSILAB_PROJECT_HOST/4TU_upload/processed/keypoints/sam3_output
#   RAW_DIR     $COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_raw
#   SMOOTH_DIR  $COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_smoothed
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
    STAGE1_DIR=$COSILAB_PROJECT_HOST/4TU_upload/processed/keypoints/sam3_output
    RAW_DIR=$COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_raw
    SMOOTH_DIR=$COSILAB_PROJECT_HOST/data_temp/keypoints/mhr_smoothed
    CAMERA_DIR=$COSILAB_PROJECT_HOST/data_temp/keypoints
    EXTRA_BINDS=(
      "$COSILAB_PROJECT_HOST/4TU_upload:/mnt/cosilab_4tu:ro"
      "$COSILAB_PROJECT_HOST/data_temp:/mnt/cosilab_temp"
    )
  fi
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
