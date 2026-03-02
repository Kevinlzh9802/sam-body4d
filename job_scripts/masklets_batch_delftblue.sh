#!/bin/bash
#SBATCH --job-name="sam3_masklets_batch"
#SBATCH --partition=gpu-a100
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

# Batch SAM-3 masklet extraction over all segments of a numbered video folder.
#
# This script is directly submittable — it creates its own code snapshot at
# job start and uses hardcoded default paths.
#
# Usage (single folder):
#   sbatch job_scripts/masklets_batch_delftblue.sh 428
#   sbatch job_scripts/masklets_batch_delftblue.sh 501
#
# Usage (all folders starting with a digit prefix):
#   sbatch job_scripts/masklets_batch_delftblue.sh -all 4
#     -> processes all <4xx> folders (e.g. 401, 428, 455, ...)
#   sbatch job_scripts/masklets_batch_delftblue.sh -all 2
#     -> processes all <2xx> folders
#
# Defaults (container paths):
#   --input-folder       /mnt/data/conflab/video_segs_20s/<NUM>
#   --annotation-folder  /mnt/data/conflab/video_frame_annotations
#   --output             /mnt/data/conflab/bbox_kp/<NUM>
#   --config             configs/body4d.yaml

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path_conflab=$scratch_path/data/conflab
data_path_sam4d=$scratch_path/data/sam4d

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path_conflab=/mnt/data/conflab
bind_data_path_sam4d=/mnt/data/sam4d_body
bind_home_path=/mnt/home/zli33

# Annotation folder: host path and container path (for --annotation-folder)
annotation_folder_host="$data_path_conflab/video_frame_annotations"
annotation_folder_container="$bind_data_path_conflab/video_frame_annotations"

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
repo_dir=$home_path/projects/sam-body4d

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ALL_MODE=0
INPUT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -all|--all)
      ALL_MODE=1
      shift
      ;;
    *)
      INPUT_ARG="$1"
      shift
      ;;
  esac
done

# Also accept VIDEO_NUM from env if not given as positional arg
INPUT_ARG="${INPUT_ARG:-${VIDEO_NUM:-}}"

if [ -z "${INPUT_ARG:-}" ]; then
  echo "[ERROR] A number must be provided." >&2
  echo "  Usage: sbatch job_scripts/masklets_batch_delftblue.sh 428" >&2
  echo "         sbatch job_scripts/masklets_batch_delftblue.sh -all 4" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Build list of VIDEO_NUMs to process
# ---------------------------------------------------------------------------
video_nums=()

if [ "$ALL_MODE" = "1" ]; then
  digit_prefix="$INPUT_ARG"
  echo "[INFO] ALL mode: looking for folders matching ${digit_prefix}* under $data_path_conflab/video_segs/"
  for dir in "$data_path_conflab/video_segs/${digit_prefix}"*/; do
    [ -d "$dir" ] || continue
    folder_name="$(basename "$dir")"
    video_nums+=("$folder_name")
  done
  if [ ${#video_nums[@]} -eq 0 ]; then
    echo "[ERROR] No folders matching ${digit_prefix}* found in $data_path_conflab/video_segs/" >&2
    exit 2
  fi
  # Sort for deterministic order
  IFS=$'\n' video_nums=($(sort <<<"${video_nums[*]}")); unset IFS
  echo "[INFO] Found ${#video_nums[@]} folder(s): ${video_nums[*]}"
else
  video_nums=("$INPUT_ARG")
  echo "[INFO] Single mode: VIDEO_NUM=$INPUT_ARG"
fi

# ---------------------------------------------------------------------------
# Create ONE code snapshot (shared across all folders in -all mode)
# ---------------------------------------------------------------------------
timestamp=$(date +%Y%m%d_%H%M%S)
rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
)

# Put the snapshot under a shared location
snapshot_parent_host="$data_path_conflab/bbox_kp/_snapshots"
mkdir -p "$snapshot_parent_host"
code_snapshot_host="$snapshot_parent_host/exp_${timestamp}_${rand_suffix}/code"
mkdir -p "$code_snapshot_host"
echo "[INFO] Creating code snapshot in: $code_snapshot_host"

if rsync -a --delete \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "outputs" \
  "$repo_dir/" \
  "$code_snapshot_host/" ; then
  project_folder="$bind_data_path_conflab/bbox_kp/_snapshots/exp_${timestamp}_${rand_suffix}/code"
else
  echo "[WARN] Failed to create code snapshot; running from live repo in home."
  project_folder="$bind_home_path/projects/sam-body4d"
fi

# bbox_pkl_dir_container="$bind_data_path_sam4d/inputs/bboxes_kps_refined"

echo "[INFO] project_folder=$project_folder"
# echo "[INFO] bbox_pkl_dir=$bbox_pkl_dir_container"
echo ""

# ---------------------------------------------------------------------------
# Process each folder
# ---------------------------------------------------------------------------
total=${#video_nums[@]}
idx=0
failed=0

for VIDEO_NUM in "${video_nums[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing VIDEO_NUM=$VIDEO_NUM"
  echo "=================================================================="

  video_input_folder_container="$bind_data_path_conflab/video_segs_20s/$VIDEO_NUM"
  output_container="$bind_data_path_conflab/bbox_kp/$VIDEO_NUM"

  # Ensure output dir exists on host
  mkdir -p "$data_path_conflab/bbox_kp/$VIDEO_NUM"

  echo "[INFO] video_input_folder=$video_input_folder_container"
  echo "[INFO] annotation_folder=$annotation_folder_container"
  echo "[INFO] output=$output_container"

  if apptainer exec --nv \
    --bind $model_path:$bind_model_path \
    --bind $data_path_conflab:$bind_data_path_conflab \
    --bind $data_path_sam4d:$bind_data_path_sam4d \
    --bind $home_path:$bind_home_path \
    --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
    --env PYOPENGL_PLATFORM=osmesa \
    $sif_path \
    python $project_folder/run_sam3_masklets_batch.py \
      --input-folder "$video_input_folder_container" \
      --annotation-folder "$annotation_folder_container" \
      --config ${CONFIG_REL:-configs/body4d.yaml} \
      --output "$output_container" ; then
    echo "[INFO] [$idx/$total] VIDEO_NUM=$VIDEO_NUM completed successfully."
  else
    echo "[ERROR] [$idx/$total] VIDEO_NUM=$VIDEO_NUM FAILED (exit code $?)." >&2
    failed=$((failed + 1))
  fi
  echo ""
done

echo "=================================================================="
echo "[INFO] All done. Processed $total folder(s), $failed failure(s)."
echo "=================================================================="
