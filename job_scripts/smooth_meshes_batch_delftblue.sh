#!/bin/bash
#SBATCH --job-name="smooth_meshes_batch"
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
#SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

# Batch Stage 3: smooth raw MHR params and export meshes for each bbox_kp/<NUM> folder.
#
# Each folder (e.g. 128, 428, 501) must contain raw_mhr.pt and masks/ (from Stage 1/2).
# Output: meshes_4d_individual/ under the same folder (default --out = dirname of raw).
#
# Usage (single folder, full-sequence Stage 3):
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh 428
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh 501
#
# Usage (single folder, per-segment Stage 3):
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh -segment 428
#
# Usage (single folder, simple centroid-based placement):
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh -simple 428
#
# Usage (all folders with digit prefix):
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all 4
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all -segment 4
#   sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all -simple 4
#
# Stage 3 mode:
#   Default ("full"):   smooth_mhr_and_export_meshes.py — processes the entire
#                        sequence as one continuous trajectory.
#   -segment / --segment: smooth_mhr_per_segment.py — processes each video
#                        segment independently, then concatenates meshes.
#                        Use this when multi-segment data shows cross-segment jitter.
#   -simple / --simple:  smooth_mhr_simple.py — no optimization; places meshes
#                        using mask centroids + a fixed world-height assumption.
#
# Paths (host):
#   Dataset root:   /scratch/zli33/data/conflab/bbox_kp/<NUM>
#   Intrinsics:      /scratch/zli33/data/conflab/intrinsics/parameters-camera-04.json
#   Extrinsics:      /scratch/zli33/data/conflab/extrinsics/extrinsic_zh_<first_digit>.json

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path_conflab=$scratch_path/data/conflab

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path_conflab=/mnt/data/conflab
bind_home_path=/mnt/home/zli33

# Intrinsics: same file for all tasks (host and container paths)
intrinsic_path_host="$data_path_conflab/intrinsics/parameters-camera-04.json"
intrinsic_path_container="$bind_data_path_conflab/intrinsics/parameters-camera-04.json"

# Extrinsics dir (container); file name is extrinsic_zh_<n>.json where n = first digit of folder name
extrinsics_dir_host="$data_path_conflab/extrinsics"
extrinsics_dir_container="$bind_data_path_conflab/extrinsics"

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
repo_dir=$home_path/projects/sam-body4d

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ALL_MODE=0
SEGMENT_MODE=0
SIMPLE_MODE=0
INPUT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -all|--all)
      ALL_MODE=1
      shift
      ;;
    -segment|--segment)
      SEGMENT_MODE=1
      shift
      ;;
    -simple|--simple)
      SIMPLE_MODE=1
      shift
      ;;
    *)
      INPUT_ARG="$1"
      shift
      ;;
  esac
done

INPUT_ARG="${INPUT_ARG:-${VIDEO_NUM:-}}"

if [ -z "${INPUT_ARG:-}" ]; then
  echo "[ERROR] A number (folder name or digit prefix) must be provided." >&2
  echo "  Usage: sbatch job_scripts/smooth_meshes_batch_delftblue.sh 428" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_delftblue.sh -segment 428" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_delftblue.sh -simple 428" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all 4" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all -segment 4" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_delftblue.sh -all -simple 4" >&2
  exit 2
fi

if [ "$SIMPLE_MODE" = "1" ]; then
  STAGE3_SCRIPT="smooth_mhr_simple.py"
  echo "[INFO] Stage 3 mode: SIMPLE (smooth_mhr_simple.py)"
elif [ "$SEGMENT_MODE" = "1" ]; then
  STAGE3_SCRIPT="smooth_mhr_per_segment.py"
  echo "[INFO] Stage 3 mode: PER-SEGMENT (smooth_mhr_per_segment.py)"
else
  STAGE3_SCRIPT="smooth_mhr_and_export_meshes.py"
  echo "[INFO] Stage 3 mode: FULL (smooth_mhr_and_export_meshes.py)"
fi

# ---------------------------------------------------------------------------
# Build list of folder names (e.g. 128, 223, 428) under bbox_kp
# ---------------------------------------------------------------------------
folder_nums=()

if [ "$ALL_MODE" = "1" ]; then
  digit_prefix="$INPUT_ARG"
  echo "[INFO] ALL mode: looking for folders matching ${digit_prefix}* under $data_path_conflab/bbox_kp/"
  for dir in "$data_path_conflab/bbox_kp/${digit_prefix}"*/; do
    [ -d "$dir" ] || continue
    folder_name="$(basename "$dir")"
    folder_nums+=("$folder_name")
  done
  if [ ${#folder_nums[@]} -eq 0 ]; then
    echo "[ERROR] No folders matching ${digit_prefix}* found in $data_path_conflab/bbox_kp/" >&2
    exit 2
  fi
  IFS=$'\n' folder_nums=($(sort <<<"${folder_nums[*]}")); unset IFS
  echo "[INFO] Found ${#folder_nums[@]} folder(s): ${folder_nums[*]}"
else
  folder_nums=("$INPUT_ARG")
  echo "[INFO] Single mode: folder=$INPUT_ARG"
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

echo "[INFO] project_folder=$project_folder"
echo "[INFO] intrinsic_path=$intrinsic_path_container"
echo ""

# ---------------------------------------------------------------------------
# Process each folder: smooth_mhr_and_export_meshes.py -> meshes_4d_individual/
# ---------------------------------------------------------------------------
total=${#folder_nums[@]}
idx=0
failed=0

for FOLDER_NUM in "${folder_nums[@]}"; do
  idx=$((idx + 1))
  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing folder=$FOLDER_NUM"
  echo "=================================================================="

  folder_container="$bind_data_path_conflab/bbox_kp/$FOLDER_NUM"
  raw_path_container="$folder_container/raw_mhr.pt"

  if [ ! -f "$data_path_conflab/bbox_kp/$FOLDER_NUM/raw_mhr.pt" ]; then
    echo "[WARN] raw_mhr.pt not found: $data_path_conflab/bbox_kp/$FOLDER_NUM/raw_mhr.pt. Skipping." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  # First digit of folder name (e.g. 428 -> 4, 501 -> 5) for extrinsics file
  first_digit="${FOLDER_NUM:0:1}"
  extrinsics_file_container="$extrinsics_dir_container/extrinsic_zh_${first_digit}.json"
  if [ ! -f "$data_path_conflab/extrinsics/extrinsic_zh_${first_digit}.json" ]; then
    echo "[WARN] Extrinsics not found: $data_path_conflab/extrinsics/extrinsic_zh_${first_digit}.json. Skipping folder $FOLDER_NUM." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  echo "[INFO] raw=$raw_path_container"
  echo "[INFO] out=(default: same folder -> meshes_4d_individual/)"
  echo "[INFO] camera-intrinsics-json=$intrinsic_path_container"
  echo "[INFO] extrinsics-json=$extrinsics_file_container"

  echo "[INFO] script=$STAGE3_SCRIPT"

  # Build script-specific arguments
  if [ "$SIMPLE_MODE" = "1" ]; then
    EXTRA_ARGS=(
      --camera-intrinsics-json "$intrinsic_path_container"
      --extrinsics-json "$extrinsics_file_container"
    )
  else
    EXTRA_ARGS=(
      --config ${CONFIG_REL:-configs/body4d.yaml}
      --camera-intrinsics-json "$intrinsic_path_container"
      --extrinsics-json "$extrinsics_file_container"
      --enable-ground
    )
  fi

  if apptainer exec --nv \
    --bind $model_path:$bind_model_path \
    --bind $data_path_conflab:$bind_data_path_conflab \
    --bind $home_path:$bind_home_path \
    --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
    --env PYOPENGL_PLATFORM=osmesa \
    $sif_path \
    python $project_folder/$STAGE3_SCRIPT \
      --raw "$raw_path_container" \
      "${EXTRA_ARGS[@]}" ; then
    echo "[INFO] [$idx/$total] folder=$FOLDER_NUM completed successfully."
  else
    echo "[ERROR] [$idx/$total] folder=$FOLDER_NUM FAILED (exit code $?)." >&2
    failed=$((failed + 1))
  fi
  echo ""
done

echo "=================================================================="
echo "[INFO] All done. Processed $total folder(s), $failed failure(s)."
echo "=================================================================="
