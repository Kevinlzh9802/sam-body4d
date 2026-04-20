#!/bin/bash
#SBATCH --job-name="smooth_meshes_batch"
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
#SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

# Batch Stage 3: smooth raw MHR params and export meshes for ingroup dataset.
#
# Folder format: /scratch/zli33/data/ingroup/bbox_kp/cam{MM}_batch{NN}
#   where MM = 2-digit camera number, NN = 2-digit batch number.
# Each folder must contain raw_mhr.pt and masks/ (from Stage 1/2).
# Output: meshes_4d_individual/ under the same folder (default --out = dirname of raw).
#
# Usage (single folder, full-sequence Stage 3):
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh cam06_batch01
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh cam08_batch02
#
# Usage (single folder, per-segment Stage 3):
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh -segment cam06_batch01
#
# Usage (single folder, simple centroid-based placement):
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh -simple cam06_batch01
#
# Usage (all folders with camera prefix):
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all cam06
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all -segment cam06
#   sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all -simple cam06
#
# Stage 3 mode:
#   Default ("full"):   smooth_mhr_and_export_meshes.py — processes the entire
#                        sequence as one continuous trajectory.
#   -segment / --segment: smooth_mhr_per_segment.py — processes each video
#                        segment independently, then concatenates meshes.
#                        Use this when multi-segment data shows cross-segment jitter.
#   -simple / --simple:  smooth_mhr_simple.py — no optimization; places meshes
#                        using mask centroids + a fixed world-height assumption.
#                        Uses fisheye centroid rays for ingroup camera intrinsics.
#
# Paths (host):
#   Dataset root:   /scratch/zli33/data/ingroup/bbox_kp/cam{MM}_batch{NN}
#   Intrinsics:      /scratch/zli33/data/ingroup/intrinsics/intrinsic_ingroup_cam{MM}.json
#   Extrinsics:      /scratch/zli33/data/ingroup/extrinsics/extrinsic_ingroup_cam{MM}.json (metres, if available)

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path=$scratch_path/data/ingroup

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path=/mnt/data/ingroup
bind_home_path=/mnt/home/zli33

# Intrinsics and extrinsics are camera-specific (derived from folder name)
extrinsics_dir_host="$data_path/extrinsics"
extrinsics_dir_container="$bind_data_path/extrinsics"
ingroup_world_scale=1.0

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
  echo "[ERROR] Folder name or camera prefix must be provided." >&2
  echo "  Usage: sbatch job_scripts/smooth_meshes_batch_ingroup.sh cam06_batch01" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_ingroup.sh -segment cam06_batch01" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_ingroup.sh -simple cam06_batch01" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all cam06" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all -segment cam06" >&2
  echo "         sbatch job_scripts/smooth_meshes_batch_ingroup.sh -all -simple cam06" >&2
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
# Build list of folder names (e.g. cam06_batch01, cam06_batch02) under bbox_kp
# ---------------------------------------------------------------------------
folder_nums=()

if [ "$ALL_MODE" = "1" ]; then
  cam_prefix="$INPUT_ARG"
  echo "[INFO] ALL mode: looking for folders matching ${cam_prefix}_* under $data_path/bbox_kp/"
  for dir in "$data_path/bbox_kp/${cam_prefix}_"*/; do
    [ -d "$dir" ] || continue
    folder_name="$(basename "$dir")"
    folder_nums+=("$folder_name")
  done
  if [ ${#folder_nums[@]} -eq 0 ]; then
    echo "[ERROR] No folders matching ${cam_prefix}_* found in $data_path/bbox_kp/" >&2
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

snapshot_parent_host="$data_path/bbox_kp/_snapshots"
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
  project_folder="$bind_data_path/bbox_kp/_snapshots/exp_${timestamp}_${rand_suffix}/code"
else
  echo "[WARN] Failed to create code snapshot; running from live repo in home."
  project_folder="$bind_home_path/projects/sam-body4d"
fi

echo "[INFO] project_folder=$project_folder"
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

  # Extract camera number from folder name (cam{MM}_batch{NN} -> MM)
  if [[ "$FOLDER_NUM" =~ ^cam([0-9]{2})_batch[0-9]{2}$ ]]; then
    CAM_NUM="${BASH_REMATCH[1]}"
  else
    echo "[WARN] Folder name must match cam{MM}_batch{NN} format. Got: $FOLDER_NUM. Skipping." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  folder_container="$bind_data_path/bbox_kp/$FOLDER_NUM"
  raw_path_container="$folder_container/raw_mhr.pt"

  if [ ! -f "$data_path/bbox_kp/$FOLDER_NUM/raw_mhr.pt" ]; then
    echo "[WARN] raw_mhr.pt not found: $data_path/bbox_kp/$FOLDER_NUM/raw_mhr.pt. Skipping." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  # Camera-specific intrinsics
  intrinsic_path_host="$data_path/intrinsics/intrinsic_ingroup_cam${CAM_NUM}.json"
  intrinsic_path_container="$bind_data_path/intrinsics/intrinsic_ingroup_cam${CAM_NUM}.json"
  
  if [ ! -f "$intrinsic_path_host" ]; then
    echo "[WARN] Intrinsics not found: $intrinsic_path_host. Skipping folder $FOLDER_NUM." >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  # Camera-specific extrinsics (if available)
  extrinsics_file_host="$extrinsics_dir_host/extrinsic_ingroup_cam${CAM_NUM}.json"
  extrinsics_file_container="$extrinsics_dir_container/extrinsic_ingroup_cam${CAM_NUM}.json"
  
  echo "[INFO] raw=$raw_path_container"
  echo "[INFO] out=(default: same folder -> meshes_4d_individual/)"
  echo "[INFO] camera-intrinsics-json=$intrinsic_path_container"
  
  if [ -f "$extrinsics_file_host" ]; then
    echo "[INFO] extrinsics-json=$extrinsics_file_container"
    EXTRINSICS_AVAILABLE=1
  else
    echo "[INFO] extrinsics-json=(not available, will skip if required)"
    EXTRINSICS_AVAILABLE=0
  fi

  echo "[INFO] script=$STAGE3_SCRIPT"

  # Build script-specific arguments
  if [ "$SIMPLE_MODE" = "1" ]; then
    EXTRA_ARGS=(
      --camera-intrinsics-json "$intrinsic_path_container"
      --centroid-ray-model fisheye
      --world-scale "$ingroup_world_scale"
    )
    if [ "$EXTRINSICS_AVAILABLE" = "1" ]; then
      EXTRA_ARGS+=(--extrinsics-json "$extrinsics_file_container")
    fi
  else
    EXTRA_ARGS=(
      --config ${CONFIG_REL:-configs/body4d.yaml}
      --camera-intrinsics-json "$intrinsic_path_container"
      --enable-ground
      --world-scale "$ingroup_world_scale"
      --contact-z-thresh 0.03
      --contact-vxy-thresh 0.05
    )
    if [ "$EXTRINSICS_AVAILABLE" = "1" ]; then
      EXTRA_ARGS+=(--extrinsics-json "$extrinsics_file_container")
    fi
  fi

  if apptainer exec --nv \
    --bind $model_path:$bind_model_path \
    --bind $data_path:$bind_data_path \
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
