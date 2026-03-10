#!/bin/bash
#SBATCH --job-name="raw_mhr_ingroup"
#SBATCH --partition=gpu-a100
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

# Batch Stage 2: raw MHR params for ingroup dataset.
#
# Folder format: /scratch/zli33/data/ingroup/bbox_kp/cam{MM}_batch{NN}
#   where MM = 2-digit camera number, NN = 2-digit batch number.
# Each folder contains images/ (00000000.jpg..00017999.jpg),
#   masks/ (00000000.png..00017999.png), id_mapping.json.
# Intrinsics derived from camera number:
#   /scratch/zli33/data/ingroup/intrinsics/parameters-camera-{MM}.json
#
# Supports splitting into 3 parts (p1/p2/p3) of 6000 frames each.
# The last part to finish auto-merges all 3.
#
# Usage (full run — all 18000 frames in one job):
#   sbatch job_scripts/raw_mhr_batch_delftblue_ingroup.sh cam06_batch01
#
# Usage (split into 3 parallel jobs):
#   sbatch job_scripts/raw_mhr_batch_delftblue_ingroup.sh cam06_batch01 p1
#   sbatch job_scripts/raw_mhr_batch_delftblue_ingroup.sh cam06_batch01 p2
#   sbatch job_scripts/raw_mhr_batch_delftblue_ingroup.sh cam06_batch01 p3
#
# Usage (manual merge, CPU-only):
#   sbatch job_scripts/raw_mhr_batch_delftblue_ingroup.sh cam06_batch01 merge

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path=$scratch_path/data/ingroup

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path=/mnt/data/ingroup
bind_home_path=/mnt/home/zli33

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
repo_dir=$home_path/projects/sam-body4d

# 18000 frames (00000000..00017999) split into 3 parts of 6000 each
FRAMES_PER_PART=6000

# ---------------------------------------------------------------------------
# Parse arguments: <FOLDER> [p1|p2|p3|merge]
# ---------------------------------------------------------------------------
FOLDER_NAME=""
PART_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    p1|p2|p3|merge)
      PART_ARG="$1"
      shift
      ;;
    *)
      FOLDER_NAME="$1"
      shift
      ;;
  esac
done

if [ -z "$FOLDER_NAME" ]; then
  echo "[ERROR] Folder name must be provided (e.g. cam06_batch01)." >&2
  echo "  Usage: sbatch raw_mhr_batch_delftblue_ingroup.sh cam06_batch01 [p1|p2|p3|merge]" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Extract camera number from folder name (cam{MM}_batch{NN} -> MM)
# ---------------------------------------------------------------------------
if [[ "$FOLDER_NAME" =~ ^cam([0-9]{2})_batch[0-9]{2}$ ]]; then
  CAM_NUM="${BASH_REMATCH[1]}"
else
  echo "[ERROR] Folder name must match cam{MM}_batch{NN} (e.g. cam06_batch01). Got: $FOLDER_NAME" >&2
  exit 2
fi

# Intrinsics path derived from camera number
intrinsic_path_host="$data_path/intrinsics/parameters-camera-${CAM_NUM}.json"
intrinsic_path_container="$bind_data_path/intrinsics/parameters-camera-${CAM_NUM}.json"

if [ ! -f "$intrinsic_path_host" ]; then
  echo "[ERROR] Intrinsics file not found: $intrinsic_path_host" >&2
  exit 2
fi

echo "[INFO] Folder:      $FOLDER_NAME"
echo "[INFO] Camera:      $CAM_NUM"
echo "[INFO] Intrinsics:  $intrinsic_path_host"
echo "[INFO] Part:        ${PART_ARG:-full}"

# ---------------------------------------------------------------------------
# Compute --frame-start / --frame-end / --part-label / --merge
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
case "$PART_ARG" in
  p1)
    FRAME_START=0
    FRAME_END=$FRAMES_PER_PART
    EXTRA_ARGS="--frame-start $FRAME_START --frame-end $FRAME_END --part-label p1"
    echo "[INFO] Part 1: frames [${FRAME_START}, ${FRAME_END})  (00000000 .. 00005999)"
    ;;
  p2)
    FRAME_START=$FRAMES_PER_PART
    FRAME_END=$((FRAMES_PER_PART * 2))
    EXTRA_ARGS="--frame-start $FRAME_START --frame-end $FRAME_END --part-label p2"
    echo "[INFO] Part 2: frames [${FRAME_START}, ${FRAME_END})  (00006000 .. 00011999)"
    ;;
  p3)
    FRAME_START=$((FRAMES_PER_PART * 2))
    FRAME_END=$((FRAMES_PER_PART * 3))
    EXTRA_ARGS="--frame-start $FRAME_START --frame-end $FRAME_END --part-label p3"
    echo "[INFO] Part 3: frames [${FRAME_START}, ${FRAME_END})  (00012000 .. 00017999)"
    ;;
  merge)
    EXTRA_ARGS="--merge"
    echo "[INFO] Merge mode: combining all parts into raw_mhr.pt"
    ;;
  "")
    echo "[INFO] Full run: processing all 18000 frames"
    ;;
  *)
    echo "[ERROR] Unknown part argument: $PART_ARG. Use p1, p2, p3, or merge." >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Validate folder exists
# ---------------------------------------------------------------------------
folder_host="$data_path/bbox_kp/$FOLDER_NAME"
if [ ! -d "$folder_host" ]; then
  echo "[ERROR] Folder does not exist: $folder_host" >&2
  exit 2
fi

input_container="$bind_data_path/bbox_kp/$FOLDER_NAME"

# ---------------------------------------------------------------------------
# Create code snapshot
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
echo "[INFO] input=$input_container"
echo ""

# ---------------------------------------------------------------------------
# Run Stage 2
# ---------------------------------------------------------------------------
echo "=================================================================="
echo "[INFO] Processing folder=$FOLDER_NAME  part=${PART_ARG:-full}"
echo "=================================================================="

if apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/run_sam3d_body_raw_params.py \
    --input "$input_container" \
    --intrinsic-path "$intrinsic_path_container" \
    --config ${CONFIG_REL:-configs/body4d.yaml} \
    $EXTRA_ARGS ; then
  echo ""
  echo "[INFO] folder=$FOLDER_NAME part=${PART_ARG:-full} completed successfully."
else
  echo ""
  echo "[ERROR] folder=$FOLDER_NAME part=${PART_ARG:-full} FAILED (exit code $?)." >&2
  exit 1
fi

echo "=================================================================="
echo "[INFO] Done."
echo "=================================================================="
