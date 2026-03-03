#!/bin/bash
#SBATCH --job-name="sam3_masklets_ingroup"
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

# Batch SAM-3 masklet extraction for ingroup dataset.
#
# Required args:
#   -cam <num>   camera number
#   -b <spec>    batch spec in [1-7], supports:
#                - single: 1
#                - list: 1,3,5
#                - range: 3-5
#                - mixed: 1,3-5
# Optional args:
#   -cam_res_scale <float>   scale annotation [x,y,w,h] to video resolution (default: 0.5)
#
# Example:
#   sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1
#   sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1,3-5
#   sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1,3-5 -cam_res_scale 0.5
#
# Constructed sequence format:
#   cam%02d_batch%02d (e.g., cam02_batch01)

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path_ingroup=$scratch_path/data/ingroup
data_path_sam4d=$scratch_path/data/sam4d

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path_ingroup=/mnt/data/ingroup
bind_data_path_sam4d=/mnt/data/sam4d_body
bind_home_path=/mnt/home/zli33

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
repo_dir=$home_path/projects/sam-body4d

CAM_NUM=""
BATCH_SPEC=""
CAM_RES_SCALE="0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -cam|--cam)
      CAM_NUM="${2:-}"
      shift 2
      ;;
    -b|--b|--batch)
      BATCH_SPEC="${2:-}"
      shift 2
      ;;
    -cam_res_scale|--cam_res_scale|--cam-res-scale)
      CAM_RES_SCALE="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: sbatch job_scripts/masklets_batch_ingroup.sh -cam <num> -b <spec> [-cam_res_scale <float>]"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      echo "Usage: sbatch job_scripts/masklets_batch_ingroup.sh -cam <num> -b <spec> [-cam_res_scale <float>]" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CAM_NUM" || -z "$BATCH_SPEC" ]]; then
  echo "[ERROR] Both -cam and -b are required." >&2
  echo "Usage: sbatch job_scripts/masklets_batch_ingroup.sh -cam <num> -b <spec> [-cam_res_scale <float>]" >&2
  exit 2
fi

if ! [[ "$CAM_NUM" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] -cam must be a non-negative integer, got: $CAM_NUM" >&2
  exit 2
fi

if ! [[ "$CAM_RES_SCALE" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  echo "[ERROR] -cam_res_scale must be a positive number, got: $CAM_RES_SCALE" >&2
  exit 2
fi

# Expand a batch spec (e.g., 1,3-5) into unique ordered batch numbers.
declare -a batch_nums=()
declare -A seen_batches=()
IFS=',' read -r -a batch_tokens <<< "$BATCH_SPEC"

if [ ${#batch_tokens[@]} -eq 0 ]; then
  echo "[ERROR] Empty -b specification." >&2
  exit 2
fi

for token in "${batch_tokens[@]}"; do
  token="${token//[[:space:]]/}"
  if [ -z "$token" ]; then
    echo "[ERROR] Invalid -b spec (empty token) in: $BATCH_SPEC" >&2
    exit 2
  fi

  if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    if (( start > end )); then
      echo "[ERROR] Invalid range in -b spec: $token (start > end)" >&2
      exit 2
    fi
    for ((bn=start; bn<=end; bn++)); do
      if (( bn < 1 || bn > 7 )); then
        echo "[ERROR] Batch out of range [1,7]: $bn (from token $token)" >&2
        exit 2
      fi
      if [ -z "${seen_batches[$bn]+x}" ]; then
        batch_nums+=("$bn")
        seen_batches[$bn]=1
      fi
    done
  elif [[ "$token" =~ ^[0-9]+$ ]]; then
    bn="$token"
    if (( bn < 1 || bn > 7 )); then
      echo "[ERROR] Batch out of range [1,7]: $bn" >&2
      exit 2
    fi
    if [ -z "${seen_batches[$bn]+x}" ]; then
      batch_nums+=("$bn")
      seen_batches[$bn]=1
    fi
  else
    echo "[ERROR] Invalid token in -b spec: $token" >&2
    echo "        Supported format examples: 1 | 1,3,5 | 3-5 | 1,3-5" >&2
    exit 2
  fi
done

if [ ${#batch_nums[@]} -eq 0 ]; then
  echo "[ERROR] No valid batches parsed from -b spec: $BATCH_SPEC" >&2
  exit 2
fi

echo "[INFO] Parsed batches: ${batch_nums[*]}"
echo "[INFO] cam_res_scale: $CAM_RES_SCALE"

timestamp=$(date +%Y%m%d_%H%M%S)
rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
)

snapshot_parent_host="$data_path_ingroup/bbox_kp/_snapshots"
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
  project_folder="$bind_data_path_ingroup/bbox_kp/_snapshots/exp_${timestamp}_${rand_suffix}/code"
else
  echo "[WARN] Failed to create code snapshot; running from live repo in home."
  project_folder="$bind_home_path/projects/sam-body4d"
fi

echo "[INFO] project_folder=$project_folder"
echo ""

total=${#batch_nums[@]}
idx=0
failed=0

for BATCH_NUM in "${batch_nums[@]}"; do
  idx=$((idx + 1))
  seq_name=$(printf "cam%02d_batch%02d" "$CAM_NUM" "$BATCH_NUM")

  echo "=================================================================="
  echo "[INFO] [$idx/$total] Processing sequence: $seq_name"
  echo "=================================================================="

  video_input_folder_container="$bind_data_path_ingroup/video_segs_10s/$seq_name"
  output_container="$bind_data_path_ingroup/bbox_kp/$seq_name"

  video_input_folder_host="$data_path_ingroup/video_segs_10s/$seq_name"
  if [ ! -d "$video_input_folder_host" ]; then
    echo "[ERROR] Input folder not found: $video_input_folder_host" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  shopt -s nullglob
  annotation_matches=("$data_path_ingroup/video_frame_annotations/${seq_name}"*)
  shopt -u nullglob

  if [ ${#annotation_matches[@]} -eq 0 ]; then
    echo "[ERROR] No annotation folder found for pattern: $data_path_ingroup/video_frame_annotations/${seq_name}*" >&2
    failed=$((failed + 1))
    echo ""
    continue
  fi

  IFS=$'\n' annotation_matches=($(printf '%s\n' "${annotation_matches[@]}" | sort)); unset IFS
  annotation_folder_host="${annotation_matches[0]}"

  if [ ${#annotation_matches[@]} -gt 1 ]; then
    echo "[WARN] Multiple annotation folders matched. Using first sorted entry:"
    echo "       $annotation_folder_host"
  fi

  annotation_folder_container="$bind_data_path_ingroup/video_frame_annotations/$(basename "$annotation_folder_host")"
  mkdir -p "$data_path_ingroup/bbox_kp/$seq_name"

  echo "[INFO] input-folder=$video_input_folder_container"
  echo "[INFO] annotation-folder=$annotation_folder_container"
  echo "[INFO] output=$output_container"

  if apptainer exec --nv \
    --bind $model_path:$bind_model_path \
    --bind $data_path_ingroup:$bind_data_path_ingroup \
    --bind $data_path_sam4d:$bind_data_path_sam4d \
    --bind $home_path:$bind_home_path \
    --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
    --env PYOPENGL_PLATFORM=osmesa \
    $sif_path \
    python $project_folder/run_sam3_masklets_batch.py \
      --input-folder "$video_input_folder_container" \
      --annotation-folder "$annotation_folder_container" \
      --config configs/body4d.yaml \
      --cam-res-scale "$CAM_RES_SCALE" \
      --output "$output_container" ; then
    echo "[INFO] [$idx/$total] Completed sequence: $seq_name"
  else
    echo "[ERROR] [$idx/$total] FAILED sequence: $seq_name" >&2
    failed=$((failed + 1))
  fi
  echo ""
done

echo "=================================================================="
echo "[INFO] All done. Processed $total batch(es), $failed failure(s)."
echo "=================================================================="
