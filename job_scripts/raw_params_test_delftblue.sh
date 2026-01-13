#!/bin/bash
#SBATCH --job-name="sam3d_raw_params"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/data/sam4d/outputs/Aslurms/slurm_%j.out
#SBATCH --error=/scratch/zli33/data/sam4d/outputs/Aslurms/slurm_%j.err

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path=$scratch_path/data/sam4d

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path=/mnt/data/sam4d_body
bind_home_path=/mnt/home/zli33

sif_path=$scratch_path/apptainers/body4d_osmesa.sif

# Translate a host path under $data_path into the corresponding container path under $bind_data_path.
host_to_container_path() {
  local p="$1"
  case "$p" in
    "$data_path"/*) echo "$bind_data_path/${p#"$data_path"/}" ;;
    *) echo "$p" ;;
  esac
}

usage() {
  echo "Usage:" >&2
  echo "  sbatch job_scripts/raw_params_test_delftblue.sh <STAGE1_DIR_HOST>" >&2
  echo "" >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,EXP_DIR=/scratch/.../outputs/exp_XXXX job_scripts/raw_params_test_delftblue.sh /scratch/.../outputs/exp_XXXX/masklets" >&2
}

# Input folder MUST be specified (host path preferred).
stage1_dir_host="${1:-${STAGE1_DIR:-${INPUT_DIR:-}}}"
if [ -z "${stage1_dir_host:-}" ]; then
  echo "[ERROR] Missing input folder (Stage1 dir containing images/ and masks/)." >&2
  usage
  exit 2
fi

stage1_dir_container="$(host_to_container_path "$stage1_dir_host")"

# This job uses a dedicated stage-2 code snapshot created AT SUBMISSION TIME under EXP_DIR/exp_s2_*/code.
# IMPORTANT: do NOT reuse or search for any existing `$EXP_DIR/code` snapshot here.
project_folder=""
if [ -z "${EXP_DIR:-}" ]; then
  echo "[ERROR] EXP_DIR must be set (host path) for this job." >&2
  echo "        Example: sbatch --export=ALL,EXP_DIR=/scratch/.../outputs/exp_XXXX ..." >&2
  exit 2
fi

if [ "${USE_LIVE_CODE:-0}" = "1" ]; then
  echo "[WARN] USE_LIVE_CODE=1; running from live repo in home."
  project_folder="$bind_home_path/projects/sam-body4d"
else
  if [ -z "${S2_CODE_DIR_HOST:-}" ]; then
    echo "[WARN] S2_CODE_DIR_HOST not set; running from live repo in home."
    project_folder="$bind_home_path/projects/sam-body4d"
  else
    if [ ! -d "$S2_CODE_DIR_HOST" ]; then
      echo "[WARN] S2_CODE_DIR_HOST does not exist: $S2_CODE_DIR_HOST; running from live repo in home."
      project_folder="$bind_home_path/projects/sam-body4d"
    else
      project_folder="$(host_to_container_path "$S2_CODE_DIR_HOST")"
    fi
  fi
fi

echo "[INFO] project_folder=$project_folder"
echo "[INFO] stage1_dir_host=$stage1_dir_host"
echo "[INFO] stage1_dir_container=$stage1_dir_container"

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/run_sam3d_body_raw_params.py \
    --input "$stage1_dir_container" \
    --config ${CONFIG_REL:-configs/body4d.yaml}

