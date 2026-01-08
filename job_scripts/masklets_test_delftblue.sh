#!/bin/bash
#SBATCH --job-name="sam3_masklets"
#SBATCH --partition=gpu-a100
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
#SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path=$scratch_path/data/sam4d

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path=/mnt/data/sam4d_body
bind_home_path=/mnt/home/zli33

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
input_folder=$bind_data_path/inputs
output_folder=$bind_data_path/outputs

# Snapshot location (preferred): EXP_DIR=/mnt/data/sam4d_body/outputs/exp_... set by submit script
if [ -z "${EXP_DIR:-}" ]; then
  echo "[WARN] EXP_DIR not set; falling back to creating a fresh exp_dir at job start."
  timestamp=$(date +%Y%m%d_%H%M%S)
  rand_suffix=$(tr -dc 'A-Z0-9' </dev/urandom | head -c 4)
  exp_dir=$output_folder/exp_${timestamp}_${rand_suffix}
  mkdir -p "$exp_dir"
else
  exp_dir="$EXP_DIR"
  mkdir -p "$exp_dir"
fi

# Where to write stage-1 outputs (avoid clobbering e2e outputs)
run_output_dir="${MASKLETS_DIR:-$exp_dir/masklets}"
mkdir -p "$run_output_dir"

# Prefer running from the snapshot if it exists.
if [ -d "$exp_dir/code" ]; then
  exp_name=$(basename "$exp_dir")
  project_folder=$bind_data_path/outputs/$exp_name/code
else
  echo "[WARN] No code snapshot found at $exp_dir/code; running from live repo in home."
  project_folder=$bind_home_path/projects/sam-body4d
fi

echo "[INFO] EXP_DIR=$exp_dir"
echo "[INFO] project_folder=$project_folder"
echo "[INFO] run_output_dir=$run_output_dir"
echo "[INFO] video=$input_folder/${VIDEO_REL:-cam04_cut_10s.mp4}"

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:$PYTHONPATH \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/run_sam3_masklets.py \
    --video $input_folder/${VIDEO_REL:-cam04_cut_10s.mp4} \
    --config ${CONFIG_REL:-configs/body4d.yaml} \
    --output $run_output_dir


