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

# Translate a host path under $data_path into the corresponding container path under $bind_data_path.
host_to_container_path() {
  local p="$1"
  case "$p" in
    "$data_path"/*) echo "$bind_data_path/${p#"$data_path"/}" ;;
    *) echo "$p" ;;
  esac
}

# Snapshot location (preferred): EXP_DIR as HOST path (e.g. /scratch/.../outputs/exp_...) set by submit script
if [ -z "${EXP_DIR:-}" ]; then
  echo "[WARN] EXP_DIR not set; falling back to creating a fresh exp_dir at job start."
  timestamp=$(date +%Y%m%d_%H%M%S)
  rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
  )
  exp_dir_host=$data_path/outputs/exp_${timestamp}_${rand_suffix}
  mkdir -p "$exp_dir_host"
else
  exp_dir_host="$EXP_DIR"
  mkdir -p "$exp_dir_host"
fi

# Where to write stage-1 outputs (avoid clobbering e2e outputs)
run_output_dir_host="${MASKLETS_DIR:-$exp_dir_host/masklets}"
mkdir -p "$run_output_dir_host"
run_output_dir_container="$(host_to_container_path "$run_output_dir_host")"

# Prefer running from the snapshot if it exists.
if [ -d "$exp_dir_host/code" ]; then
  exp_dir_container="$(host_to_container_path "$exp_dir_host")"
  project_folder=$exp_dir_container/code
else
  echo "[WARN] No code snapshot found at $exp_dir_host/code; running from live repo in home."
  project_folder=$bind_home_path/projects/sam-body4d
fi

echo "[INFO] EXP_DIR(host)=$exp_dir_host"
echo "[INFO] project_folder=$project_folder"
echo "[INFO] run_output_dir_container=$run_output_dir_container"
echo "[INFO] video=$input_folder/${VIDEO_REL:-cam04_cut_10s.mp4}"

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/run_sam3_masklets.py \
    --video $input_folder/${VIDEO_REL:-cam04_cut_10s.mp4} \
    --config ${CONFIG_REL:-configs/body4d.yaml} \
    --output $run_output_dir_container


