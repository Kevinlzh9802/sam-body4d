#!/bin/bash
#SBATCH --job-name="inference_test_delftblue"
#SBATCH --partition=gpu-a100 # Request partition. Default is 'general' 
#SBATCH --time=3:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=12    
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId


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

# Option 2: run from a snapshot created at SUBMISSION time.
# The submit helper script passes EXP_DIR as a HOST path (e.g. /scratch/.../outputs/exp_...).
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

# If you want to run multiple jobs under the same EXP_DIR (e.g., e2e + masklets + meshes),
# set OUTPUT_SUBDIR (e.g., "e2e") to avoid clobbering images/masks in the root.
run_output_dir_host="$exp_dir_host"
if [ -n "${OUTPUT_SUBDIR:-}" ]; then
  run_output_dir_host="$exp_dir_host/$OUTPUT_SUBDIR"
  mkdir -p "$run_output_dir_host"
fi
run_output_dir_container="$(host_to_container_path "$run_output_dir_host")"

# Prefer running from the snapshot if it exists.
if [ -d "$exp_dir_host/code" ]; then
  exp_name=$(basename "$exp_dir_host")
  exp_dir_container="$(host_to_container_path "$exp_dir_host")"
  project_folder="$exp_dir_container/code"
else
  echo "[WARN] No code snapshot found at $exp_dir_host/code; running from live repo in home."
  project_folder=$bind_home_path/projects/sam-body4d
fi

echo "[INFO] exp_dir_host=$exp_dir_host"
echo "[INFO] run_output_dir_container=$run_output_dir_container"
echo "[INFO] project_folder=$project_folder"

# apptainer exec --nv --bind $neon_path:$bind_neon_path --bind $zli_path:$bind_zli_path $sif_path python $project_folder/infer_video.py --video $input_folder/cam04_cut_03.mp4 --output $output_folder 

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/infer_video.py --video $input_folder/${VIDEO_REL:-cam04_cut_10s_undistorted_scaled_s1.mp4} --output $run_output_dir_container

# apptainer exec --env PYOPENGL_PLATFORM=osmesa $sif_path python -c "from OpenGL.osmesa import OSMesaCreateContextAttribs; print('ok')"