# #!/bin/bash
# #SBATCH --job-name="sam3d_meshes"
# #SBATCH --partition=gpu-a100
# #SBATCH --time=3:00:00
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=12
# #SBATCH --mem-per-cpu=8000M
# #SBATCH --gpus-per-task=1
# #SBATCH --mail-type=END
# #SBATCH --account=research-eemcs-insy
# #SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out
# #SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err

# set -euo pipefail

# home_path=/home/zli33
# scratch_path=/scratch/zli33

# model_path=$scratch_path/models/sam4d_checkpoints
# data_path=$scratch_path/data/sam4d

# bind_model_path=/mnt/sam4d_checkpoints
# bind_data_path=/mnt/data/sam4d_body
# bind_home_path=/mnt/home/zli33

# sif_path=$scratch_path/apptainers/body4d_osmesa.sif

# # Translate a host path under $data_path into the corresponding container path under $bind_data_path.
# host_to_container_path() {
#   local p="$1"
#   case "$p" in
#     "$data_path"/*) echo "$bind_data_path/${p#"$data_path"/}" ;;
#     *) echo "$p" ;;
#   esac
# }

# # Snapshot location (preferred): EXP_DIR as HOST path (e.g. /scratch/.../outputs/exp_...) set by submit script
# if [ -z "${EXP_DIR:-}" ]; then
#   echo "[ERROR] EXP_DIR must be set (it should point to the snapshot exp folder)."
#   exit 2
# fi
# exp_dir_host="$EXP_DIR"

# # Stage-2 reads stage-1 outputs from this folder
# input_dir_host="${MASKLETS_DIR:-$exp_dir_host/masklets}"
# input_dir_container="$(host_to_container_path "$input_dir_host")"
# if [ ! -d "$input_dir_host" ]; then
#   echo "[ERROR] masklets input dir not found (host): $input_dir_host"
#   echo "        Run stage-1 first (masklets_test_delftblue.sh) or set MASKLETS_DIR."
#   exit 2
# fi

# # Stage-2 gets its own dedicated code snapshot under EXP_DIR.
# # IMPORTANT: do NOT reuse or search for any existing `$exp_dir_host/code` snapshot here.
# timestamp=$(date +%Y%m%d_%H%M%S)
# rand_suffix=$(python3 - <<'PY'
# import random, string
# print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
# PY
# )
# s2_dir_host="$exp_dir_host/exp_s2_${timestamp}_${rand_suffix}"
# code_snapshot_host="$s2_dir_host/code"
# mkdir -p "$code_snapshot_host"
# echo "[INFO] Creating stage-2 code snapshot in: $code_snapshot_host"

# # Copy a snapshot of the repo at submission/run time.
# # If this fails for any reason, fall back to the live repo.
# repo_dir="$home_path/projects/sam-body4d"
# if rsync -a --delete \
#   --exclude ".git" \
#   --exclude "__pycache__" \
#   --exclude "*.pyc" \
#   --exclude "outputs" \
#   "$repo_dir/" \
#   "$code_snapshot_host/" ; then
#   project_folder="$(host_to_container_path "$code_snapshot_host")"
# else
#   echo "[WARN] Failed to create stage-2 code snapshot; running from live repo in home."
#   project_folder="$bind_home_path/projects/sam-body4d"
# fi

# echo "[INFO] EXP_DIR(host)=$exp_dir_host"
# echo "[INFO] s2_dir_host=$s2_dir_host"
# echo "[INFO] project_folder=$project_folder"
# echo "[INFO] input_dir_container=$input_dir_container"

# camera_args=()
# if [ -n "${CAMERA_INTRINSICS:-}" ]; then
#   camera_args+=(--camera-intrinsics "$CAMERA_INTRINSICS")
#   camera_args+=(--camera-scale "${CAMERA_SCALE:-0.5}")
# fi

# apptainer exec --nv \
#   --bind $model_path:$bind_model_path \
#   --bind $data_path:$bind_data_path \
#   --bind $home_path:$bind_home_path \
#   --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
#   --env PYOPENGL_PLATFORM=osmesa \
#   $sif_path \
#   python $project_folder/run_sam3d_body_meshes.py \
#     --input $input_dir_container \
#     ${CONFIG_REL:+--config $CONFIG_REL} \
#     ${BATCH_SIZE:+--batch-size $BATCH_SIZE} \
#     ${NO_RENDER:+--no-render} \
#     "${camera_args[@]}"


