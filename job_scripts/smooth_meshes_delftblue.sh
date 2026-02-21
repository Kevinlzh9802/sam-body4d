#!/bin/bash
#SBATCH --job-name="sam3d_smooth_meshes"
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
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
  echo "  sbatch job_scripts/smooth_meshes_delftblue.sh <EXP_DIR_HOST>" >&2
  echo "" >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,EXP_DIR=/scratch/.../outputs/exp_XXXX job_scripts/smooth_meshes_delftblue.sh /scratch/.../outputs/exp_XXXX" >&2
}

# EXP_DIR: stage1/2 parent folder (expects masklets/raw_mhr.pt under it).
exp_dir_host="${EXP_DIR:-}"
# EXP_S3_DIR: stage3 snapshot folder where outputs should be written.
exp_s3_dir_host="${1:-${EXP_S3_DIR:-}}"
if [ -z "${exp_s3_dir_host:-}" ]; then
  echo "[ERROR] Missing EXP_S3_DIR (stage3 snapshot folder)." >&2
  usage
  exit 2
fi
if [ -z "${exp_dir_host:-}" ]; then
  echo "[ERROR] Missing EXP_DIR (stage1/2 parent folder)." >&2
  usage
  exit 2
fi

exp_dir_container="$(host_to_container_path "$exp_dir_host")"
exp_s3_dir_container="$(host_to_container_path "$exp_s3_dir_host")"
raw_path_container="$exp_dir_container/masklets/raw_mhr.pt"

# Optional: use a dedicated code snapshot if provided, otherwise live repo in home.
project_folder=""
if [ "${USE_LIVE_CODE:-0}" = "1" ]; then
  echo "[WARN] USE_LIVE_CODE=1; running from live repo in home."
  project_folder="$bind_home_path/projects/sam-body4d"
else
  if [ -z "${S3_CODE_DIR_HOST:-}" ]; then
    echo "[WARN] S3_CODE_DIR_HOST not set; running from live repo in home."
    project_folder="$bind_home_path/projects/sam-body4d"
  else
    project_folder="$(host_to_container_path "$S3_CODE_DIR_HOST")"
  fi
fi

echo "[INFO] project_folder=$project_folder"
echo "[INFO] exp_dir_host=$exp_dir_host"
echo "[INFO] exp_dir_container=$exp_dir_container"
echo "[INFO] exp_s3_dir_host=$exp_s3_dir_host"
echo "[INFO] exp_s3_dir_container=$exp_s3_dir_container"
echo "[INFO] raw_path_container=$raw_path_container"

# Default output dir: <EXP_S3_DIR> (meshes_4d_individual will be created inside)
out_dir_container="${OUT_DIR:-$exp_s3_dir_container}"

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:${PYTHONPATH:-} \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/smooth_mhr_and_export_meshes.py \
    --raw "$raw_path_container" \
    --config ${CONFIG_REL:-configs/body4d.yaml} \
    --out "$out_dir_container" \
    ${NO_OPTION1:+--no-option1} \
    ${NO_MASK_REPROJ:+--no-mask-reproj} \
    ${ENABLE_KPS_REPROJ:+--enable-kps-reproj} \
    ${ENABLE_GROUND:+--enable-ground} \
    ${CAMERA_INTRINSICS_JSON:+--camera-intrinsics-json "$CAMERA_INTRINSICS_JSON"} \
    ${CAMERA_SCALE:+--camera-scale "$CAMERA_SCALE"} \
    ${MASK_REPROJ_ITERS:+--mask-reproj-iters "$MASK_REPROJ_ITERS"} \
    ${MASK_REPROJ_LR:+--mask-reproj-lr "$MASK_REPROJ_LR"} \
    ${MASK_REPROJ_LAMBDA_VERTEX:+--mask-reproj-lambda-vertex "$MASK_REPROJ_LAMBDA_VERTEX"} \
    ${MASK_REPROJ_LAMBDA_PRIOR:+--mask-reproj-lambda-prior "$MASK_REPROJ_LAMBDA_PRIOR"} \
    ${MASK_REPROJ_LAMBDA_VEL:+--mask-reproj-lambda-vel "$MASK_REPROJ_LAMBDA_VEL"} \
    ${MASK_REPROJ_LAMBDA_ACCEL:+--mask-reproj-lambda-accel "$MASK_REPROJ_LAMBDA_ACCEL"} \
    ${MASK_REPROJ_NUM_VERTS:+--mask-reproj-num-verts "$MASK_REPROJ_NUM_VERTS"} \
    ${BBOX_KPS_PKL:+--bbox-kps-pkl "$BBOX_KPS_PKL"} \
    ${REPROJ_ITERS:+--reproj-iters "$REPROJ_ITERS"} \
    ${REPROJ_LR:+--reproj-lr "$REPROJ_LR"} \
    ${REPROJ_HUBER_DELTA:+--reproj-huber-delta "$REPROJ_HUBER_DELTA"} \
    ${REPROJ_LAMBDA_PRIOR:+--reproj-lambda-prior "$REPROJ_LAMBDA_PRIOR"} \
    ${REPROJ_LAMBDA_VEL:+--reproj-lambda-vel "$REPROJ_LAMBDA_VEL"} \
    ${REPROJ_LAMBDA_ACCEL:+--reproj-lambda-accel "$REPROJ_LAMBDA_ACCEL"} \
    ${OBS_SCALE_CANDS:+--obs-scale-cands $OBS_SCALE_CANDS} \
    ${EXTRINSICS_JSON:+--extrinsics-json "$EXTRINSICS_JSON"} \
    ${GROUND_ITERS:+--ground-iters "$GROUND_ITERS"} \
    ${GROUND_LR:+--ground-lr "$GROUND_LR"} \
    ${GROUND_LAMBDA_PLANE:+--ground-lambda-plane "$GROUND_LAMBDA_PLANE"} \
    ${GROUND_LAMBDA_SLIDE:+--ground-lambda-slide "$GROUND_LAMBDA_SLIDE"} \
    ${GROUND_LAMBDA_PRIOR:+--ground-lambda-prior "$GROUND_LAMBDA_PRIOR"} \
    ${GROUND_LAMBDA_VEL:+--ground-lambda-vel "$GROUND_LAMBDA_VEL"} \
    ${GROUND_LAMBDA_ACCEL:+--ground-lambda-accel "$GROUND_LAMBDA_ACCEL"} \
    ${CONTACT_Z_THRESH:+--contact-z-thresh "$CONTACT_Z_THRESH"} \
    ${CONTACT_VXY_THRESH:+--contact-vxy-thresh "$CONTACT_VXY_THRESH"} \
    ${EXPORT_CAMERA_SPACE:+--export-camera-space} \
    ${WORLD_SCALE:+--world-scale "$WORLD_SCALE"}

