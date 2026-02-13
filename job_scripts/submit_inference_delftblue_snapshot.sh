#!/bin/bash
set -euo pipefail

# Submit helper for DelftBlue (Option 2: snapshot at submit time).
#
# This script:
# - Creates a unique exp folder under the cluster output directory
# - Rsyncs the current repo into exp_dir/code (freezing the code version at submit time)
# - Submits the Slurm job, passing EXP_DIR to the job script
#
# Usage (run on login node):
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode masklets --video cam04_cut_10s.mp4
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode raw_params --input /scratch/.../outputs/exp_XXXX
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../outputs/exp_XXXX
#
# Smoothing options (for --mode smooth):
#   --no-option1              Disable built-in smoothing (EMA/Kalman + shape/scale freeze). Default: ON
#   --no-mask-reproj          Disable mask-based reprojection optimization. Default: ON (uses Stage 1 masks)
#   --enable-kps-reproj       Enable legacy keypoint-based reprojection (requires --bbox-kps-pkl). Default: OFF
#   --enable-ground           Enable Ground-Plane / Contact Optimization. Default: OFF
#   --camera-intrinsics-json <path>  Path to camera intrinsics JSON (required for mask reproj and kps reproj)
#   --camera-scale <float>    Camera scale factor (default: 0.5)
#   --bbox-kps-pkl <path>     Path to bboxes_kps pickle (required for --enable-kps-reproj)
#   --extrinsics-json <path>  Path to camera extrinsics JSON (required for --enable-ground and world-space export)
#   --export-camera-space     Export meshes in camera coordinates instead of world space (default is world space)
#   --world-scale <float>     Scale factor to convert SMPL-X meters to extrinsic units (default: 100.0 for cm)
#                             Use 1.0 if your extrinsics are already in meters.
#
# NOTE: Paths can be specified as HOST paths (e.g. /scratch/zli33/data/sam4d/inputs/...)
#       and will be automatically translated to container paths (/mnt/data/sam4d_body/...).
#
# Examples:
#   # Smooth with all defaults (Option1 ON, Mask Reproj ON, Kps Reproj OFF, Ground OFF)
#   # NOTE: Mask reproj requires --camera-intrinsics-json; auto-uses masks from <exp>/masklets/masks/
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../exp_XXXX \
#     --camera-intrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/parameters-camera-04.json
#
#   # Smooth with mask reproj disabled (no 2D optimization)
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../exp_XXXX \
#     --no-mask-reproj
#
#   # Smooth with legacy keypoint-based reprojection (requires bbox and intrinsics paths)
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../exp_XXXX \
#     --no-mask-reproj \
#     --enable-kps-reproj \
#     --bbox-kps-pkl /scratch/zli33/data/sam4d/inputs/bboxes_kps_refined/428.pkl \
#     --camera-intrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/parameters-camera-04.json
#
#   # Smooth with ground-plane optimization (exports in world space by default)
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../exp_XXXX \
#     --camera-intrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/parameters-camera-04.json \
#     --enable-ground \
#     --extrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/extrinsics-camera-04.json
#
#   # Smooth with ground optimization but export in camera space (legacy behavior)
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --mode smooth --input /scratch/.../exp_XXXX \
#     --camera-intrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/parameters-camera-04.json \
#     --enable-ground \
#     --extrinsics-json /scratch/zli33/data/sam4d/inputs/camera_params_new/extrinsics-camera-04.json \
#     --export-camera-space
#
# You can still edit your repo after submission; the job will run the frozen snapshot.

home_path=/home/zli33
scratch_path=/scratch/zli33

# Where the job reads inputs/outputs (matches inference_test_delftblue.sh bindings)
data_path=$scratch_path/data/sam4d
output_folder=$data_path/outputs
input_folder=$data_path/inputs

repo_dir=$home_path/projects/sam-body4d
job_script_masklets=$repo_dir/job_scripts/masklets_test_delftblue.sh
job_script_meshes=$repo_dir/job_scripts/meshes_test_delftblue.sh
job_script_raw_params=$repo_dir/job_scripts/raw_params_test_delftblue.sh
job_script_smooth=$repo_dir/job_scripts/smooth_meshes_delftblue.sh

video_rel="cam04_cut_20s.mp4"
mode="masklets" # masklets | raw_params | smooth
exp_dir_override=""
input_dir=""
stage1_dir_host=""
s2_dir_host=""
s2_code_dir_host=""
use_live_code="0"

# Smoothing options (for --mode smooth)
# Option 1: Built-in smoothing (EMA/Kalman + shape/scale freeze) - default ON
# Mask Reproj: Mask-based reprojection optimization - default ON (uses Stage 1 masks)
# Kps Reproj: Legacy keypoint-based reprojection - default OFF
# Ground: Ground-Plane / Contact Optimization - default OFF
smooth_no_option1=""
smooth_no_mask_reproj=""
smooth_enable_kps_reproj=""
smooth_enable_ground=""
# Reproj/Ground inputs (host paths - will be translated to container paths)
bbox_kps_pkl_host=""
camera_intrinsics_json_host=""
camera_scale=""
extrinsics_json_host=""
export_camera_space=""
world_scale="100.0"  # Default: convert SMPL-X meters to centimeters

# Translate a host path under $data_path into the corresponding container path.
# This is defined early so it can be used in argument validation.
host_to_container_path() {
  local p="$1"
  case "$p" in
    "$data_path"/*) echo "/mnt/data/sam4d_body/${p#"$data_path"/}" ;;
    *) echo "$p" ;;  # Return unchanged if not under data_path
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
      shift 2
      ;;
    --video)
      video_rel="$2"
      shift 2
      ;;
    --input|--input-dir|--stage1-dir)
      input_dir="$2"
      shift 2
      ;;
    --exp-dir)
      exp_dir_override="$2"
      shift 2
      ;;
    # Smoothing options (for --mode smooth)
    --no-option1|--no-smooth)
      smooth_no_option1="1"
      shift
      ;;
    --no-mask-reproj)
      smooth_no_mask_reproj="1"
      shift
      ;;
    --enable-kps-reproj|--kps-reproj)
      smooth_enable_kps_reproj="1"
      shift
      ;;
    --enable-ground|--ground)
      smooth_enable_ground="1"
      shift
      ;;
    # Reproj/Ground input paths (can be host paths - will be auto-translated)
    --bbox-kps-pkl)
      bbox_kps_pkl_host="$2"
      shift 2
      ;;
    --camera-intrinsics-json|--camera-intrinsics)
      camera_intrinsics_json_host="$2"
      shift 2
      ;;
    --camera-scale)
      camera_scale="$2"
      shift 2
      ;;
    --extrinsics-json|--extrinsics)
      extrinsics_json_host="$2"
      shift 2
      ;;
    --export-camera-space|--camera-space)
      export_camera_space="1"
      shift
      ;;
    --world-scale)
      world_scale="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "$mode" = "raw_params" ]; then
  # For raw params, the user provides the experiment folder (.../exp_XXXX) and we
  # auto-complete the Stage1 folder as <exp_XXXX>/masklets.
  if [ -z "${input_dir:-}" ]; then
    echo "[ERROR] --mode raw_params requires --input <exp_dir_host> (e.g. .../exp_XXXX)" >&2
    exit 2
  fi

  # Require the experiment dir (exp_XXXX) only; always derive Stage1 as <exp_XXXX>/masklets.
  if [ "$(basename "$input_dir")" = "masklets" ]; then
    echo "[ERROR] --input must be the experiment folder (e.g. .../exp_XXXX), not .../exp_XXXX/masklets" >&2
    exit 2
  fi
  exp_dir_from_input="$input_dir"
  stage1_dir_host="$input_dir/masklets"

  # If the user didn't explicitly set --exp-dir, use the one derived from --input.
  if [ -z "${exp_dir_override:-}" ]; then
    exp_dir_override="$exp_dir_from_input"
  fi
fi

if [ "$mode" = "smooth" ]; then
  # For smoothing, the user provides the experiment folder (.../exp_XXXX) only.
  if [ -z "${input_dir:-}" ]; then
    echo "[ERROR] --mode smooth requires --input <exp_dir_host> (e.g. .../exp_XXXX)" >&2
    exit 2
  fi
  if [ "$(basename "$input_dir")" = "masklets" ]; then
    echo "[ERROR] --input must be the experiment folder (e.g. .../exp_XXXX), not .../exp_XXXX/masklets" >&2
    exit 2
  fi
  exp_dir_from_input="$input_dir"
  if [ -z "${exp_dir_override:-}" ]; then
    exp_dir_override="$exp_dir_from_input"
  fi
fi

if [ -n "${exp_dir_override:-}" ]; then
  exp_dir="$exp_dir_override"
  echo "[INFO] Using existing EXP_DIR: $exp_dir"
  if [ ! -d "$exp_dir" ]; then
    echo "[ERROR] --exp-dir does not exist: $exp_dir" >&2
    exit 2
  fi
else
  timestamp=$(date +%Y%m%d_%H%M%S)
  # NOTE: avoid `tr ... | head -c 4` under `set -o pipefail` (can exit with SIGPIPE=141).
  rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
  )
  exp_dir=$output_folder/exp_${timestamp}_${rand_suffix}

  mkdir -p "$exp_dir/code"
  echo "[INFO] Creating snapshot in: $exp_dir/code"

  rsync -a --delete \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "outputs" \
    "$repo_dir/" \
    "$exp_dir/code/"
fi

echo "[INFO] Submitting job with EXP_DIR=$exp_dir"

# Pass EXP_DIR so the compute job uses the snapshot.
case "$mode" in
  masklets)
    # Stage 1 only (SAM-3 -> masks). Writes under EXP_DIR/masklets by default.
    echo "[INFO] Video: $input_folder/$video_rel"
    sbatch --export=ALL,EXP_DIR=$exp_dir,VIDEO_REL=$video_rel "$job_script_masklets"
    ;;
  raw_params)
    # Stage 2 (decoupled): masks/images -> raw params.
    # IMPORTANT: create exp_s2_*/code snapshot AT SUBMISSION TIME (not at job start).
    if [ -z "${stage1_dir_host:-}" ]; then
      echo "[ERROR] Internal error: stage1_dir_host not set for raw_params mode." >&2
      exit 2
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
    )
    s2_dir_host="$exp_dir/exp_s2_${timestamp}_${rand_suffix}"
    s2_code_dir_host="$s2_dir_host/code"
    mkdir -p "$s2_code_dir_host"
    echo "[INFO] Creating stage-2 snapshot in: $s2_code_dir_host"

    if rsync -a --delete \
      --exclude ".git" \
      --exclude "__pycache__" \
      --exclude "*.pyc" \
      --exclude "outputs" \
      "$repo_dir/" \
      "$s2_code_dir_host/" ; then
      use_live_code="0"
    else
      echo "[WARN] Failed to snapshot stage-2 code at submission; job will use live repo at runtime." >&2
      use_live_code="1"
      s2_code_dir_host=""
    fi

    echo "[INFO] Input (Stage1 dir): $stage1_dir_host"
    echo "[INFO] EXP_S2_DIR: $s2_dir_host"
    sbatch --export=ALL,EXP_DIR=$exp_dir,S2_CODE_DIR_HOST=$s2_code_dir_host,USE_LIVE_CODE=$use_live_code \
      "$job_script_raw_params" "$stage1_dir_host"
    ;;
  smooth)
    # Stage 3: smooth raw params and export meshes.
    # IMPORTANT: create exp_s3_*/code snapshot AT SUBMISSION TIME (not at job start).
    if [ -z "${exp_dir_override:-}" ]; then
      echo "[ERROR] Internal error: exp_dir not set for smooth mode." >&2
      exit 2
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    rand_suffix=$(python3 - <<'PY'
import random, string
print("".join(random.choices(string.ascii_uppercase + string.digits, k=4)))
PY
    )
    s3_dir_host="$exp_dir/exp_s3_${timestamp}_${rand_suffix}"
    s3_code_dir_host="$s3_dir_host/code"
    mkdir -p "$s3_code_dir_host"
    echo "[INFO] Creating stage-3 snapshot in: $s3_code_dir_host"

    if rsync -a --delete \
      --exclude ".git" \
      --exclude "__pycache__" \
      --exclude "*.pyc" \
      --exclude "outputs" \
      "$repo_dir/" \
      "$s3_code_dir_host/" ; then
      use_live_code="0"
    else
      echo "[WARN] Failed to snapshot stage-3 code at submission; job will use live repo at runtime." >&2
      use_live_code="1"
      s3_code_dir_host=""
    fi

    echo "[INFO] EXP_DIR: $exp_dir"
    echo "[INFO] EXP_S3_DIR: $s3_dir_host"
    
    # Build smoothing options string for sbatch --export
    smooth_opts="EXP_DIR=$exp_dir,EXP_S3_DIR=$s3_dir_host,S3_CODE_DIR_HOST=$s3_code_dir_host,USE_LIVE_CODE=$use_live_code"
    if [ -n "${smooth_no_option1:-}" ]; then
      smooth_opts="$smooth_opts,NO_OPTION1=1"
      echo "[INFO] Smoothing: Option1 (built-in) DISABLED"
    else
      echo "[INFO] Smoothing: Option1 (built-in) enabled (default)"
    fi
    
    # Mask-based reprojection (default ON) - uses Stage 1 masks from <exp>/masklets/masks/
    if [ -n "${smooth_no_mask_reproj:-}" ]; then
      smooth_opts="$smooth_opts,NO_MASK_REPROJ=1"
      echo "[INFO] Smoothing: Mask Reprojection DISABLED"
    else
      echo "[INFO] Smoothing: Mask Reprojection enabled (default)"
      # Mask reproj requires camera intrinsics
      if [ -z "${camera_intrinsics_json_host:-}" ]; then
        echo "[WARN] Mask reproj requires --camera-intrinsics-json. Will be disabled at runtime if missing."
      fi
    fi
    
    # Legacy keypoint-based reprojection (default OFF)
    if [ -n "${smooth_enable_kps_reproj:-}" ]; then
      smooth_opts="$smooth_opts,ENABLE_KPS_REPROJ=1"
      echo "[INFO] Smoothing: Legacy Keypoint Reprojection ENABLED"
      
      # Validate required inputs for kps reproj
      if [ -z "${bbox_kps_pkl_host:-}" ]; then
        echo "[ERROR] --enable-kps-reproj requires --bbox-kps-pkl <path>" >&2
        exit 2
      fi
      if [ -z "${camera_intrinsics_json_host:-}" ]; then
        echo "[ERROR] --enable-kps-reproj requires --camera-intrinsics-json <path>" >&2
        exit 2
      fi
    else
      echo "[INFO] Smoothing: Legacy Keypoint Reprojection disabled (default)"
    fi
    
    if [ -n "${smooth_enable_ground:-}" ]; then
      smooth_opts="$smooth_opts,ENABLE_GROUND=1"
      echo "[INFO] Smoothing: Ground-Plane/Contact ENABLED"
      
      # Validate required inputs for ground-plane optimization
      if [ -z "${extrinsics_json_host:-}" ]; then
        echo "[ERROR] --enable-ground requires --extrinsics-json <path>" >&2
        exit 2
      fi
    else
      echo "[INFO] Smoothing: Ground-Plane/Contact disabled (default)"
    fi
    
    # Add reproj/ground input paths (translate host -> container paths)
    if [ -n "${bbox_kps_pkl_host:-}" ]; then
      bbox_kps_pkl_container="$(host_to_container_path "$bbox_kps_pkl_host")"
      smooth_opts="$smooth_opts,BBOX_KPS_PKL=$bbox_kps_pkl_container"
      echo "[INFO] bbox_kps_pkl: $bbox_kps_pkl_host -> $bbox_kps_pkl_container"
    fi
    if [ -n "${camera_intrinsics_json_host:-}" ]; then
      camera_intrinsics_json_container="$(host_to_container_path "$camera_intrinsics_json_host")"
      smooth_opts="$smooth_opts,CAMERA_INTRINSICS_JSON=$camera_intrinsics_json_container"
      echo "[INFO] camera_intrinsics_json: $camera_intrinsics_json_host -> $camera_intrinsics_json_container"
    fi
    if [ -n "${camera_scale:-}" ]; then
      smooth_opts="$smooth_opts,CAMERA_SCALE=$camera_scale"
      echo "[INFO] camera_scale: $camera_scale"
    fi
    if [ -n "${extrinsics_json_host:-}" ]; then
      extrinsics_json_container="$(host_to_container_path "$extrinsics_json_host")"
      smooth_opts="$smooth_opts,EXTRINSICS_JSON=$extrinsics_json_container"
      echo "[INFO] extrinsics_json: $extrinsics_json_host -> $extrinsics_json_container"
    fi
    if [ -n "${export_camera_space:-}" ]; then
      smooth_opts="$smooth_opts,EXPORT_CAMERA_SPACE=1"
      echo "[INFO] Export: CAMERA space"
    else
      echo "[INFO] Export: WORLD space (default, ground plane at z=0)"
      # World space is default, but requires extrinsics - warn if missing
      if [ -z "${extrinsics_json_host:-}" ]; then
        echo "[WARN] World-space export requires --extrinsics-json. Will fall back to camera space if not provided."
      fi
    fi
    
    # World scale (default 100.0 to convert SMPL-X meters to centimeters)
    smooth_opts="$smooth_opts,WORLD_SCALE=$world_scale"
    echo "[INFO] World scale: $world_scale"
    
    sbatch --export=ALL,$smooth_opts "$job_script_smooth" "$s3_dir_host"
    ;;
  *)
    echo "[ERROR] Unknown --mode: $mode (expected: masklets|raw_params)" >&2
    exit 2
    ;;
esac

echo "[OK] Submitted. Snapshot frozen at submit time."
echo "     EXP_DIR=$exp_dir"


