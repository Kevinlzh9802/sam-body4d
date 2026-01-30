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
    sbatch --export=ALL,EXP_DIR=$exp_dir,EXP_S3_DIR=$s3_dir_host,S3_CODE_DIR_HOST=$s3_code_dir_host,USE_LIVE_CODE=$use_live_code \
      "$job_script_smooth" "$s3_dir_host"
    ;;
  *)
    echo "[ERROR] Unknown --mode: $mode (expected: masklets|raw_params)" >&2
    exit 2
    ;;
esac

echo "[OK] Submitted. Snapshot frozen at submit time."
echo "     EXP_DIR=$exp_dir"


