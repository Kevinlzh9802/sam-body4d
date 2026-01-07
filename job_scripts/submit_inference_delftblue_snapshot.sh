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
#   bash job_scripts/submit_inference_delftblue_snapshot.sh
#   bash job_scripts/submit_inference_delftblue_snapshot.sh --video cam04_cut_10s.mp4
#
# You can still edit your repo after submission; the job will run the frozen snapshot.

home_path=/home/zli33
scratch_path=/scratch/zli33

# Where the job reads inputs/outputs (matches inference_test_delftblue.sh bindings)
data_path=$scratch_path/data/sam4d
output_folder=$data_path/outputs
input_folder=$data_path/inputs

repo_dir=$home_path/projects/sam-body4d
job_script=$repo_dir/job_scripts/inference_test_delftblue.sh

video_rel="cam04_cut_10s.mp4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      video_rel="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

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

echo "[INFO] Submitting job with EXP_DIR=$exp_dir"
echo "[INFO] Video: $input_folder/$video_rel"

# Pass EXP_DIR so the compute job uses the snapshot.
# Also pass VIDEO so you can change it without editing the job script.
sbatch --export=ALL,EXP_DIR=$exp_dir,VIDEO_REL=$video_rel "$job_script"

echo "[OK] Submitted. Snapshot frozen at submit time."
echo "     EXP_DIR=$exp_dir"


