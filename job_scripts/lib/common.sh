#!/bin/bash
# Shared helpers for the stage launchers in job_scripts/. Source this file after
# setting SB4D_CODE to the repo (or snapshot) root on the host.
#
# Common launcher arguments (handled by parse_common_args):
#   -dataset <cosilab|conflab>   dataset (default: cosilab)
#   <seq> ...                    sequence folder names (cam06_batch01, 428, ...)
#   -all <prefix>                every sequence whose folder name starts with <prefix>
#   -cam <num> -b <spec>         cosilab only: cam<num> batches, spec like 1 | 1,3,5 | 3-5 | 1,3-5
#   -sbatch "<flags>"            extra sbatch flags, appended last (e.g. "--time=4:00:00")
#   -h                           show the launcher's usage
# Env: CONFIG_REL (config path, default configs/body4d.yaml), SB4D_DRY_RUN=1
# (print sbatch/apptainer commands instead of running them), SB4D_CLUSTER.

LIB_DIR="$SB4D_CODE/job_scripts/lib"
source "$LIB_DIR/cluster.sh"
source "$LIB_DIR/dataset.sh"
source "$LIB_DIR/slurm.sh"

CONFIG="${CONFIG_REL:-configs/body4d.yaml}"

DATASET_ARG=cosilab
ALL_PREFIX=""
CAM_NUM=""
BATCH_SPEC=""
EXTRA_SBATCH=""
POS_SEQS=()

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

# Prints the comment block at the top of a launcher script.
usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$1"
}

# Handles one common argument starting at "$1"; sets COMMON_SHIFT to the number
# of words consumed. Unknown dash options are an error.
parse_common_args() {
  COMMON_SHIFT=1
  case "$1" in
    -dataset|--dataset) DATASET_ARG="${2:-}"; COMMON_SHIFT=2 ;;
    -all|--all)         ALL_PREFIX="${2:-}"; COMMON_SHIFT=2 ;;
    -cam|--cam)         CAM_NUM="${2:-}"; COMMON_SHIFT=2 ;;
    -b|--b|--batch)     BATCH_SPEC="${2:-}"; COMMON_SHIFT=2 ;;
    -sbatch|--sbatch)   EXTRA_SBATCH="${2:-}"; COMMON_SHIFT=2 ;;
    -h|--help)          usage "$LAUNCHER"; exit 0 ;;
    -*)                 die "Unknown argument: $1 (see -h)" ;;
    *)                  POS_SEQS+=("$1") ;;
  esac
  if [ "$COMMON_SHIFT" = 2 ] && [ $# -lt 2 ]; then
    die "$1 needs a value"
  fi
}

# Expands a batch spec (e.g. 1,3-5) into BATCH_NUMS, unique and in order.
expand_batch_spec() {
  local token start end bn
  local -A seen=()
  BATCH_NUMS=()
  IFS=',' read -r -a tokens <<< "$1"
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=$((10#${BASH_REMATCH[1]})); end=$((10#${BASH_REMATCH[2]}))
      (( start <= end )) || die "Invalid range in -b spec: $token (start > end)"
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      start=$((10#$token)); end=$start
    else
      die "Invalid token in -b spec: '$token' (examples: 1 | 1,3,5 | 3-5 | 1,3-5)"
    fi
    for ((bn = start; bn <= end; bn++)); do
      (( bn >= 1 )) || die "Batch numbers start at 1, got $bn"
      if [ -z "${seen[$bn]+x}" ]; then
        BATCH_NUMS+=("$bn")
        seen[$bn]=1
      fi
    done
  done
  [ ${#BATCH_NUMS[@]} -gt 0 ] || die "Empty -b spec"
}

# Builds SEQS from -all/-cam -b/positional args. -all scans <scan_dir>.
resolve_seqs() {
  local scan_dir="$1" d name seq
  local -A seen=()
  local found=()

  if [ -n "$ALL_PREFIX" ]; then
    for d in "$scan_dir/${ALL_PREFIX}"*/; do
      [ -d "$d" ] || continue
      name="$(basename "$d")"
      [[ "$name" =~ $SEQ_REGEX ]] && found+=("$name")
    done
    [ ${#found[@]} -gt 0 ] || die "No $DATASET sequences matching ${ALL_PREFIX}* in $scan_dir"
    IFS=$'\n' found=($(printf '%s\n' "${found[@]}" | sort)); unset IFS
  fi

  if [ -n "$CAM_NUM" ] || [ -n "$BATCH_SPEC" ]; then
    [ "$DATASET" = cosilab ] || die "-cam/-b are only supported for cosilab"
    [ -n "$CAM_NUM" ] && [ -n "$BATCH_SPEC" ] || die "-cam and -b must be given together"
    [[ "$CAM_NUM" =~ ^[0-9]+$ ]] || die "-cam must be a non-negative integer, got: $CAM_NUM"
    expand_batch_spec "$BATCH_SPEC"
    for bn in "${BATCH_NUMS[@]}"; do
      found+=("$(printf 'cam%02d_batch%02d' "$((10#$CAM_NUM))" "$bn")")
    done
  fi

  found+=(${POS_SEQS[@]+"${POS_SEQS[@]}"})

  SEQS=()
  for seq in ${found[@]+"${found[@]}"}; do
    [[ "$seq" =~ $SEQ_REGEX ]] || die "Sequence '$seq' does not match the $DATASET naming ($SEQ_REGEX)"
    if [ -z "${seen[$seq]+x}" ]; then
      SEQS+=("$seq")
      seen[$seq]=1
    fi
  done
  [ ${#SEQS[@]} -gt 0 ] || die "No sequences given (see -h)"
}

# Copies the repo to <dataset>/bbox_kp/_snapshots/exp_<ts>_<rand>/code and points
# SB4D_CODE at it, so queued jobs are unaffected by later edits. Falls back to
# the live repo if rsync fails.
make_snapshot() {
  local snap
  snap="$BBOX_DIR/_snapshots/exp_$(date +%Y%m%d_%H%M%S)_$(printf '%04X' "$RANDOM")/code"
  if [ "${SB4D_DRY_RUN:-0}" = 1 ]; then
    echo "[DRY-RUN] snapshot $SB4D_CODE -> $snap"
    return 0
  fi
  mkdir -p "$snap"
  echo "[INFO] Creating code snapshot in: $snap"
  if rsync -a --delete \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "outputs" \
    "$SB4D_CODE/" "$snap/"; then
    SB4D_CODE="$snap"
  else
    echo "[WARN] Failed to create code snapshot; running from live repo: $SB4D_CODE" >&2
  fi
}

# Sets BINDS to the container mounts (host:container[:ro]).
container_binds() {
  BINDS=(
    "$DATA_ROOT:$BIND_DATA"
    "$MODEL_HOST:$BIND_MODEL"
    "$HOME_HOST:$BIND_HOME"
    ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"}
  )
}

# Maps a host path to its path inside the container (unchanged if not mounted).
host_to_container() {
  local p="$1" b src dst
  container_binds
  for b in "${BINDS[@]}"; do
    src="${b%%:*}"
    dst="${b#*:}"
    dst="${dst%%:*}"
    case "$p" in
      "$src"|"$src"/*) echo "$dst${p#"$src"}"; return 0 ;;
    esac
  done
  echo "$p"
}

# Runs <script.py> from the code dir inside the apptainer image.
run_in_container() {
  local script="$1"
  shift
  local project b
  project="$(host_to_container "$SB4D_CODE")"
  container_binds
  local cmd=(apptainer exec --nv)
  for b in "${BINDS[@]}"; do
    cmd+=(--bind "$b")
  done
  cmd+=(
    --env "PYTHONPATH=$project/models/sam3:$project:${PYTHONPATH:-}"
    --env PYOPENGL_PLATFORM=osmesa
    "$SIF_PATH"
    python "$project/$script" "$@")
  if [ "${SB4D_DRY_RUN:-0}" = 1 ]; then
    echo "[DRY-RUN] $(printf '%q ' "${cmd[@]}")"
    return 0
  fi
  "${cmd[@]}"
}
