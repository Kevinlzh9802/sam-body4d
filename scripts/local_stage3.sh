#!/bin/bash
# Local Stage 3 + jitter attribution on one 10 s segment (no SLURM). Run from
# anywhere; everything lives under experiments/ (gitignored):
#
#   experiments/sam4d_checkpoints/sam-3d-body-dinov3/   model.ckpt, model_config.yaml,
#                                                       assets/mhr_model.pt (copy from the
#                                                       cluster's sam4d_checkpoints)
#   experiments/mhr_raw/<seq>/                          raw_mhr.pt or raw_mhr_segments/*.pt
#   experiments/sam3_output/<seq>/                      masks.zip, images.zip, id_mapping.json
#   experiments/intrinsics/parameters-camera-<MM>.json
#   experiments/extrinsics/extrinsic_cam_<MM>.json      optional (adds the ground step;
#                                                       scripts/make_ground_extrinsics.py)
#
# Output: experiments/stage3_local/<seq>/<segment>/
#   raw_mhr.pt          the segment's raw params
#   <label>/            stage3_mhr.py output (camera space, so it compares with the
#                       jitter variants), incl. diagnostics/, reproj_overlay/ and
#                       reproj_overlay.mp4 (not with --camera);
#                       label default: stage3_height with extrinsics (height option),
#                       else stage3_noground; stage3_camera with --camera
#   jitter/             jitter_report.json from scripts/jitter_check.py
#
# Usage:
#   bash scripts/local_stage3.sh cam06_batch01 seg010
#   bash scripts/local_stage3.sh cam06_batch01 10 --cpu          # no GPU
#   bash scripts/local_stage3.sh cam06_batch01 seg010 --no-stage3 # jitter attribution only
#   bash scripts/local_stage3.sh cam06_batch01 seg010 --label noground --stage3-args "--no-ground"
#   bash scripts/local_stage3.sh cam06_batch01 seg010 --stage3-args "--overlay-interval 20"
#   bash scripts/local_stage3.sh cam06_batch01 seg010 --camera --no-jitter  # multi-view input
#   bash scripts/local_stage3.sh cam06_batch01 seg010 -- --stage3-sigmas 2 4  # extra jitter_check.py args
# The multi-view option over several cameras: scripts/local_stage3_multiview.sh
# Env: SB4D_SIF (default apptainer/body4d_osmesa_cu128.sif)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

[ $# -ge 2 ] || { sed -n '2,/^set -/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2; }
SEQ="$1"; SEG="$2"; shift 2
DEVICE=cuda
RUN_STAGE3=1
RUN_JITTER=1
LABEL=""
S3_EXTRA=()
S3_MODE=()
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)       DEVICE=cpu; shift ;;
    --no-stage3) RUN_STAGE3=0; shift ;;
    --no-jitter) RUN_JITTER=0; shift ;;
    --camera)    S3_MODE=(--no-ground --no-video); LABEL="${LABEL:-stage3_camera}"; shift ;;
    --label)     LABEL="$2"; shift 2 ;;
    --stage3-args) read -r -a S3_EXTRA <<< "$2"; shift 2 ;;
    --)          shift; EXTRA=("$@"); break ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$SEQ" =~ ^cam([0-9]{2})_batch[0-9]{2}$ ]] || { echo "[ERROR] Expected a cosilab sequence like cam06_batch01, got $SEQ" >&2; exit 2; }
CAM="${BASH_REMATCH[1]}"
SEG="${SEG#seg}"
SEG_KEY="$(printf 'cam%s_seg%03d' "$CAM" "$((10#$SEG))")"

E=experiments
CKPT="$E/sam4d_checkpoints"
RAW_DIR="$E/mhr_raw/$SEQ"
STAGE1="$E/sam3_output/$SEQ"
INTR="$E/intrinsics/parameters-camera-$CAM.json"
EXTR="$E/extrinsics/extrinsic_cam_$CAM.json"
OUT="$E/stage3_local/$SEQ/$SEG_KEY"
if [ -z "$LABEL" ]; then
  if [ -f "$EXTR" ]; then LABEL=stage3_height; else LABEL=stage3_noground; fi
fi
SIF="${SB4D_SIF:-apptainer/body4d_osmesa_cu128.sif}"

# Raw source: the per-segment file if there is one, else the whole raw_mhr.pt
RAW_SRC="$(ls "$RAW_DIR"/raw_mhr_segments/raw_mhr_seg_*_"$SEG_KEY".pt 2>/dev/null | head -1 || true)"
[ -n "$RAW_SRC" ] || RAW_SRC="$RAW_DIR/raw_mhr.pt"

missing=0
for f in "$CKPT/sam-3d-body-dinov3/model.ckpt" "$CKPT/sam-3d-body-dinov3/assets/mhr_model.pt" \
         "$RAW_SRC" "$STAGE1/masks.zip" "$INTR" "$SIF"; do
  [ -e "$f" ] || { echo "[ERROR] Missing: $f" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 1

NV=()
if [ "$DEVICE" = cuda ]; then
  if nvidia-smi -L >/dev/null 2>&1; then NV=(--nv); else echo "[WARN] No GPU found; using CPU"; DEVICE=cpu; fi
fi

mkdir -p "$OUT"
# Local config: checkpoints under experiments/, no MoGe (Stage 3 never uses the FOV estimator)
CFG="$OUT/body4d_local.yaml"
sed 's|^\( *fov_path:\).*|\1 ""|' configs/body4d.yaml > "$CFG"

run() {
  apptainer exec ${NV[@]+"${NV[@]}"} --pwd /mnt/sam-body4d \
    --bind "$REPO:/mnt/sam-body4d" --bind "$REPO/$CKPT:/mnt/sam4d_checkpoints" \
    --env "PYTHONPATH=/mnt/sam-body4d/models/sam3:/mnt/sam-body4d" --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" python "$@"
}

echo "[INFO] $SEQ $SEG_KEY from $RAW_SRC -> $OUT (device: $DEVICE)"
run scripts/jitter_check.py --raw "$RAW_SRC" --segment "$SEG_KEY" --write-slice "$OUT/raw_mhr.pt"

S3_ARGS=()
if [ "$RUN_STAGE3" = 1 ]; then
  GROUND=()
  if [ -f "$EXTR" ]; then
    GROUND=(--extrinsics-json "$EXTR")
    echo "[INFO] Extrinsics: $EXTR (ground step on)"
  fi
  # Camera-space export so the jitter numbers are comparable (m, same frame as the raw variants)
  run stage3_mhr.py --raw "$OUT/raw_mhr.pt" --out "$OUT/$LABEL" --input-dir "$STAGE1" \
    --camera-intrinsics-json "$INTR" --config "$CFG" --device "$DEVICE" --export-camera-space \
    ${GROUND[@]+"${GROUND[@]}"} ${S3_MODE[@]+"${S3_MODE[@]}"} ${S3_EXTRA[@]+"${S3_EXTRA[@]}"}
  S3_ARGS=(--stage3-dir "$LABEL=$OUT/$LABEL")
fi

if [ "$RUN_JITTER" = 1 ]; then
mkdir -p "$OUT/jitter"
run scripts/jitter_check.py --raw "$OUT/raw_mhr.pt" --config "$CFG" --device "$DEVICE" --out "$OUT/jitter" \
  ${S3_ARGS[@]+"${S3_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} | tee "$OUT/jitter/jitter_report.txt"
fi

[ "$RUN_STAGE3" = 0 ] || echo "[INFO] View the meshes: python scripts/view_stage3.py $OUT/$LABEL  (needs viser)"
