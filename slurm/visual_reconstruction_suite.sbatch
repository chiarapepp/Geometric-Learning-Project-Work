#!/bin/bash
#SBATCH --job-name=gl_visual_eval
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ -n "${UMBRELLA:-}" ]]; then
    WORKDIR="$UMBRELLA/projects/gl"
  else
    echo "Set WORKDIR or UMBRELLA before launching this job." >&2
    exit 1
  fi
fi

NFSHOME="${NFSHOME:-/home/nfs/chiarapeppicel}"
IMG="${IMG:-$UMBRELLA/apptainer/gl.sif}"

run_in_container() {
  local payload="$1"
  apptainer exec --nv \
    --bind "$WORKDIR":"$WORKDIR" \
    --bind "$NFSHOME":"$NFSHOME" \
    --bind "$UMBRELLA":"$UMBRELLA" \
    "$IMG" \
    bash -lc "set -euo pipefail; cd '$WORKDIR'; export PYTHONPATH='$WORKDIR':\${PYTHONPATH:-}; $payload"
}

DATASETS="${DATASETS:-dvsgesture nmnist ncaltech101}"
MODELS="${MODELS:-pointnet_ae pointnet_vae pointnetpp_ae}"
LOSSES="${LOSSES:-chamfer sinkhorn temporal_weighted_chamfer hausdorff density_aware_chamfer emd}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0 1 2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/visual_eval}"
METRIC_TIME_WEIGHT="${METRIC_TIME_WEIGHT:-5.0}"
MAX_PLOT_POINTS="${MAX_PLOT_POINTS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FULL_GRID_FLAG="${FULL_GRID_FLAG:-}"

run_in_container "
python - <<'PY'
import torch
import tonic
print('tonic OK')
print('cuda available:', torch.cuda.is_available())
PY

python -u -m experiments.visual_reconstruction_eval \
  --datasets $DATASETS \
  --models $MODELS \
  --losses $LOSSES \
  --sample-indices $SAMPLE_INDICES \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --max-plot-points $MAX_PLOT_POINTS \
  --metric-time-weight $METRIC_TIME_WEIGHT \
  --device cuda \
  --output-dir '$OUTPUT_DIR' \
  $FULL_GRID_FLAG
"