#!/bin/bash
#SBATCH --job-name=gl_shuffle
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

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
IMG="${IMG:-docker://nvcr.io/nvidia/pytorch:24.01-py3}"
WANDB_PROJECT="${WANDB_PROJECT:-geometric-learning-project}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$NFSHOME/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$NFSHOME/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$NFSHOME/.config/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$NFSHOME/wandb_artifacts}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR"

DATASETS=(dvsgesture nmnist ncaltech101)
BENCHMARK_LOSSES=(
  chamfer
  density_aware_chamfer
  sinkhorn
  temporal_weighted_chamfer
  hausdorff
  projection
  voxel
)
EMD_LOSSES=(emd)

join_by() {
  local delimiter="$1"
  shift
  local first="${1:-}"
  shift || true
  printf "%s" "$first"
  for item in "$@"; do
    printf "%s%s" "$delimiter" "$item"
  done
}

run_in_container() {
  local payload="$1"
  apptainer exec --nv \
    --bind "$WORKDIR":"$WORKDIR" \
    --bind "$NFSHOME":"$NFSHOME" \
    "$IMG" \
    bash -lc "set -euo pipefail; cd '$WORKDIR'; $payload"
}

wandb_cli_args() {
  local args="--wandb online --wandb-project $WANDB_PROJECT"
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    args="$args --wandb-entity $WANDB_ENTITY"
  fi
  printf "%s" "$args"
}

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
LOSS_TIME_WEIGHT="${LOSS_TIME_WEIGHT:-1.0}"
SHUFFLE_FRACTIONS="${SHUFFLE_FRACTIONS:-0.0 0.1 0.25 0.5 1.0}"
REGULAR_NUM_POINTS="${REGULAR_NUM_POINTS:-1024}"
REGULAR_BATCH_SIZE="${REGULAR_BATCH_SIZE:-8}"
REGULAR_MAX_BATCHES="${REGULAR_MAX_BATCHES:-20}"
REGULAR_REPEATS="${REGULAR_REPEATS:-3}"
EMD_NUM_POINTS="${EMD_NUM_POINTS:-128}"
EMD_BATCH_SIZE="${EMD_BATCH_SIZE:-4}"
EMD_MAX_BATCHES="${EMD_MAX_BATCHES:-8}"
EMD_REPEATS="${EMD_REPEATS:-2}"
WANDB_ARGS="$(wandb_cli_args)"

REGULAR_LOSSES_STR="$(join_by ' ' "${BENCHMARK_LOSSES[@]}")"
EMD_LOSSES_STR="$(join_by ' ' "${EMD_LOSSES[@]}")"

run_in_container "
python -u -m experiments.temporal_shuffle \
  --dataset '$DATASET' \
  --save-to ./data \
  --losses $REGULAR_LOSSES_STR \
  --fractions $SHUFFLE_FRACTIONS \
  --num-points '$REGULAR_NUM_POINTS' \
  --input-dim 4 \
  --batch-size '$REGULAR_BATCH_SIZE' \
  --num-workers 4 \
  --max-batches '$REGULAR_MAX_BATCHES' \
  --repeats '$REGULAR_REPEATS' \
  --device cuda \
  --loss-time-weight '$LOSS_TIME_WEIGHT' \
  $WANDB_ARGS \
  --wandb-run-name temporal_shuffle_${DATASET}_core \
  --wandb-group temporal_shuffle \
  --output outputs/benchmarks/${DATASET}_temporal_shuffle.csv
"

run_in_container "
python -u -m experiments.temporal_shuffle \
  --dataset '$DATASET' \
  --save-to ./data \
  --losses $EMD_LOSSES_STR \
  --fractions $SHUFFLE_FRACTIONS \
  --num-points '$EMD_NUM_POINTS' \
  --input-dim 4 \
  --batch-size '$EMD_BATCH_SIZE' \
  --num-workers 4 \
  --max-batches '$EMD_MAX_BATCHES' \
  --repeats '$EMD_REPEATS' \
  --device cuda \
  $WANDB_ARGS \
  --wandb-run-name temporal_shuffle_${DATASET}_emd \
  --wandb-group temporal_shuffle \
  --output outputs/benchmarks/${DATASET}_temporal_shuffle_emd.csv
"
