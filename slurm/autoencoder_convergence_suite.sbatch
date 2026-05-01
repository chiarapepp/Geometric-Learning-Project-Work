#!/bin/bash
#SBATCH --job-name=gl_ae_conv
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-8
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
MODELS=(pointnet_ae pointnet_vae pointnetpp_ae)
TRAIN_LOSSES=(
  chamfer
  density_aware_chamfer
  sinkhorn
  temporal_weighted_chamfer
  hausdorff
  projection
  voxel
)

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

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#MODELS[@]}))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % ${#MODELS[@]}))
DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

TRAIN_LOSSES_STR="$(join_by ' ' "${TRAIN_LOSSES[@]}")"

NUM_POINTS="${NUM_POINTS:-1024}"
INPUT_DIM="${INPUT_DIM:-4}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
LATENT_DIM="${LATENT_DIM:-256}"
DECODER_HIDDEN_DIM="${DECODER_HIDDEN_DIM:-512}"
TEMPORAL_WEIGHT="${TEMPORAL_WEIGHT:-1.0}"
LOSS_TIME_WEIGHT="${LOSS_TIME_WEIGHT:-1.0}"
KL_WEIGHT="${KL_WEIGHT:-1e-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/autoencoder_convergence}"
WANDB_ARGS="$(wandb_cli_args)"

run_in_container "
python -u -m experiments.autoencoder_convergence \
  --dataset '$DATASET' \
  --save-to ./data \
  --model-name '$MODEL' \
  --losses $TRAIN_LOSSES_STR \
  --num-points '$NUM_POINTS' \
  --input-dim '$INPUT_DIM' \
  --temporal-weight '$TEMPORAL_WEIGHT' \
  --loss-time-weight '$LOSS_TIME_WEIGHT' \
  --epochs '$EPOCHS' \
  --batch-size '$BATCH_SIZE' \
  --test-batch-size '$TEST_BATCH_SIZE' \
  --num-workers '$NUM_WORKERS' \
  --lr '$LR' \
  --latent-dim '$LATENT_DIM' \
  --decoder-hidden-dim '$DECODER_HIDDEN_DIM' \
  --kl-weight '$KL_WEIGHT' \
  --device cuda \
  $WANDB_ARGS \
  --wandb-run-name ae_convergence_${DATASET}_${MODEL} \
  --wandb-group autoencoder_convergence \
  --output-dir '$OUTPUT_ROOT'
"
