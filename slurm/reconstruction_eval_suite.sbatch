#!/bin/bash
#SBATCH --job-name=gl_recon_eval
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-71
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
EMD_LOSSES=(emd)
EVAL_METRICS=(chamfer temporal_weighted_chamfer hausdorff mse)

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

ALL_LOSSES=("${TRAIN_LOSSES[@]}" "${EMD_LOSSES[@]}")
LOSS_INDEX=$((SLURM_ARRAY_TASK_ID % ${#ALL_LOSSES[@]}))
COMBO_INDEX=$((SLURM_ARRAY_TASK_ID / ${#ALL_LOSSES[@]}))
DATASET_INDEX=$((COMBO_INDEX / ${#MODELS[@]}))
MODEL_INDEX=$((COMBO_INDEX % ${#MODELS[@]}))

DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"
LOSS_NAME="${ALL_LOSSES[$LOSS_INDEX]}"

if [[ "$LOSS_NAME" == "emd" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT_EMD:-outputs/autoencoder_convergence_emd}"
else
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT_STANDARD:-outputs/autoencoder_convergence}"
fi

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${DATASET}_${MODEL}_${LOSS_NAME}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DATASET}_${MODEL}_${LOSS_NAME}_best.pth"
OUTPUT_DIR="outputs/eval/${DATASET}_${MODEL}_${LOSS_NAME}"
PLOT_DIR="${OUTPUT_DIR}/plots"
METRICS_STR="$(join_by ' ' "${EVAL_METRICS[@]}")"
WANDB_ARGS="$(wandb_cli_args)"

if [[ ! -f "$WORKDIR/$CHECKPOINT_PATH" ]]; then
  echo "Missing checkpoint: $WORKDIR/$CHECKPOINT_PATH" >&2
  exit 1
fi

run_in_container "
python -u -m experiments.reconstruction_corruption_eval \
  --checkpoint '$CHECKPOINT_PATH' \
  --split test \
  --metrics $METRICS_STR \
  --noise-stds 0.0 0.01 0.03 0.05 0.1 \
  --temporal-shuffle-fractions 0.0 0.1 0.25 0.5 1.0 \
  --drop-fractions 0.0 0.1 0.25 0.5 \
  --max-batches 20 \
  --device cuda \
  $WANDB_ARGS \
  --wandb-run-name recon_eval_${DATASET}_${MODEL}_${LOSS_NAME} \
  --wandb-group reconstruction_corruption_eval \
  --output '$OUTPUT_DIR/${DATASET}_${MODEL}_${LOSS_NAME}_corruptions.csv' \
  --plot-dir '$PLOT_DIR'
"
