#!/bin/bash
#SBATCH --job-name=gl_ae_w4096_loss
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-20
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ -n "${UMBRELLA:-}" ]]; then
    WORKDIR="$UMBRELLA/projects/event"
  else
    echo "Set WORKDIR or UMBRELLA before launching this job." >&2
    exit 1
  fi
fi

NFSHOME="${NFSHOME:-/home/nfs/chiarapeppicel}"
IMG="${IMG:-$UMBRELLA/apptainer/gl.sif}"
DATA_DIR="${DATA_DIR:-$UMBRELLA/projects/gl/data}"

DATASETS=(${DATASETS:-dvsgesture nmnist ncaltech101})
LOSSES=(${LOSSES:-chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff projection voxel})

TOTAL_TASKS=$((${#DATASETS[@]} * ${#LOSSES[@]}))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID must be in [0, $((TOTAL_TASKS - 1))], got $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / ${#LOSSES[@]}))
LOSS_INDEX=$((SLURM_ARRAY_TASK_ID % ${#LOSSES[@]}))
DATASET="${DATASETS[$DATASET_INDEX]}"
LOSS_NAME="${LOSSES[$LOSS_INDEX]}"

WANDB_PROJECT="${WANDB_PROJECT:-geometric-learning-project}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$NFSHOME/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$NFSHOME/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$NFSHOME/.config/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$NFSHOME/wandb_artifacts}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR"

MODEL_NAME="${MODEL_NAME:-pointnet_ae}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LATENT_DIM="${LATENT_DIM:-256}"
LR="${LR:-5e-5}"
SAVE_EVERY="${SAVE_EVERY:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/autoencoder_convergence_${DATASET}_windowed4096}"
RUN_DIR="${OUTPUT_ROOT}/${DATASET}_${MODEL_NAME}_${LOSS_NAME}"

export WORKDIR DATA_DIR NFSHOME IMG DATASET LOSS_NAME WANDB_PROJECT
export MODEL_NAME EPOCHS BATCH_SIZE TEST_BATCH_SIZE NUM_WORKERS LATENT_DIM LR SAVE_EVERY OUTPUT_ROOT RUN_DIR

apptainer exec --nv \
  --bind "$WORKDIR":"$WORKDIR" \
  --bind "$DATA_DIR":"$DATA_DIR" \
  --bind "$NFSHOME":"$NFSHOME" \
  --bind "$UMBRELLA":"$UMBRELLA" \
  "$IMG" \
  bash -lc '
    set -euo pipefail

    cd "$WORKDIR"
    export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

    echo "=== Container check ==="
    echo "IMG: $IMG"
    echo "WORKDIR: $WORKDIR"
    echo "DATA_DIR: $DATA_DIR"
    echo "DATASET: $DATASET"
    echo "LOSS_NAME: $LOSS_NAME"
    echo "RUN_DIR: $RUN_DIR"
    which python
    python -c "import sys; print(sys.executable)"
    python -c "import torch; print(\"cuda available:\", torch.cuda.is_available())"
    python -c "import tonic; print(\"tonic OK\")"
    echo "======================="

    mkdir -p "$RUN_DIR"

    DATA_ARG=()
    if python -m src.train_ae --help 2>&1 | grep -q -- "--save-to"; then
      DATA_ARG=(--save-to "$DATA_DIR")
    elif python -m src.train_ae --help 2>&1 | grep -q -- "--save_to"; then
      DATA_ARG=(--save_to "$DATA_DIR")
    fi

    WANDB_ARGS=(--wandb "$WANDB_MODE" --wandb-project "$WANDB_PROJECT")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi

    RESUME_ARGS=()
    if [[ -n "${RESUME_FROM:-}" ]]; then
      RESUME_ARGS=(--resume-from "$RESUME_FROM")
    fi

    python -u -m src.train_ae \
      --dataset "$DATASET" \
      "${DATA_ARG[@]}" \
      --model-name "$MODEL_NAME" \
      --loss-name "$LOSS_NAME" \
      --stream-mode windowed \
      --window-size 4096 \
      --window-stride 4096 \
      --num-points 4096 \
      --input-dim 3 \
      --no-shuffle-points \
      --latent-dim "$LATENT_DIM" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --test-batch-size "$TEST_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$LR" \
      --device cuda \
      --output-dir "$RUN_DIR" \
      --save-every "$SAVE_EVERY" \
      --wandb-run-name "ae_w4096_${DATASET}_${MODEL_NAME}_${LOSS_NAME}" \
      --wandb-group autoencoder_convergence_windowed4096 \
      --wandb-job-type autoencoder_convergence \
      "${RESUME_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  '
