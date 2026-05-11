#!/bin/bash
#SBATCH --job-name=gl_time_weight
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
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
WANDB_PROJECT="${WANDB_PROJECT:-geometric-learning-project}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$NFSHOME/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$NFSHOME/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$NFSHOME/.config/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$NFSHOME/wandb_artifacts}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR"

run_in_container() {
  local payload="$1"
  apptainer exec --nv \
    --bind "$WORKDIR":"$WORKDIR" \
    --bind "$NFSHOME":"$NFSHOME" \
    --bind "$UMBRELLA":"$UMBRELLA" \
    "$IMG" \
    bash -lc "
      set -euo pipefail
      cd '$WORKDIR'
      export PYTHONPATH='$WORKDIR':\${PYTHONPATH:-}

      echo '=== Container check ==='
      echo 'IMG: $IMG'
      echo 'WORKDIR: $WORKDIR'
      echo 'Python:'
      which python
      python -c 'import sys; print(sys.executable)'
      python -c 'import torch; print(\"cuda available:\", torch.cuda.is_available())'
      python -c 'import tonic; print(\"tonic OK\")'
      echo '======================='

      $payload
    "
}

DATASETS="${DATASETS:-dvsgesture nmnist}"
MODELS="${MODELS:-pointnet_ae}"
TIME_WEIGHTS="${TIME_WEIGHTS:-1 2 5 10}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_POINTS="${NUM_POINTS:-1024}"
INPUT_DIM="${INPUT_DIM:-4}"
LR="${LR:-1e-3}"
EVAL_METRIC_TIME_WEIGHT="${EVAL_METRIC_TIME_WEIGHT:-5.0}"
VISUAL_SAMPLE_INDICES="${VISUAL_SAMPLE_INDICES:-0 1 2}"
SAVE_EVERY="${SAVE_EVERY:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/time_weight_ablation}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/time_weight_ablation_eval}"
VISUAL_OUTPUT_DIR="${VISUAL_OUTPUT_DIR:-outputs/time_weight_ablation_visual}"
SKIP_EXISTING_FLAG="${SKIP_EXISTING_FLAG:---skip-existing}"
RESUME_EXISTING_FLAG="${RESUME_EXISTING_FLAG:---resume-existing}"
RUN_EVAL_FLAG="${RUN_EVAL_FLAG:---run-eval}"
RUN_VISUAL_FLAG="${RUN_VISUAL_FLAG:---run-visual}"

WANDB_ARGS="--wandb ${WANDB_MODE} --wandb-project $WANDB_PROJECT"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"
fi

run_in_container "
python -u -m experiments.time_weight_ablation \
  --datasets $DATASETS \
  --models $MODELS \
  --time-weights $TIME_WEIGHTS \
  --num-points $NUM_POINTS \
  --input-dim $INPUT_DIM \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --save-every $SAVE_EVERY \
  --device cuda \
  --eval-metric-time-weight $EVAL_METRIC_TIME_WEIGHT \
  --visual-sample-indices $VISUAL_SAMPLE_INDICES \
  --output-dir '$OUTPUT_DIR' \
  --eval-output-dir '$EVAL_OUTPUT_DIR' \
  --visual-output-dir '$VISUAL_OUTPUT_DIR' \
  $RUN_EVAL_FLAG \
  $RUN_VISUAL_FLAG \
  $SKIP_EXISTING_FLAG \
  $RESUME_EXISTING_FLAG \
  $WANDB_ARGS
"