#!/bin/bash
#SBATCH --job-name=gl_tw_w4096
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# =========================
# Paths
# =========================

NFSHOME="${NFSHOME:-/home/nfs/chiarapeppicel}"

if [[ -z "${UMBRELLA:-}" ]]; then
  echo "ERROR: UMBRELLA is not set. Please export UMBRELLA before sbatch." >&2
  exit 1
fi

WORKDIR="${WORKDIR:-$UMBRELLA/projects/event}"
IMG="${IMG:-$UMBRELLA/apptainer/gl.sif}"
DATA_DIR="${DATA_DIR:-$UMBRELLA/projects/gl/data}"

JOB_ROOT="${JOB_ROOT:-$UMBRELLA/projects/event/slurm_runs}"
CACHE_ROOT="${CACHE_ROOT:-$UMBRELLA/projects/event/cache}"
TMP_ROOT="${TMP_ROOT:-$UMBRELLA/projects/event/tmp}"

mkdir -p "$JOB_ROOT" "$CACHE_ROOT" "$TMP_ROOT" "$DATA_DIR"

# =========================
# Avoid filling $HOME + avoid AF_UNIX path too long
# =========================

export TMPDIR="/tmp/ch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export APPTAINER_TMPDIR="/tmp/appt_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"

export APPTAINER_CACHEDIR="$CACHE_ROOT/apptainer_cache"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"

export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"

export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

mkdir -p \
  "$TMPDIR" \
  "$APPTAINER_TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$APPTAINER_CACHEDIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$PIP_CACHE_DIR"

# =========================
# CPU/thread control
# =========================

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCH_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# =========================
# Dataset/model/time-weight combinations
# =========================

DATASETS=(${DATASETS:-dvsgesture nmnist ncaltech101})
MODELS=(${MODELS:-pointnet_ae pointnetpp_ae})
TIME_WEIGHTS=(${TIME_WEIGHTS:-2 5})

LOSS_NAME="temporal_weighted_chamfer"

TOTAL_TASKS=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#TIME_WEIGHTS[@]}))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, $((TOTAL_TASKS - 1))], got $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

TASKS_PER_DATASET=$((${#MODELS[@]} * ${#TIME_WEIGHTS[@]}))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_DATASET))
REST_INDEX=$((SLURM_ARRAY_TASK_ID % TASKS_PER_DATASET))
MODEL_INDEX=$((REST_INDEX / ${#TIME_WEIGHTS[@]}))
TIME_WEIGHT_INDEX=$((REST_INDEX % ${#TIME_WEIGHTS[@]}))

DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL_NAME="${MODELS[$MODEL_INDEX]}"
LOSS_TIME_WEIGHT="${TIME_WEIGHTS[$TIME_WEIGHT_INDEX]}"
TW_TAG="${LOSS_TIME_WEIGHT//./p}"

# =========================
# W&B outside home
# =========================

WANDB_PROJECT="${WANDB_PROJECT:-GL-project-work}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$JOB_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_ROOT/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$CACHE_ROOT/wandb_config}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$JOB_ROOT/wandb_artifacts}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

mkdir -p \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_ARTIFACT_DIR"

# =========================
# Training parameters
# =========================

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LATENT_DIM="${LATENT_DIM:-256}"
DECODER_HIDDEN_DIM="${DECODER_HIDDEN_DIM:-512}"
LR="${LR:-5e-5}"
SAVE_EVERY="${SAVE_EVERY:-1}"

STREAM_MODE="${STREAM_MODE:-windowed}"
WINDOW_SIZE="${WINDOW_SIZE:-4096}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4096}"
NUM_POINTS="${NUM_POINTS:-4096}"
INPUT_DIM="${INPUT_DIM:-3}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$JOB_ROOT/outputs/time_weight_ablation_windowed4096}"
RUN_ID="${DATASET}_${MODEL_NAME}_${LOSS_NAME}_tw${TW_TAG}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

mkdir -p "$RUN_DIR"

# Resume from the latest periodic checkpoint by default if this task is relaunched.
AUTO_RESUME="${AUTO_RESUME:-1}"
if [[ -z "${RESUME_FROM:-}" && "$AUTO_RESUME" == "1" ]]; then
  shopt -s nullglob
  CHECKPOINTS=("$RUN_DIR/${DATASET}_${MODEL_NAME}_${LOSS_NAME}_epoch_"*.pth)
  shopt -u nullglob
  if (( ${#CHECKPOINTS[@]} > 0 )); then
    IFS=$'\n' SORTED_CHECKPOINTS=($(printf '%s\n' "${CHECKPOINTS[@]}" | sort -V))
    unset IFS
    LAST_INDEX=$((${#SORTED_CHECKPOINTS[@]} - 1))
    RESUME_FROM="${SORTED_CHECKPOINTS[$LAST_INDEX]}"
  fi
fi

# =========================
# Export variables for container
# =========================

export \
  UMBRELLA \
  WORKDIR \
  DATA_DIR \
  NFSHOME \
  IMG \
  DATASET \
  MODEL_NAME \
  LOSS_NAME \
  LOSS_TIME_WEIGHT \
  WANDB_PROJECT \
  EPOCHS \
  BATCH_SIZE \
  TEST_BATCH_SIZE \
  NUM_WORKERS \
  LATENT_DIM \
  DECODER_HIDDEN_DIM \
  LR \
  SAVE_EVERY \
  STREAM_MODE \
  WINDOW_SIZE \
  WINDOW_STRIDE \
  NUM_POINTS \
  INPUT_DIM \
  OUTPUT_ROOT \
  RUN_DIR \
  RUN_ID \
  JOB_ROOT \
  CACHE_ROOT \
  TMP_ROOT \
  TMPDIR \
  XDG_CACHE_HOME \
  HF_HOME \
  TRANSFORMERS_CACHE \
  HF_DATASETS_CACHE \
  TORCH_HOME \
  MPLCONFIGDIR \
  NUMBA_CACHE_DIR \
  PIP_CACHE_DIR \
  APPTAINER_CACHEDIR \
  APPTAINER_TMPDIR \
  SINGULARITY_CACHEDIR \
  SINGULARITY_TMPDIR \
  WANDB_MODE \
  WANDB_DIR \
  WANDB_CACHE_DIR \
  WANDB_CONFIG_DIR \
  WANDB_ARTIFACT_DIR \
  WANDB_START_METHOD \
  OMP_NUM_THREADS \
  MKL_NUM_THREADS \
  OPENBLAS_NUM_THREADS \
  NUMEXPR_NUM_THREADS \
  TORCH_NUM_THREADS

if [[ -n "${RESUME_FROM:-}" ]]; then
  export RESUME_FROM
fi

echo "=== Host check ==="
echo "JOB_ID: ${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "UMBRELLA: $UMBRELLA"
echo "WORKDIR: $WORKDIR"
echo "IMG: $IMG"
echo "DATA_DIR: $DATA_DIR"
echo "DATASET: $DATASET"
echo "MODEL_NAME: $MODEL_NAME"
echo "LOSS_NAME: $LOSS_NAME"
echo "LOSS_TIME_WEIGHT: $LOSS_TIME_WEIGHT"
echo "RUN_DIR: $RUN_DIR"
echo "RESUME_FROM: ${RESUME_FROM:-}"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_MODE: $WANDB_MODE"
echo "WANDB_DIR: $WANDB_DIR"
echo "WANDB_CACHE_DIR: $WANDB_CACHE_DIR"
echo "WANDB_ARTIFACT_DIR: $WANDB_ARTIFACT_DIR"
echo "APPTAINER_CACHEDIR: $APPTAINER_CACHEDIR"
echo "APPTAINER_TMPDIR: $APPTAINER_TMPDIR"
echo "TMPDIR: $TMPDIR"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "SAVE_EVERY: $SAVE_EVERY"
echo "STREAM_MODE: $STREAM_MODE"
echo "WINDOW_SIZE: $WINDOW_SIZE"
echo "WINDOW_STRIDE: $WINDOW_STRIDE"
echo "NUM_POINTS: $NUM_POINTS"
echo "INPUT_DIM: $INPUT_DIM"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
echo "TORCH_NUM_THREADS: $TORCH_NUM_THREADS"
echo "=================="

# =========================
# Run container
# =========================

apptainer exec --nv \
  --bind "$WORKDIR":"$WORKDIR" \
  --bind "$DATA_DIR":"$DATA_DIR" \
  --bind "$UMBRELLA":"$UMBRELLA" \
  --bind "$NFSHOME":"$NFSHOME" \
  --bind "$JOB_ROOT":"$JOB_ROOT" \
  --bind "$CACHE_ROOT":"$CACHE_ROOT" \
  --bind "$TMP_ROOT":"$TMP_ROOT" \
  --bind "$TMPDIR":"$TMPDIR" \
  --bind "$APPTAINER_TMPDIR":"$APPTAINER_TMPDIR" \
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
    echo "MODEL_NAME: $MODEL_NAME"
    echo "LOSS_NAME: $LOSS_NAME"
    echo "LOSS_TIME_WEIGHT: $LOSS_TIME_WEIGHT"
    echo "RUN_ID: $RUN_ID"
    echo "RUN_DIR: $RUN_DIR"
    echo "RESUME_FROM: ${RESUME_FROM:-}"
    echo "WANDB_PROJECT: $WANDB_PROJECT"
    echo "WANDB_MODE: $WANDB_MODE"
    echo "WANDB_DIR: $WANDB_DIR"
    echo "WANDB_CACHE_DIR: $WANDB_CACHE_DIR"
    echo "WANDB_CONFIG_DIR: $WANDB_CONFIG_DIR"
    echo "WANDB_ARTIFACT_DIR: $WANDB_ARTIFACT_DIR"
    echo "TMPDIR: $TMPDIR"
    echo "XDG_CACHE_HOME: $XDG_CACHE_HOME"
    echo "HF_HOME: $HF_HOME"
    echo "TORCH_HOME: $TORCH_HOME"
    echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
    echo "NUM_WORKERS: $NUM_WORKERS"
    echo "SAVE_EVERY: $SAVE_EVERY"
    echo "STREAM_MODE: $STREAM_MODE"
    echo "WINDOW_SIZE: $WINDOW_SIZE"
    echo "WINDOW_STRIDE: $WINDOW_STRIDE"
    echo "NUM_POINTS: $NUM_POINTS"
    echo "INPUT_DIM: $INPUT_DIM"
    echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
    echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
    echo "NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
    echo "TORCH_NUM_THREADS: $TORCH_NUM_THREADS"
    which python
    python -c "import sys; print(sys.executable)"
    python -c "import torch; print(\"cuda available:\", torch.cuda.is_available())"
    python -c "import torch; torch.set_num_threads(int(\"$TORCH_NUM_THREADS\")); print(\"torch num threads:\", torch.get_num_threads())"
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
      --loss-time-weight "$LOSS_TIME_WEIGHT" \
      --stream-mode "$STREAM_MODE" \
      --window-size "$WINDOW_SIZE" \
      --window-stride "$WINDOW_STRIDE" \
      --num-points "$NUM_POINTS" \
      --input-dim "$INPUT_DIM" \
      --no-shuffle-points \
      --latent-dim "$LATENT_DIM" \
      --decoder-hidden-dim "$DECODER_HIDDEN_DIM" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --test-batch-size "$TEST_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$LR" \
      --device cuda \
      --output-dir "$RUN_DIR" \
      --save-every "$SAVE_EVERY" \
      --wandb-run-name "ae_w4096_${DATASET}_${MODEL_NAME}_${LOSS_NAME}_tw${LOSS_TIME_WEIGHT}" \
      --wandb-group time_weight_ablation_windowed4096 \
      --wandb-job-type autoencoder_time_weight_ablation \
      "${RESUME_ARGS[@]}" \
      "${WANDB_ARGS[@]}"
  '

rm -rf "$TMPDIR" "$APPTAINER_TMPDIR"
