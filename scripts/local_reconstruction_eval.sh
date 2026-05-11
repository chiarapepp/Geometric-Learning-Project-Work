#!/usr/bin/env bash
set -euo pipefail

# Local equivalent of slurm/reconstruction_eval_suite.sbatch.
# Usage:
#   ./scripts/local_reconstruction_eval.sh 0
#   ./scripts/local_reconstruction_eval.sh all
#   WANDB_MODE=disabled DEVICE=cpu ./scripts/local_reconstruction_eval.sh 0

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WANDB_PROJECT="${WANDB_PROJECT:-geometric-learning-project}"
WANDB_ROOT="${WANDB_ROOT:-$WORKDIR/.wandb_local}"
DEVICE="${DEVICE:-cuda}"
TASK="${1:-0}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_ROOT/config}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_ROOT/artifacts}"
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
ALL_LOSSES=("${TRAIN_LOSSES[@]}" "${EMD_LOSSES[@]}")
TOTAL_TASKS=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#ALL_LOSSES[@]}))

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

wandb_cli_args() {
  if [[ "$WANDB_MODE" == "disabled" ]]; then
    printf "%s" "--wandb disabled"
    return
  fi

  local args="--wandb online --wandb-project $WANDB_PROJECT"
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    args="$args --wandb-entity $WANDB_ENTITY"
  fi
  printf "%s" "$args"
}

run_task() {
  local task_id="$1"
  if (( task_id < 0 || task_id >= TOTAL_TASKS )); then
    echo "Task index must be between 0 and $((TOTAL_TASKS - 1)); got $task_id." >&2
    exit 1
  fi

  local loss_index=$((task_id % ${#ALL_LOSSES[@]}))
  local combo_index=$((task_id / ${#ALL_LOSSES[@]}))
  local dataset_index=$((combo_index / ${#MODELS[@]}))
  local model_index=$((combo_index % ${#MODELS[@]}))

  local dataset="${DATASETS[$dataset_index]}"
  local model="${MODELS[$model_index]}"
  local loss_name="${ALL_LOSSES[$loss_index]}"

  local checkpoint_root
  if [[ "$loss_name" == "emd" ]]; then
    checkpoint_root="${CHECKPOINT_ROOT_EMD:-outputs/autoencoder_convergence_emd}"
  else
    checkpoint_root="${CHECKPOINT_ROOT_STANDARD:-outputs/autoencoder_convergence}"
  fi

  local checkpoint_dir="${checkpoint_root}/${dataset}_${model}_${loss_name}"
  local checkpoint_path="${checkpoint_dir}/${dataset}_${model}_${loss_name}_best.pth"
  local output_dir="outputs/eval/${dataset}_${model}_${loss_name}"
  local plot_dir="${output_dir}/plots"
  local metrics_str
  local wandb_args
  metrics_str="$(join_by ' ' "${EVAL_METRICS[@]}")"
  wandb_args="$(wandb_cli_args)"

  if [[ ! -f "$WORKDIR/$checkpoint_path" ]]; then
    echo "Skipping task $task_id: missing checkpoint $WORKDIR/$checkpoint_path" >&2
    return 0
  fi

  echo "Running task $task_id/$((TOTAL_TASKS - 1)): $dataset $model $loss_name"
  (
    cd "$WORKDIR"
    python -u -m experiments.reconstruction_corruption_eval \
      --checkpoint "$checkpoint_path" \
      --split test \
      --metrics $metrics_str \
      --noise-stds 0.0 0.01 0.03 0.05 0.1 \
      --temporal-shuffle-fractions 0.0 0.1 0.25 0.5 1.0 \
      --drop-fractions 0.0 0.1 0.25 0.5 \
      --max-batches "${MAX_BATCHES:-20}" \
      --device "$DEVICE" \
      $wandb_args \
      --wandb-run-name "recon_eval_${dataset}_${model}_${loss_name}" \
      --wandb-group reconstruction_corruption_eval \
      --output "$output_dir/${dataset}_${model}_${loss_name}_corruptions.csv" \
      --plot-dir "$plot_dir"
  )
}

if [[ "$TASK" == "all" ]]; then
  for task_id in $(seq 0 $((TOTAL_TASKS - 1))); do
    run_task "$task_id"
  done
else
  run_task "$TASK"
fi
