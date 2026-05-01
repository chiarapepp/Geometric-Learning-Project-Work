#!/bin/bash
#SBATCH --job-name=gl_plots
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

WANDB_ARGS="$(wandb_cli_args)"

run_in_container "
set -euo pipefail

for dataset in ${DATASETS[*]}; do
  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_loss_comparison.csv \
    --kind loss_comparison \
    $WANDB_ARGS \
    --wandb-run-name plot_loss_comparison_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/loss_comparison

  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_loss_comparison_emd.csv \
    --kind loss_comparison \
    $WANDB_ARGS \
    --wandb-run-name plot_loss_comparison_emd_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/loss_comparison_emd

  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_noise_robustness.csv \
    --kind noise \
    $WANDB_ARGS \
    --wandb-run-name plot_noise_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/noise

  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_noise_robustness_emd.csv \
    --kind noise \
    $WANDB_ARGS \
    --wandb-run-name plot_noise_emd_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/noise_emd

  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_temporal_shuffle.csv \
    --kind temporal_shuffle \
    $WANDB_ARGS \
    --wandb-run-name plot_temporal_shuffle_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/temporal_shuffle

  python -u -m experiments.plot_results \
    --input outputs/benchmarks/\${dataset}_temporal_shuffle_emd.csv \
    --kind temporal_shuffle \
    $WANDB_ARGS \
    --wandb-run-name plot_temporal_shuffle_emd_\${dataset} \
    --wandb-group plots \
    --output-dir outputs/plots/\${dataset}/temporal_shuffle_emd
done

for dataset in ${DATASETS[*]}; do
  for model in ${MODELS[*]}; do
    python -u -m experiments.plot_results \
      --input outputs/autoencoder_convergence/\${dataset}_\${model}_convergence.csv \
      --kind convergence \
      $WANDB_ARGS \
      --wandb-run-name plot_convergence_\${dataset}_\${model} \
      --wandb-group plots \
      --output-dir outputs/plots/\${dataset}/convergence/\${model}

    python -u -m experiments.plot_results \
      --input outputs/autoencoder_convergence_emd/\${dataset}_\${model}_convergence.csv \
      --kind convergence \
      $WANDB_ARGS \
      --wandb-run-name plot_convergence_emd_\${dataset}_\${model} \
      --wandb-group plots \
      --output-dir outputs/plots/\${dataset}/convergence_emd/\${model}
  done
done
"
