#!/bin/bash
set -euo pipefail

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ -n "${UMBRELLA:-}" ]]; then
    WORKDIR="$UMBRELLA/projects/gl"
  else
    echo "Set WORKDIR or UMBRELLA before launching these Slurm jobs." >&2
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

BENCHMARK_LOSSES=(
  chamfer
  density_aware_chamfer
  sinkhorn
  temporal_weighted_chamfer
  hausdorff
  projection
  voxel
)

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
