#!/bin/bash
#SBATCH --job-name=gl_final_analysis
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Final aggregation job for experiments.final_project_analysis.
# It is CPU-only: it reads CSV/checkpoint filenames and writes summary tables/plots.

if [[ -z "${WORKDIR:-}" ]]; then
  if [[ -n "${UMBRELLA:-}" ]]; then
    WORKDIR="$UMBRELLA/projects/event"
  elif [[ -d "/tudelft.net/staff-umbrella/StudentsCVlab/chiarapeppicel/projects/event" ]]; then
    WORKDIR="/tudelft.net/staff-umbrella/StudentsCVlab/chiarapeppicel/projects/event"
  else
    echo "ERROR: set WORKDIR or UMBRELLA before launching this job." >&2
    exit 1
  fi
fi

if [[ -z "${UMBRELLA:-}" ]]; then
  UMBRELLA="$(cd "$WORKDIR/../.." && pwd)"
fi

NFSHOME="${NFSHOME:-/home/nfs/chiarapeppicel}"
IMG="${IMG:-$UMBRELLA/apptainer/gl.sif}"

ROOT="${ROOT:-$WORKDIR/outputs/final_windowed4096}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$WORKDIR/outputs/windows_exp/benchmarks}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/outputs/final_report}"
TIME_WEIGHT_ROOT="${TIME_WEIGHT_ROOT:-$WORKDIR/outputs/time_weight_ablation}"
TIME_WEIGHT_EVAL_ROOT="${TIME_WEIGHT_EVAL_ROOT:-$WORKDIR/outputs/time_weight_ablation_eval}"
VISUAL_ROOT="${VISUAL_ROOT:-$WORKDIR/outputs/visual_eval $ROOT/visual_eval}"

DATASETS="${DATASETS:-dvsgesture nmnist ncaltech101}"
MODELS="${MODELS:-pointnet_ae pointnetpp_ae}"
LOSSES="${LOSSES:-chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff}"
METRICS="${METRICS:-chamfer temporal_weighted_chamfer hausdorff mse}"
PRIMARY_METRIC="${PRIMARY_METRIC:-chamfer}"
STRICT_FLAG="${STRICT_FLAG:-}"

CACHE_ROOT="${CACHE_ROOT:-$WORKDIR/cache}"
TMP_ROOT="${TMP_ROOT:-$WORKDIR/tmp}"
mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT" "$TMP_ROOT"

export TMPDIR="${TMPDIR:-/tmp/ch_final_${SLURM_JOB_ID}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/appt_final_${SLURM_JOB_ID}}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"

mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"

export \
  WORKDIR \
  UMBRELLA \
  NFSHOME \
  IMG \
  ROOT \
  BENCHMARK_ROOT \
  EVAL_ROOT \
  OUTPUT_DIR \
  TIME_WEIGHT_ROOT \
  TIME_WEIGHT_EVAL_ROOT \
  VISUAL_ROOT \
  DATASETS \
  MODELS \
  LOSSES \
  METRICS \
  PRIMARY_METRIC \
  STRICT_FLAG

# Make the variables visible even on Apptainer setups that sanitize the env.
export APPTAINERENV_WORKDIR="$WORKDIR"
export APPTAINERENV_ROOT="$ROOT"
export APPTAINERENV_BENCHMARK_ROOT="$BENCHMARK_ROOT"
export APPTAINERENV_EVAL_ROOT="$EVAL_ROOT"
export APPTAINERENV_OUTPUT_DIR="$OUTPUT_DIR"
export APPTAINERENV_TIME_WEIGHT_ROOT="$TIME_WEIGHT_ROOT"
export APPTAINERENV_TIME_WEIGHT_EVAL_ROOT="$TIME_WEIGHT_EVAL_ROOT"
export APPTAINERENV_VISUAL_ROOT="$VISUAL_ROOT"
export APPTAINERENV_DATASETS="$DATASETS"
export APPTAINERENV_MODELS="$MODELS"
export APPTAINERENV_LOSSES="$LOSSES"
export APPTAINERENV_METRICS="$METRICS"
export APPTAINERENV_PRIMARY_METRIC="$PRIMARY_METRIC"
export APPTAINERENV_STRICT_FLAG="$STRICT_FLAG"
export APPTAINERENV_MPLCONFIGDIR="$MPLCONFIGDIR"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TORCH_HOME="$TORCH_HOME"

echo "=== Host check ==="
echo "JOB_ID: ${SLURM_JOB_ID:-local}"
echo "WORKDIR: $WORKDIR"
echo "IMG: $IMG"
echo "ROOT: $ROOT"
echo "BENCHMARK_ROOT: $BENCHMARK_ROOT"
echo "EVAL_ROOT: $EVAL_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "=================="

if [[ ! -d "$WORKDIR" ]]; then
  echo "ERROR: WORKDIR does not exist: $WORKDIR" >&2
  exit 1
fi

if [[ ! -e "$IMG" && "$IMG" != docker://* ]]; then
  echo "ERROR: Apptainer image does not exist: $IMG" >&2
  exit 1
fi

apptainer exec \
  --bind "$WORKDIR":"$WORKDIR" \
  --bind "$NFSHOME":"$NFSHOME" \
  "$IMG" \
  bash -lc '
    set -euo pipefail
    cd "$WORKDIR"
    export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

    echo "=== Container check ==="
    echo "PWD: $PWD"
    echo "WORKDIR: $WORKDIR"
    python -V
    echo "======================="

    python -u -m experiments.final_project_analysis \
      --root "$ROOT" \
      --benchmark-root $BENCHMARK_ROOT \
      --eval-root "$EVAL_ROOT" \
      --time-weight-root "$TIME_WEIGHT_ROOT" \
      --time-weight-eval-root "$TIME_WEIGHT_EVAL_ROOT" \
      --visual-root $VISUAL_ROOT \
      --output-dir "$OUTPUT_DIR" \
      --datasets $DATASETS \
      --models $MODELS \
      --losses $LOSSES \
      --metrics $METRICS \
      --primary-metric "$PRIMARY_METRIC" \
      --include-emd-small \
      $STRICT_FLAG
  '

echo "Final analysis written to: $OUTPUT_DIR"
