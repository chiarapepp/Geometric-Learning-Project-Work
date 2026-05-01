# Slurm Suite

These jobs cover the full project-work workflow with the current codebase:

- loss-comparison benchmarks on all 3 datasets
- Gaussian noise robustness on all 3 datasets
- temporal shuffle robustness on all 3 datasets
- autoencoder convergence training for all 3 datasets x all 3 models
- dedicated EMD training/eval jobs with reduced point count
- reconstruction-under-corruption evaluation for every trained checkpoint
- PNG plot generation for benchmark and convergence CSV outputs

## Files

- `slurm/loss_comparison_suite.sbatch`
- `slurm/noise_robustness_suite.sbatch`
- `slurm/temporal_shuffle_suite.sbatch`
- `slurm/autoencoder_convergence_suite.sbatch`
- `slurm/autoencoder_convergence_emd_suite.sbatch`
- `slurm/reconstruction_eval_suite.sbatch`
- `slurm/plot_results_suite.sbatch`

## Assumptions

- `WORKDIR` resolves to the repository root, typically via `UMBRELLA/projects/gl`
- `NFSHOME=/home/nfs/chiarapeppicel`
- Apptainer is available and can pull `docker://nvcr.io/nvidia/pytorch:24.01-py3`
- W&B login and permissions are already configured on the cluster account

Each `.sbatch` is self-contained, so you can keep the files directly in `$HOME/slurm` and submit them from there without relying on `common_env.sh`.

## Recommended Submission Order

```bash
sbatch slurm/loss_comparison_suite.sbatch
sbatch slurm/noise_robustness_suite.sbatch
sbatch slurm/temporal_shuffle_suite.sbatch
sbatch slurm/autoencoder_convergence_suite.sbatch
sbatch slurm/autoencoder_convergence_emd_suite.sbatch
sbatch slurm/reconstruction_eval_suite.sbatch
sbatch slurm/plot_results_suite.sbatch
```

For stricter orchestration, submit later stages with Slurm dependencies after the training jobs complete.

## Coverage Notes

- Regular benchmarks and convergence runs include:
  `chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff projection voxel`
- `emd` is split into dedicated jobs with smaller `num_points` and batch size, to avoid making the entire suite impractical
- Reconstruction evaluation runs on every expected best checkpoint:
  `3 datasets x 3 models x (7 regular losses + 1 emd) = 72 array tasks`

## Useful Overrides

You can override defaults at submission time. Example:

```bash
sbatch --export=ALL,LOSS_TIME_WEIGHT=2.0,EPOCHS=75 slurm/autoencoder_convergence_suite.sbatch
```

Common overrides:

- `WORKDIR`
- `NFSHOME`
- `IMG`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `LOSS_TIME_WEIGHT`
- `EPOCHS`
- `BATCH_SIZE`
- `TEST_BATCH_SIZE`
- `NUM_POINTS`
- `REGULAR_MAX_BATCHES`
- `EMD_MAX_BATCHES`
