# Geometric-Learning-Project-Work

Research scaffold for the geometric learning project work on point-cloud reconstruction losses for neuromorphic data.

## What is implemented

- Point-cloud conversion utilities for neuromorphic event streams.
- Support for `DVSGesture`, `NMNIST`, and `SHD` through `tonic` when available.
- A synthetic fallback dataset so the whole pipeline is runnable without external downloads.
- Reconstruction losses:
  - Chamfer Distance
  - Density-Aware Chamfer Distance
  - Sinkhorn approximation of Earth Mover Distance
- Controlled perturbations for benchmarking:
  - Gaussian noise
  - Temporal shuffle
- A PointNet-style autoencoder for convergence comparisons across losses.
- A PointNet++-style autoencoder for hierarchical local-neighborhood encoding.
- CLI scripts for benchmarking and training.

## Project structure

- `src/data.py`: datasets, event-to-point-cloud conversion, corruptions
- `src/losses.py`: reconstruction losses
- `src/model.py`: PointNet autoencoder
- `src/experiments.py`: benchmark and training workflows
- `scripts/benchmark_losses.py`: benchmark losses on a dataset
- `scripts/train_autoencoder.py`: train the autoencoder
- `scripts/compare_autoencoder_losses.py`: compare convergence for several losses
- `scripts/compare_models.py`: compare PointNet and PointNet++ across losses
- `scripts/visualize_reconstruction.py`: save a target-vs-reconstruction plot
- `scripts/smoke_test.py`: quick end-to-end sanity check

## Quick start

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run the synthetic smoke test:

```bash
python scripts/smoke_test.py
```

Benchmark the losses:

```bash
python scripts/benchmark_losses.py --dataset synthetic --split test
```

Train the autoencoder:

```bash
python scripts/train_autoencoder.py --dataset synthetic --loss chamfer --epochs 10
```

Train the PointNet++ autoencoder:

```bash
python scripts/train_autoencoder.py --dataset synthetic --model pointnet++ --loss chamfer --epochs 10
```

Compare convergence across multiple losses:

```bash
python scripts/compare_autoencoder_losses.py --dataset synthetic --losses chamfer density_aware_chamfer sinkhorn_emd
```

Compare PointNet and PointNet++:

```bash
python scripts/compare_models.py --dataset synthetic --models pointnet pointnet++ --losses chamfer density_aware_chamfer
```

Visualize a reconstruction from a trained checkpoint:

```bash
python scripts/visualize_reconstruction.py --checkpoint outputs/autoencoder/synthetic_pointnet_chamfer_train.pt --model pointnet
```

## Tonic datasets

For the course assignment, you will likely want `DVSGesture-128` plus one or two additional neuromorphic datasets. This code expects the `tonic` package for those datasets:

```bash
python -m pip install tonic
python scripts/benchmark_losses.py --dataset dvsgesture --download
python scripts/train_autoencoder.py --dataset dvsgesture --loss density_aware_chamfer --download
```

Supported dataset names:

- `dvsgesture`
- `nmnist`
- `shd`

## Outputs

Generated artifacts are stored in:

- `outputs/benchmarks`: CSV and JSON benchmark summaries
- `outputs/autoencoder`: training curves and model checkpoints

## Notes

- The Sinkhorn EMD implementation is a differentiable approximation and avoids external CUDA extensions.
- The synthetic dataset exists so that development, debugging, and smoke testing do not depend on network access.
- If you want, the next natural step is to add plotting/report notebooks for the final 10-15 minute presentation.
