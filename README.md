# Laboratory: Point-Cloud Reconstruction for Event-Camera Data

This repository studies point-cloud reconstruction for event-camera data. Raw events are converted into fixed-size point clouds and reconstructed with PointNet-based autoencoders. The project compares several reconstruction losses and evaluates robustness to noise, temporal shuffling, and point removal.

The supported datasets are DVS Gesture, N-MNIST, and N-Caltech101.

## Requirements

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible version of PyTorch when GPU training is required.

## Dataset setup

The datasets are downloaded automatically through `tonic` the first time they are used. By default, they are stored in `./data`, which is ignored by Git.

| Dataset | CLI value | Notes |
|---|---|---|
| DVS Gesture | `dvsgesture` | Gesture events with an official train/test split. |
| N-MNIST | `nmnist` | Event-based handwritten digits with an official train/test split. |
| N-Caltech101 | `ncaltech101` | Event-based object recognition; this project uses a deterministic 80/20 split. |

Use a different dataset directory with `--save-to PATH`.

# Usage

Run all commands from the repository root.

## Training

Train a PointNet autoencoder with the default point-cloud representation:

```bash
python -m src.train_ae \
  --dataset dvsgesture \
  --model-name pointnet_ae \
  --loss-name chamfer \
  --device cuda \
  --output-dir outputs/autoencoder/dvsgesture_pointnet_chamfer
```

### Main training options

| Parameter | Default | Description |
|---|---:|---|
| `--dataset` | `dvsgesture` | Dataset: `dvsgesture`, `nmnist`, or `ncaltech101`. |
| `--model-name` | `pointnet_ae` | Model: `pointnet_ae`, `pointnet_vae`, or `pointnetpp_ae`. |
| `--loss-name` | `chamfer` | Reconstruction loss used for training. |
| `--input-dim` | `4` | Point format: `3` for `[x, y, t]`, `4` for `[x, y, t, p]`. |
| `--num-points` | `1024` | Number of points in each input cloud. |
| `--epochs` | `50` | Number of training epochs. |
| `--batch-size` | `16` | Training batch size. |
| `--device` | automatic | Device used for training, for example `cpu` or `cuda`. |
| `--stream-mode` | `sample` | Use one cloud per sample or event windows with `windowed`. |
| `--output-dir` | `outputs/autoencoder` | Directory for histories and checkpoints. |

The final experiments use non-overlapping windows of 4096 events, three-dimensional `[x, y, t]` points, and preserved temporal order:

```bash
python -m src.train_ae \
  --dataset dvsgesture \
  --model-name pointnet_ae \
  --loss-name chamfer \
  --stream-mode windowed \
  --window-size 4096 \
  --window-stride 4096 \
  --num-points 4096 \
  --input-dim 3 \
  --no-shuffle-points \
  --device cuda
```

## Evaluate a trained model

Evaluate a checkpoint on clean and corrupted inputs:

```bash
python -m experiments.reconstruction_corruption_eval \
  --checkpoint outputs/autoencoder/dvsgesture_pointnet_chamfer/dvsgesture_pointnet_ae_chamfer_best.pth \
  --split test \
  --metrics chamfer temporal_weighted_chamfer hausdorff mse \
  --device cuda \
  --output outputs/eval/dvsgesture_pointnet_chamfer.csv \
  --plot-dir outputs/eval/dvsgesture_pointnet_chamfer
```

The evaluation applies Gaussian noise, temporal shuffling, and random point removal. It writes a CSV summary and reconstruction plots.

## Compare reconstruction losses

Run a small benchmark on one dataset:

```bash
python -m experiments.loss_comparison \
  --dataset dvsgesture \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --num-points 1024 \
  --batch-size 8 \
  --max-batches 10 \
  --device cuda \
  --output outputs/benchmarks/dvsgesture_loss_comparison.csv
```

## Models

| Model | Description |
|---|---|
| `pointnet_ae` | PointNet encoder with an MLP decoder. |
| `pointnet_vae` | Variational PointNet autoencoder. |
| `pointnetpp_ae` | PointNet++ encoder with an MLP decoder. |

## Losses

The project includes Chamfer, density-aware Chamfer, temporal-weighted Chamfer, Sinkhorn, Hausdorff, Hungarian EMD, projection, and voxel losses. An optional CUDA EMD implementation is also supported when its external extension is installed.

## Weights & Biases

Experiment tracking is disabled by default. Enable it with:

```bash
--wandb online --wandb-project geometric-learning-project
```

## Results and report

Small, presentation-ready artifacts are versioned in [`outputs/final`](outputs/final/README.md). This directory contains selected plots, qualitative examples, CSV tables, Markdown tables, and a concise summary.

Raw experiment runs, downloaded datasets, checkpoints, logs, and intermediate outputs are ignored by Git.

The written project report belongs in [`docs`](docs/). Before submission, add the final document as `docs/report.pdf`.

## Quick checks

```bash
python -m src.test_model
python -m src.test_losses
python -m src.test_flops
```

## Repository structure

```text
src/            Models, datasets, losses, training, and evaluation code
experiments/    Benchmark, robustness, ablation, and analysis scripts
outputs/final/  Curated figures, tables, and summaries tracked by Git
docs/           Final project report
```
