# Laboratory: Point-Cloud Reconstruction for Event-Camera Data

This repository studies point-cloud reconstruction for event-camera data. Raw events are converted into fixed-size point clouds and reconstructed with PointNet-based autoencoders. The project compares several reconstruction losses and evaluates robustness to noise, temporal shuffling, and point removal.

The supported datasets are DVS Gesture, N-MNIST, and N-Caltech101.

## Requirements

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset setup

The datasets are downloaded automatically through `tonic` the first time they are used. By default, they are stored in `./data`.

| Dataset | CLI value | Notes |
|---|---|---|
| DVS Gesture | `dvsgesture` | Official train split divided deterministically into 80% train / 20% validation; official test is held out. |
| N-MNIST | `nmnist` | Official train split divided deterministically into 80% train / 20% validation; official test is held out. |
| N-Caltech101 | `ncaltech101` | Deterministic 80% train / 10% validation / 10% test split. |

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
| `--model-name` | `pointnet_ae` | Model: `pointnet_ae` or `pointnetpp_ae`. |
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

### Representative evaluation subsets

For windowed datasets, do not stop at the first batches: they can contain a
single class or several windows from one recording. First inspect the split
and create a deterministic, class-balanced manifest:

```bash
python -m experiments.inspect_dataset_composition \
  --dataset ncaltech101 --save-to /path/to/data --split test \
  --stream-mode windowed --window-size 4096 --window-stride 4096 \
  --items-per-class 4 --max-windows-per-source 1 \
  --output-dir outputs/composition
```

The generated `*_balanced_subset.csv` records the exact dataset indices,
class labels, original recording identifiers, and window positions. Reuse it
for every checkpoint so loss comparisons remain paired:

```bash
python -m experiments.reconstruction_corruption_eval \
  --checkpoint /path/to/checkpoint.pth --split test --device cuda \
  --eval-subset-from outputs/composition/ncaltech101_test_windowed_balanced_subset.csv \
  --output outputs/eval/ncaltech101_balanced.csv
```

Or create the subset directly during evaluation with
`--eval-sampling stratified --eval-items-per-class 4
--eval-max-windows-per-source 1`. The selected manifest is always saved next
to the evaluation CSV. The same options are available in
`experiments.latent_robustness_eval`.

For qualitative figures, pass the same manifest to
`experiments.visual_reconstruction_eval --subset-manifest ...` and choose
three `dataset_index` values from its CSV with `--sample-indices`.

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

Evaluate the same standalone objectives after removing a controlled fraction
of points (the perturbed cloud is not padded back to its original size):

```bash
python -m experiments.random_drop \
  --dataset dvsgesture \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --fractions 0.0 0.1 0.25 0.5 \
  --num-points 4096 \
  --input-dim 3 \
  --stream-mode windowed \
  --window-size 4096 \
  --window-stride 4096 \
  --no-shuffle-points \
  --batch-size 4 \
  --max-batches 10 \
  --device cuda \
  --output outputs/benchmarks/dvsgesture_random_drop.csv
```

EMD is intentionally excluded because its one-to-one assignment requires the
two clouds to contain the same number of points.

## Models

| Model | Description |
|---|---|
| `pointnet_ae` | PointNet encoder with an MLP decoder. |
| `pointnetpp_ae` | PointNet++ encoder with an MLP decoder. |

## Losses

The project includes Chamfer, density-aware Chamfer, temporal-weighted Chamfer, Sinkhorn and Hausdorff.

## Results and report

Small, artifacts are present in [`outputs/final`](outputs/final/README.md). This directory contains selected plots, qualitative examples, CSV tables, Markdown tables, and a concise summary.

Raw experiment runs, downloaded datasets, checkpoints, logs, and intermediate outputs are ignored by Git.

The written project report belongs in [`docs`](docs/). 

## Repository structure

```text
src/            Models, datasets, losses, training, and evaluation code
experiments/    Benchmark, robustness, ablation, and analysis scripts
outputs/final/  Curated figures, tables, and summaries tracked by Git
docs/           Final project report
```
