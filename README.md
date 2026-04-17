# Geometric Learning Project Work

Project work for testing point-cloud reconstruction losses on neuromorphic camera datasets.

The goal is to compare losses commonly used for 3D point-cloud reconstruction and similarity measurement in a noisier event-camera setting, especially DVSGesture-128, plus NMNIST and NCaltech101.

## Implemented Components

- Dataset wrappers for:
  - `dvsgesture`
  - `nmnist`
  - `ncaltech101`
- Point-cloud preprocessing:
  - raw tonic events to `[x, y, t, p]`
  - optional `[x, y, t]`
  - x/y/t normalization
  - per-sample x/y normalization for NCaltech101
  - fixed-size sampling/padding
- Losses:
  - Chamfer
  - Density-Aware Chamfer
  - Hungarian EMD
  - Sinkhorn
  - Temporal-weighted Chamfer
  - Hausdorff
  - Projection loss
  - Voxel loss
- Models:
  - PointNet AE
  - PointNet VAE
  - PointNet++ AE
- Experiment scripts:
  - loss comparison
  - noise robustness
  - temporal shuffle sensitivity
  - autoencoder convergence across losses
  - simple CSV-to-PNG plotting

## Install

Install PyTorch according to the cluster CUDA setup, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root.

## Weights & Biases

All training and benchmark scripts run with W&B disabled by default. Enable it with:

```bash
--wandb online --wandb-project geometric-learning-project
```

Optional fields:

```bash
--wandb-entity YOUR_ENTITY
--wandb-run-name custom_run_name
--wandb-group custom_group_name
```

Benchmark scripts log aggregate metrics, full result tables, and CSV artifacts. Training logs epoch metrics, history CSV artifacts, and the best checkpoint artifact.

## Dataset Point Clouds

The common transform is created by:

```python
build_pointcloud_transform(
    dataset_name="dvsgesture",
    num_points=1024,
    input_dim=4,
)
```

Use `input_dim=4` for `[x, y, t, p]` and `input_dim=3` for `[x, y, t]`.

NCaltech101 has no official train/test split in tonic, so this project uses a deterministic 80/20 split with `split_seed=13`.

## Loss Comparison

```bash
python -m experiments.loss_comparison \
  --dataset dvsgesture \
  --save-to ./data \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --num-points 1024 \
  --batch-size 8 \
  --max-batches 20 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output outputs/benchmarks/dvsgesture_loss_comparison.csv
```

Hungarian EMD is available as `emd`, but use fewer points:

```bash
python -m experiments.loss_comparison \
  --dataset dvsgesture \
  --losses emd chamfer \
  --num-points 128 \
  --batch-size 4 \
  --max-batches 10 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output outputs/benchmarks/dvsgesture_emd_small.csv
```

## Robustness Experiments

Gaussian noise:

```bash
python -m experiments.noise_robustness \
  --dataset dvsgesture \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --noise-stds 0.0 0.01 0.03 0.05 0.1 \
  --num-points 1024 \
  --batch-size 8 \
  --max-batches 20 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output outputs/benchmarks/dvsgesture_noise.csv
```

Temporal shuffle:

```bash
python -m experiments.temporal_shuffle \
  --dataset dvsgesture \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --fractions 0.0 0.1 0.25 0.5 1.0 \
  --num-points 1024 \
  --batch-size 8 \
  --max-batches 20 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output outputs/benchmarks/dvsgesture_temporal_shuffle.csv
```

Repeat the same commands with `--dataset nmnist` and `--dataset ncaltech101`.

## Autoencoder Training

Single run:

```bash
python -m src.train_ae \
  --dataset dvsgesture \
  --model-name pointnet_ae \
  --loss-name chamfer \
  --num-points 1024 \
  --input-dim 4 \
  --epochs 50 \
  --batch-size 16 \
  --test-batch-size 16 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/autoencoder/dvsgesture_pointnet_chamfer
```

Convergence comparison across losses:

```bash
python -m experiments.autoencoder_convergence \
  --dataset dvsgesture \
  --model-name pointnet_ae \
  --losses chamfer density_aware_chamfer sinkhorn temporal_weighted_chamfer hausdorff \
  --num-points 1024 \
  --epochs 50 \
  --batch-size 16 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/autoencoder_convergence
```

This writes one history CSV per loss and an aggregate convergence CSV.

## Plotting

```bash
python -m experiments.plot_results \
  --input outputs/benchmarks/dvsgesture_loss_comparison.csv \
  --kind loss_comparison \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/plots

python -m experiments.plot_results \
  --input outputs/benchmarks/dvsgesture_noise.csv \
  --kind noise \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/plots

python -m experiments.plot_results \
  --input outputs/benchmarks/dvsgesture_temporal_shuffle.csv \
  --kind temporal_shuffle \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/plots

python -m experiments.plot_results \
  --input outputs/autoencoder_convergence/dvsgesture_pointnet_ae_convergence.csv \
  --kind convergence \
  --wandb online \
  --wandb-project geometric-learning-project \
  --output-dir outputs/plots
```

## Suggested Final Results

For the presentation, collect:

- runtime and memory table by loss and dataset
- noise robustness curves
- temporal shuffle robustness curves
- PointNet AE convergence curves across losses
- optional PointNet VAE or PointNet++ comparison if cluster time allows

Keep EMD small-point because Hungarian matching is expensive.
