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
- FLOP estimates:
  - analytic per-loss forward-pass operation count
  - estimated FLOP/s throughput from measured wall-clock time

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

The main project workflow should use W&B for:

- training curves: `train/loss`, `val/loss`, optional VAE `recon_loss` and `kl_loss`
- model tracking: best checkpoint artifact and optional periodic checkpoint artifacts
- trained-model corruption evaluation: reconstruction metrics under Gaussian noise, temporal shuffle, and random point drop
- media: optional 3D point-cloud previews for target, corrupted input, and model reconstruction

This is a reconstruction task, so there is no classification accuracy unless a classifier head is added. Use reconstruction metrics such as Chamfer, temporal-weighted Chamfer, Hausdorff, and MSE instead.

## FLOP Estimates

Benchmark CSV files include:

- `estimated_flops`: estimated floating-point operations for one forward loss evaluation on the batch
- `estimated_flops_per_sample`: estimated operations divided by batch size
- `estimated_flops_per_second`: estimated throughput from `estimated_flops / seconds`
- `flops_method`: analytic estimate used for the specific loss

These are implementation-level estimates intended for relative comparison between losses, point counts, and datasets. They are not exact hardware counters. For Sinkhorn, the estimate uses a default rough budget of 50 iterations because `geomloss` chooses the internal schedule dynamically. Override that reporting assumption with:

```bash
--sinkhorn-iterations-estimate 100
```

Quick local check:

```bash
python -m src.test_flops
```

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

### Event-Stream Windowing

By default, each tonic sample is converted to one normalized fixed-size point cloud. To follow the reference EventCloudReconstruction-style stream loading more closely, enable event-count windows:

```bash
python -m experiments.loss_comparison \
  --dataset dvsgesture \
  --stream-mode windowed \
  --window-size 5000 \
  --window-stride 2500 \
  --num-points 1024 \
  --max-windows-per-sample 20
```

This slices each raw event stream into windows before point-cloud normalization and sampling. The model still receives fixed-size tensors, but each tensor represents a local temporal segment rather than the whole sample.

For the closest match to the reference loading configuration (`slice_size=4096`, `stride=-1`, `use_polarity=False`), keep every event in the window and preserve point order:

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
  --no-shuffle-points
```

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

An optional CUDA approximation from `meder411/PyTorch-EMDLoss` is exposed as
`emd_cuda`. Install that external CUDA extension in the active environment first,
then run the synthetic smoke benchmark:

```bash
python -m experiments.emd_cuda_benchmark \
  --losses emd_cuda emd sinkhorn \
  --num-points 128 \
  --batch-size 2 \
  --device cuda \
  --output outputs/benchmarks/emd_cuda_synthetic.csv
```

You can also include it in the dataset benchmark:

```bash
python -m experiments.loss_comparison \
  --dataset dvsgesture \
  --losses emd_cuda emd sinkhorn \
  --num-points 128 \
  --batch-size 4 \
  --max-batches 10 \
  --device cuda \
  --output outputs/benchmarks/dvsgesture_emd_cuda_small.csv
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
  --save-every 10 \
  --output-dir outputs/autoencoder/dvsgesture_pointnet_chamfer
```

Resume a long run from a checkpoint:

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
  --resume-from outputs/autoencoder/dvsgesture_pointnet_chamfer/dvsgesture_pointnet_ae_chamfer_epoch_10.pth \
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
  --save-every 10 \
  --output-dir outputs/autoencoder_convergence
```

This writes one history CSV per loss and an aggregate convergence CSV.

## Trained Reconstruction Evaluation

After training, evaluate a saved model under controlled corruptions. This measures whether the trained autoencoder can reconstruct the clean target when its input is corrupted.

```bash
python -m experiments.reconstruction_corruption_eval \
  --checkpoint outputs/autoencoder_convergence/dvsgesture_pointnet_ae_chamfer/dvsgesture_pointnet_ae_chamfer_best.pth \
  --split test \
  --metrics chamfer temporal_weighted_chamfer hausdorff mse \
  --noise-stds 0.0 0.01 0.03 0.05 0.1 \
  --temporal-shuffle-fractions 0.0 0.1 0.25 0.5 1.0 \
  --drop-fractions 0.0 0.1 0.25 0.5 \
  --max-batches 20 \
  --device cuda \
  --wandb online \
  --wandb-project geometric-learning-project \
  --log-media \
  --output outputs/eval/dvsgesture_pointnet_ae_chamfer_corruptions.csv
```

The script logs:

- `eval/.../mean_reconstruction`: metric between model reconstruction and clean target
- `eval/.../mean_corrupted_input`: metric between corrupted input and clean target
- `eval/results`: a full W&B table
- a CSV artifact
- PNG summary plots
- optional 3D point-cloud previews when `--log-media` is set

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
- estimated FLOPs and estimated GFLOP/s table by loss and dataset
- noise robustness curves
- temporal shuffle robustness curves
- PointNet AE convergence curves across losses
- optional PointNet VAE or PointNet++ comparison if cluster time allows

Keep EMD small-point because Hungarian matching is expensive.
