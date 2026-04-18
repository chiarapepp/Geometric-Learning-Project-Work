import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.dataset_factory import build_pointcloud_transform, get_dataset
from src.flops import estimate_loss_flops
from src.losses.loss_factory import get_loss
from src.utils import cuda_peak_memory, ensure_dir, peak_memory_mb, time_call


DEFAULT_LOSSES = [
    "chamfer",
    "density_aware_chamfer",
    "sinkhorn",
    "temporal_weighted_chamfer",
    "hausdorff",
]

ALL_LOSSES = DEFAULT_LOSSES + ["emd", "projection", "voxel"]


def make_loader(
    dataset_name,
    save_to,
    split,
    num_points,
    input_dim,
    batch_size,
    num_workers,
    temporal_weight=1.0,
    sample_mode="random",
    pad_mode="repeat",
    shuffle_points=True,
    split_ratio=0.8,
    split_seed=13,
):
    train = split == "train"
    transform = build_pointcloud_transform(
        dataset_name,
        num_points=num_points,
        input_dim=input_dim,
        temporal_weight=temporal_weight,
        sample_mode=sample_mode,
        pad_mode=pad_mode,
        shuffle=shuffle_points,
    )
    dataset = get_dataset(
        dataset_name=dataset_name,
        save_to=save_to,
        train=train,
        transform=transform,
        split_ratio=split_ratio,
        split_seed=split_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def unpack_points(batch):
    if isinstance(batch, (list, tuple)):
        points = batch[0]
    else:
        points = batch
    if points.ndim == 4:
        points = points.squeeze(1)
    return points.float()


def perturb_points(points, noise_std=0.0, temporal_shuffle_fraction=0.0, drop_fraction=0.0):
    perturbed = points.clone()
    if noise_std > 0:
        noise = torch.randn_like(perturbed)
        if perturbed.shape[-1] >= 4:
            noise[..., 3] = 0.0
        perturbed = perturbed + noise_std * noise
    if temporal_shuffle_fraction > 0:
        batch_size, num_points, _ = perturbed.shape
        count = max(1, int(round(num_points * temporal_shuffle_fraction)))
        for b in range(batch_size):
            idx = torch.randperm(num_points, device=perturbed.device)[:count]
            shuffled = idx[torch.randperm(count, device=perturbed.device)]
            perturbed[b, idx, 2] = perturbed[b, shuffled, 2]
    if drop_fraction > 0:
        keep_probability = 1.0 - drop_fraction
        mask = torch.rand(perturbed.shape[:2], device=perturbed.device) < keep_probability
        for b in range(perturbed.shape[0]):
            kept = perturbed[b][mask[b]]
            if len(kept) == 0:
                kept = perturbed[b, :1]
            if len(kept) >= perturbed.shape[1]:
                perturbed[b] = kept[: perturbed.shape[1]]
            else:
                extra = kept[torch.randint(0, len(kept), (perturbed.shape[1] - len(kept),), device=perturbed.device)]
                perturbed[b] = torch.cat([kept, extra], dim=0)
    if perturbed.shape[-1] >= 2:
        perturbed[..., 0] = perturbed[..., 0].clamp(0.0, 1.0)
        perturbed[..., 1] = perturbed[..., 1].clamp(0.0, 1.0)
    if perturbed.shape[-1] >= 3:
        time_max = float(points[..., 2].max().detach().cpu().item())
        perturbed[..., 2] = perturbed[..., 2].clamp(0.0, max(time_max, 1.0))
    return perturbed


def benchmark_losses(
    loader,
    losses,
    device,
    dataset_name,
    split,
    max_batches=5,
    repeats=3,
    noise_std=0.0,
    temporal_shuffle_fraction=0.0,
    drop_fraction=0.0,
    loss_kwargs=None,
):
    rows = []
    loss_kwargs = loss_kwargs or {}
    loss_functions = {
        name: get_loss(name, **loss_kwargs.get(name, {}))
        for name in losses
    }
    corruption_bits = []
    if noise_std > 0:
        corruption_bits.append(f"noise={noise_std}")
    if temporal_shuffle_fraction > 0:
        corruption_bits.append(f"shuffle={temporal_shuffle_fraction}")
    if drop_fraction > 0:
        corruption_bits.append(f"drop={drop_fraction}")
    corruption_desc = ", ".join(corruption_bits) if corruption_bits else "clean"
    progress_desc = (
        f"Benchmark batches [{dataset_name}/{split}, {corruption_desc}, "
        f"max={max_batches}]"
    )

    progress = tqdm(loader, total=min(max_batches, len(loader)), desc=progress_desc)
    for batch_idx, batch in enumerate(progress):
        if batch_idx >= max_batches:
            break
        target = unpack_points(batch).to(device)
        prediction = perturb_points(
            target,
            noise_std=noise_std,
            temporal_shuffle_fraction=temporal_shuffle_fraction,
            drop_fraction=drop_fraction,
        )

        for loss_name, loss_fn in loss_functions.items():
            progress.set_postfix(loss=loss_name)
            flop_estimate = estimate_loss_flops(
                loss_name,
                prediction,
                target,
                loss_kwargs=loss_kwargs.get(loss_name, {}),
            )

            def run_loss():
                return loss_fn(prediction, target)

            with cuda_peak_memory(device):
                loss_value, seconds = time_call(run_loss, warmup=1, repeats=repeats, device=device)
                memory = peak_memory_mb(device)
            flops_per_second = (
                float(flop_estimate.flops) / seconds
                if seconds > 0 and flop_estimate.flops > 0
                else 0.0
            )
            rows.append(
                {
                    "batch": batch_idx,
                    "dataset": dataset_name,
                    "split": split,
                    "loss": loss_name,
                    "value": float(loss_value.detach().cpu().item()),
                    "seconds": seconds,
                    "peak_memory_mb": memory,
                    "estimated_flops": flop_estimate.flops,
                    "estimated_flops_per_sample": flop_estimate.flops_per_sample,
                    "estimated_flops_per_second": flops_per_second,
                    "flops_method": flop_estimate.method,
                    "batch_size": int(target.shape[0]),
                    "num_points": int(target.shape[1]),
                    "point_dim": int(target.shape[2]),
                    "noise_std": noise_std,
                    "temporal_shuffle_fraction": temporal_shuffle_fraction,
                    "drop_fraction": drop_fraction,
                }
            )
    return rows


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
