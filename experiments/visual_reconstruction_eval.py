import argparse
import csv
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.evaluate import make_loader, perturb_points, unpack_points
from src.losses.loss_factory import get_loss
from src.train_ae import Config, build_model
from src.utils import ensure_dir, set_seed


DEFAULT_DATASETS = ["dvsgesture", "nmnist", "ncaltech101"]
DEFAULT_MODELS = ["pointnet_ae", "pointnet_vae", "pointnetpp_ae"]
DEFAULT_LOSSES = [
    "chamfer",
    "density_aware_chamfer",
    "sinkhorn",
    "temporal_weighted_chamfer",
    "hausdorff",
    "projection",
    "emd",
]
DEFAULT_METRICS = ["chamfer", "temporal_weighted_chamfer", "hausdorff", "mse"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create visual panels for trained point-cloud reconstruction models: "
            "clean target, corrupted input, and model reconstruction."
        )
    )
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument(
        "--checkpoint-roots",
        nargs="+",
        default=["outputs/autoencoder_convergence", "outputs/autoencoder_convergence_emd"],
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--save-to", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--metric-time-weight", type=float, default=None)
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument(
        "--temporal-shuffle-fractions",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
    )
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--full-eval-grid", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/visual_eval")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit-t-max", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def discover_checkpoints(args):
    if args.checkpoints:
        return [Path(path) for path in args.checkpoints]

    checkpoints = []
    for root in [Path(path) for path in args.checkpoint_roots]:
        for dataset in args.datasets:
            for model in args.models:
                for loss_name in args.losses:
                    path = root / f"{dataset}_{model}_{loss_name}" / f"{dataset}_{model}_{loss_name}_best.pth"
                    if path.exists():
                        checkpoints.append(path)
    return checkpoints


def config_from_checkpoint(checkpoint, args):
    checkpoint_config = checkpoint.get("config", {})
    values = {field.name: getattr(Config, field.name) for field in fields(Config)}
    values.update({key: value for key, value in checkpoint_config.items() if key in values})
    values["device"] = args.device
    values["test_batch_size"] = args.batch_size
    values["num_workers"] = args.num_workers
    if args.save_to is not None:
        values["save_to"] = args.save_to
    return Config(**values)


def reconstruct(model, model_type, points):
    if model_type == "vae":
        mu, _ = model.encode(points)
        return model.decode(mu)
    return model(points)


def corruption_specs(args):
    if args.full_eval_grid:
        args.noise_stds = [0.0, 0.01, 0.03, 0.05, 0.1]
        args.temporal_shuffle_fractions = [0.0, 0.1, 0.25, 0.5, 1.0]
        args.drop_fractions = [0.0, 0.1, 0.25, 0.5]

    specs = []
    for noise_std in args.noise_stds:
        specs.append(
            {
                "corruption": "gaussian_noise",
                "level": noise_std,
                "noise_std": noise_std,
                "temporal_shuffle_fraction": 0.0,
                "drop_fraction": 0.0,
            }
        )
    for fraction in args.temporal_shuffle_fractions:
        specs.append(
            {
                "corruption": "temporal_shuffle",
                "level": fraction,
                "noise_std": 0.0,
                "temporal_shuffle_fraction": fraction,
                "drop_fraction": 0.0,
            }
        )
    for fraction in args.drop_fractions:
        specs.append(
            {
                "corruption": "random_drop",
                "level": fraction,
                "noise_std": 0.0,
                "temporal_shuffle_fraction": 0.0,
                "drop_fraction": fraction,
            }
        )
    return specs


def build_metric_functions(metric_names, time_weight):
    metric_functions = {}
    for metric_name in metric_names:
        kwargs = {}
        if metric_name == "temporal_weighted_chamfer" and time_weight is not None:
            kwargs["time_weight"] = time_weight
        metric_functions[metric_name] = get_loss(metric_name, **kwargs)
    return metric_functions


def stable_seed(*parts):
    text = "|".join(str(part) for part in parts)
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) % (2**32)
    return value


def take_sample_indices(loader, sample_indices, device):
    wanted = sorted(set(sample_indices))
    samples = {}
    seen = 0
    for batch in loader:
        points = unpack_points(batch).float()
        batch_size = points.shape[0]
        for local_idx in range(batch_size):
            global_idx = seen + local_idx
            if global_idx in wanted:
                samples[global_idx] = points[local_idx].to(device)
        seen += batch_size
        if len(samples) == len(wanted):
            break

    missing = sorted(set(wanted) - set(samples))
    if missing:
        raise ValueError(f"Requested sample indices not available: {missing}")
    return samples


def to_numpy(points):
    if hasattr(points, "detach"):
        points = points.detach().cpu()
    return points.float().numpy()


def in_range_fraction(points, t_max):
    if points.shape[1] < 3:
        return 0.0
    mask = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= 1.0)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= 1.0)
        & (points[:, 2] >= 0.0)
        & (points[:, 2] <= t_max)
    )
    return float(mask.mean()) if len(mask) else 0.0


def point_ranges(points):
    ranges = []
    labels = ["x", "y", "t", "p"]
    for dim in range(min(points.shape[1], 4)):
        ranges.append(f"{labels[dim]}[{points[:, dim].min():.2f},{points[:, dim].max():.2f}]")
    return " ".join(ranges)


def sample_for_plot(points, max_points, seed):
    if hasattr(points, "detach"):
        points = points.detach().cpu()
    if len(points) <= max_points:
        return points
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(points), generator=generator)[:max_points]
    return points[idx]


def scatter_2d(ax, points, x_dim, y_dim, title, t_max):
    color = points[:, 2] if points.shape[1] >= 3 else None
    ax.scatter(
        points[:, x_dim],
        points[:, y_dim],
        c=color,
        cmap="viridis",
        vmin=0.0,
        vmax=t_max,
        s=3,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, t_max if y_dim == 2 else 1.0)
    ax.grid(True, linewidth=0.25, alpha=0.35)


def scatter_3d(ax, points, title, t_max):
    if points.shape[1] >= 4:
        color = points[:, 3]
        cmap = "coolwarm"
    else:
        color = points[:, 2]
        cmap = "viridis"
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        cmap=cmap,
        s=3,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, t_max)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("t", fontsize=8)
    ax.view_init(elev=22, azim=-58)


def format_metric_summary(metric_values):
    bits = []
    for metric_name, values in metric_values.items():
        rec_value = values["reconstruction"]
        input_value = values["corrupted_input"]
        bits.append(f"{metric_name}: rec={rec_value:.4g}, input={input_value:.4g}")
    return " | ".join(bits)


def plot_visual_panel(
    target,
    corrupted,
    reconstruction,
    metadata,
    metric_values,
    output_path,
    max_plot_points,
    seed,
    t_max,
    dpi,
):
    clouds = [
        ("Clean target", target),
        ("Corrupted input", corrupted),
        ("Reconstruction", reconstruction),
    ]
    clouds_np = [
        (name, to_numpy(sample_for_plot(points, max_plot_points, seed + idx)))
        for idx, (name, points) in enumerate(clouds)
    ]

    fig = plt.figure(figsize=(13.5, 15.0))
    axes = []
    for col, (name, points) in enumerate(clouds_np):
        axes.append(fig.add_subplot(4, 3, col + 1, projection="3d"))
        inside = 100.0 * in_range_fraction(points, t_max)
        scatter_3d(
            axes[-1],
            points,
            f"{name}\nin range={inside:.1f}% | {point_ranges(points)}",
            t_max,
        )

    projection_specs = [
        ("xy projection, color=t", 0, 1, "x", "y"),
        ("xt projection, color=t", 0, 2, "x", "t"),
        ("yt projection, color=t", 1, 2, "y", "t"),
    ]
    for row, (row_title, x_dim, y_dim, x_label, y_label) in enumerate(projection_specs, start=1):
        for col, (name, points) in enumerate(clouds_np):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1)
            scatter_2d(ax, points, x_dim, y_dim, f"{name}\n{row_title}", t_max)
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)

    metric_text = format_metric_summary(metric_values)
    title = (
        f"{metadata['dataset']} / {metadata['model']} / trained loss={metadata['loss']} | "
        f"sample={metadata['sample_idx']} | {metadata['corruption']}={metadata['level']}\n"
        f"{metric_text}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.0, 1, 0.955))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_manifest(path, rows):
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path, rows):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Visual Reconstruction Evaluation\n\n")
        handle.write(
            "Each figure compares clean target, corrupted input, and model reconstruction. "
            "The first row is the 3D x-y-t cloud; the other rows are xy, xt, and yt projections.\n\n"
        )
        grouped = {}
        for row in rows:
            key = (row["dataset"], row["model"], row["loss"])
            grouped.setdefault(key, []).append(row)
        for key in sorted(grouped):
            dataset, model, loss_name = key
            handle.write(f"## {dataset} / {model} / {loss_name}\n\n")
            for row in grouped[key]:
                image_path = Path(row["image"])
                try:
                    rel_path = image_path.relative_to(path.parent).as_posix()
                except ValueError:
                    rel_path = image_path.as_posix()
                handle.write(
                    f"### sample {row['sample_idx']} - {row['corruption']}={row['level']}\n\n"
                )
                handle.write(f"![{rel_path}]({rel_path})\n\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoints = discover_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found. Pass --checkpoints or adjust filters.")

    specs = corruption_specs(args)
    manifest_rows = []

    for checkpoint_path in tqdm(checkpoints, desc="Checkpoints"):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        cfg = config_from_checkpoint(checkpoint, args)
        model, model_type = build_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(cfg.device)
        model.eval()

        loader = make_loader(
            dataset_name=cfg.dataset,
            save_to=cfg.save_to,
            split=args.split,
            num_points=cfg.num_points,
            input_dim=cfg.input_dim,
            batch_size=cfg.test_batch_size,
            num_workers=cfg.num_workers,
            temporal_weight=cfg.temporal_weight,
            sample_mode=cfg.sample_mode,
            pad_mode=cfg.pad_mode,
            shuffle_points=cfg.shuffle_points,
            split_ratio=cfg.split_ratio,
            split_seed=cfg.split_seed,
            stream_mode=cfg.stream_mode,
            window_size=cfg.window_size,
            window_stride=cfg.window_stride,
            window_drop_last=cfg.window_drop_last,
            max_windows_per_sample=cfg.max_windows_per_sample,
        )
        set_seed(args.seed)
        samples = take_sample_indices(loader, args.sample_indices, cfg.device)
        time_weight = args.metric_time_weight
        if time_weight is None:
            time_weight = cfg.loss_time_weight if cfg.loss_time_weight != 1.0 else None
        metric_functions = build_metric_functions(args.metrics, time_weight)
        t_max = max(float(args.limit_t_max), float(cfg.temporal_weight), 1.0)

        with torch.no_grad():
            for sample_idx, target in samples.items():
                target_batch = target.unsqueeze(0)
                for spec in specs:
                    spec_seed = stable_seed(args.seed, cfg.dataset, sample_idx, spec["corruption"], spec["level"])
                    torch.manual_seed(spec_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(spec_seed)
                    corrupted = perturb_points(
                        target_batch,
                        noise_std=spec["noise_std"],
                        temporal_shuffle_fraction=spec["temporal_shuffle_fraction"],
                        drop_fraction=spec["drop_fraction"],
                    )
                    reconstruction = reconstruct(model, model_type, corrupted)

                    metric_values = {}
                    for metric_name, metric_fn in metric_functions.items():
                        metric_values[metric_name] = {
                            "reconstruction": float(metric_fn(reconstruction, target_batch).detach().cpu().item()),
                            "corrupted_input": float(metric_fn(corrupted, target_batch).detach().cpu().item()),
                        }

                    run_name = f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}"
                    stem = (
                        f"sample_{sample_idx:04d}_"
                        f"{spec['corruption']}_{str(spec['level']).replace('.', 'p')}"
                    )
                    image_path = output_dir / run_name / f"{stem}.png"
                    metadata = {
                        "dataset": cfg.dataset,
                        "model": cfg.model_name,
                        "loss": cfg.loss_name,
                        "sample_idx": sample_idx,
                        "corruption": spec["corruption"],
                        "level": spec["level"],
                    }
                    plot_visual_panel(
                        target=target,
                        corrupted=corrupted.squeeze(0),
                        reconstruction=reconstruction.squeeze(0),
                        metadata=metadata,
                        metric_values=metric_values,
                        output_path=image_path,
                        max_plot_points=args.max_plot_points,
                        seed=args.seed + sample_idx,
                        t_max=t_max,
                        dpi=args.dpi,
                    )

                    row = {
                        "dataset": cfg.dataset,
                        "model": cfg.model_name,
                        "loss": cfg.loss_name,
                        "checkpoint": str(checkpoint_path),
                        "sample_idx": sample_idx,
                        "corruption": spec["corruption"],
                        "level": spec["level"],
                        "image": str(image_path),
                        "num_points": cfg.num_points,
                        "input_dim": cfg.input_dim,
                    }
                    for metric_name, values in metric_values.items():
                        row[f"{metric_name}_reconstruction"] = values["reconstruction"]
                        row[f"{metric_name}_corrupted_input"] = values["corrupted_input"]
                    manifest_rows.append(row)

    manifest_path = output_dir / "manifest.csv"
    report_path = output_dir / "visual_report.md"
    write_manifest(manifest_path, manifest_rows)
    write_markdown_report(report_path, manifest_rows)
    print(f"Wrote {len(manifest_rows)} visual panels to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
