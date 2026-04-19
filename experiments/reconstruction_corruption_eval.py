import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.evaluate import make_loader, perturb_points, unpack_points, write_csv
from src.losses.loss_factory import get_loss
from src.train_ae import Config, build_model
from src.utils import cuda_peak_memory, peak_memory_mb, set_seed
from src.wandb_util import WandbHandler


DEFAULT_METRICS = ["chamfer", "temporal_weighted_chamfer", "hausdorff", "mse"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained reconstruction model under controlled point-cloud corruptions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--temporal-shuffle-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/reconstruction_corruption_eval.csv")
    parser.add_argument("--plot-dir", default="outputs/eval/plots")
    parser.add_argument("--log-media", action="store_true")
    parser.add_argument("--wandb", default="disabled", choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default="geometric-learning-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="reconstruction_corruption_eval")
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def config_from_checkpoint(checkpoint, args):
    checkpoint_config = checkpoint.get("config", {})
    values = {field.name: getattr(Config, field.name) for field in fields(Config)}
    values.update({key: value for key, value in checkpoint_config.items() if key in values})

    if args.dataset is not None:
        values["dataset"] = args.dataset
    if args.save_to is not None:
        values["save_to"] = args.save_to
    values["device"] = args.device
    if args.batch_size is not None:
        values["test_batch_size"] = args.batch_size
    if args.num_workers is not None:
        values["num_workers"] = args.num_workers
    values["wandb"] = args.wandb
    values["wandb_project"] = args.wandb_project
    values["wandb_entity"] = args.wandb_entity
    values["wandb_run_name"] = args.wandb_run_name
    values["wandb_group"] = args.wandb_group
    values["wandb_job_type"] = "reconstruction_corruption_eval"
    values["wandb_tags"] = args.wandb_tags
    values["seed"] = args.seed
    return Config(**values)


def build_metric_functions(metric_names, loss_time_weight):
    metric_functions = {}
    for metric_name in metric_names:
        kwargs = {}
        if metric_name == "temporal_weighted_chamfer":
            kwargs["time_weight"] = loss_time_weight
        metric_functions[metric_name] = get_loss(metric_name, **kwargs)
    return metric_functions


def corruption_specs(args):
    specs = []
    for noise_std in args.noise_stds:
        specs.append(
            {
                "corruption": "gaussian_noise",
                "corruption_level": noise_std,
                "noise_std": noise_std,
                "temporal_shuffle_fraction": 0.0,
                "drop_fraction": 0.0,
            }
        )
    for fraction in args.temporal_shuffle_fractions:
        specs.append(
            {
                "corruption": "temporal_shuffle",
                "corruption_level": fraction,
                "noise_std": 0.0,
                "temporal_shuffle_fraction": fraction,
                "drop_fraction": 0.0,
            }
        )
    for fraction in args.drop_fractions:
        specs.append(
            {
                "corruption": "random_drop",
                "corruption_level": fraction,
                "noise_std": 0.0,
                "temporal_shuffle_fraction": 0.0,
                "drop_fraction": fraction,
            }
        )
    return specs


def reconstruct(model, model_type, points):
    if model_type == "vae":
        mu, _ = model.encode(points)
        return model.decode(mu)
    return model(points)


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["corruption"], row["corruption_level"], row["metric"])
        grouped[key].append(row)

    aggregate = []
    for (corruption, level, metric), group_rows in grouped.items():
        reconstruction_values = [float(row["reconstruction_value"]) for row in group_rows]
        corrupted_input_values = [float(row["corrupted_input_value"]) for row in group_rows]
        aggregate.append(
            {
                "corruption": corruption,
                "corruption_level": float(level),
                "metric": metric,
                "mean_reconstruction_value": sum(reconstruction_values) / len(reconstruction_values),
                "mean_corrupted_input_value": sum(corrupted_input_values) / len(corrupted_input_values),
            }
        )
    return aggregate


def write_plots(rows, plot_dir, logger):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    by_metric_corruption = defaultdict(list)
    for row in aggregate:
        by_metric_corruption[(row["metric"], row["corruption"])].append(row)

    for (metric, corruption), plot_rows in by_metric_corruption.items():
        plot_rows = sorted(plot_rows, key=lambda row: row["corruption_level"])
        levels = [row["corruption_level"] for row in plot_rows]
        reconstruction = [row["mean_reconstruction_value"] for row in plot_rows]
        corrupted_input = [row["mean_corrupted_input_value"] for row in plot_rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(levels, reconstruction, marker="o", label="model reconstruction")
        ax.plot(levels, corrupted_input, marker="o", label="corrupted input")
        ax.set_xlabel(corruption)
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()

        plot_path = plot_dir / f"{metric}_{corruption}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logger.log_artifact(plot_path, artifact_type="eval-plot")
        logger.log_image(str(plot_path), f"eval/plots/{metric}_{corruption}")


def main():
    args = parse_args()
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = config_from_checkpoint(checkpoint, args)
    if cfg.wandb_run_name is None:
        checkpoint_stem = Path(args.checkpoint).stem
        cfg.wandb_run_name = f"eval_{cfg.dataset}_{cfg.model_name}_{checkpoint_stem}"

    logger = WandbHandler(cfg)
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
        split_ratio=cfg.split_ratio,
        split_seed=cfg.split_seed,
    )
    metric_functions = build_metric_functions(args.metrics, cfg.loss_time_weight)

    rows = []
    specs = corruption_specs(args)
    with torch.no_grad():
        for spec_idx, spec in enumerate(tqdm(specs, desc="Corruptions")):
            progress = tqdm(loader, total=min(args.max_batches, len(loader)), desc=spec["corruption"])
            for batch_idx, batch in enumerate(progress):
                if batch_idx >= args.max_batches:
                    break

                target = unpack_points(batch).to(cfg.device)
                corrupted = perturb_points(
                    target,
                    noise_std=spec["noise_std"],
                    temporal_shuffle_fraction=spec["temporal_shuffle_fraction"],
                    drop_fraction=spec["drop_fraction"],
                )

                with cuda_peak_memory(cfg.device):
                    if torch.device(cfg.device).type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(cfg.device)
                    start = time.perf_counter()
                    reconstruction = reconstruct(model, model_type, corrupted)
                    if torch.device(cfg.device).type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(cfg.device)
                    model_seconds = time.perf_counter() - start
                    memory = peak_memory_mb(cfg.device)

                for metric_name, metric_fn in metric_functions.items():
                    reconstruction_value = metric_fn(reconstruction, target)
                    corrupted_input_value = metric_fn(corrupted, target)
                    rows.append(
                        {
                            "batch": batch_idx,
                            "dataset": cfg.dataset,
                            "split": args.split,
                            "model": cfg.model_name,
                            "trained_loss": cfg.loss_name,
                            "checkpoint": args.checkpoint,
                            "corruption": spec["corruption"],
                            "corruption_level": spec["corruption_level"],
                            "noise_std": spec["noise_std"],
                            "temporal_shuffle_fraction": spec["temporal_shuffle_fraction"],
                            "drop_fraction": spec["drop_fraction"],
                            "metric": metric_name,
                            "reconstruction_value": float(reconstruction_value.detach().cpu().item()),
                            "corrupted_input_value": float(corrupted_input_value.detach().cpu().item()),
                            "model_seconds": model_seconds,
                            "peak_memory_mb": memory,
                            "batch_size": int(target.shape[0]),
                            "num_points": int(target.shape[1]),
                            "point_dim": int(target.shape[2]),
                        }
                    )

                if args.log_media and batch_idx == 0:
                    prefix = f"eval/{spec['corruption']}/level_{spec['corruption_level']}"
                    logger.log_point_cloud(target[0], f"{prefix}/target", spec_idx, "Clean target")
                    logger.log_point_cloud(corrupted[0], f"{prefix}/corrupted_input", spec_idx, "Corrupted input")
                    logger.log_point_cloud(reconstruction[0], f"{prefix}/reconstruction", spec_idx, "Model reconstruction")

    write_csv(Path(args.output), rows)
    logger.log_reconstruction_eval_results(rows, csv_path=args.output)
    write_plots(rows, args.plot_dir, logger)
    logger.finish()
    print(f"Wrote {len(rows)} evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
