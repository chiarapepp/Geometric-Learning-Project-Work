import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.wandb_util import WandbHandler


def read_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return 0.0
    return sum(values) / len(values)


def plot_loss_comparison(rows, output_dir):
    values = defaultdict(list)
    times = defaultdict(list)
    flops = defaultdict(list)
    throughput = defaultdict(list)
    for row in rows:
        values[row["loss"]].append(row["value"])
        times[row["loss"]].append(row["seconds"])
        if "estimated_flops" in row:
            flops[row["loss"]].append(row["estimated_flops"])
        if "estimated_flops_per_second" in row:
            throughput[row["loss"]].append(row["estimated_flops_per_second"])

    losses = sorted(values)
    include_flops = bool(flops)
    if include_flops:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(losses, [mean(values[loss]) for loss in losses])
    axes[0].set_title("Mean loss value")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(losses, [mean(times[loss]) for loss in losses])
    axes[1].set_title("Mean execution time (s)")
    axes[1].tick_params(axis="x", rotation=35)
    if include_flops:
        axes[2].bar(losses, [mean(flops[loss]) / 1e9 for loss in losses])
        axes[2].set_title("Estimated operations (GFLOPs)")
        axes[2].tick_params(axis="x", rotation=35)
        axes[3].bar(losses, [mean(throughput[loss]) / 1e9 for loss in losses])
        axes[3].set_title("Estimated throughput (GFLOP/s)")
        axes[3].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_comparison.png", dpi=200)
    plt.close(fig)


def plot_corruption(rows, output_dir, field, filename):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["loss"]][float(row[field])].append(row["value"])

    fig, ax = plt.subplots(figsize=(7, 4))
    for loss_name, by_level in sorted(grouped.items()):
        levels = sorted(by_level)
        ax.plot(levels, [mean(by_level[level]) for level in levels], marker="o", label=loss_name)
    ax.set_xlabel(field)
    ax.set_ylabel("Mean loss value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_convergence(rows, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["loss_name"]].append(row)

    fig, ax = plt.subplots(figsize=(7, 4))
    for loss_name, loss_rows in sorted(grouped.items()):
        loss_rows = sorted(loss_rows, key=lambda row: int(row["epoch"]))
        ax.plot(
            [int(row["epoch"]) for row in loss_rows],
            [float(row["val_loss"]) for row in loss_rows],
            marker="o",
            label=loss_name,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "autoencoder_convergence.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create simple presentation plots from experiment CSV files.")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["loss_comparison", "noise", "temporal_shuffle", "convergence"],
    )
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--wandb", default="disabled", choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default="geometric-learning-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="plots")
    args = parser.parse_args()
    if args.wandb_run_name is None:
        args.wandb_run_name = f"plot_{args.kind}"

    logger = WandbHandler(
        vars(args),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="plot_results",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)

    if args.kind == "loss_comparison":
        plot_loss_comparison(rows, output_dir)
    elif args.kind == "noise":
        plot_corruption(rows, output_dir, "noise_std", "noise_robustness.png")
    elif args.kind == "temporal_shuffle":
        plot_corruption(rows, output_dir, "temporal_shuffle_fraction", "temporal_shuffle.png")
    else:
        plot_convergence(rows, output_dir)

    for plot_path in tqdm(sorted(output_dir.glob("*.png")), desc="Plot artifacts"):
        logger.log_artifact(plot_path, artifact_type="plot")
    logger.finish()
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
