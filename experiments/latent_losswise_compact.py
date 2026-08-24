"""Render compact, loss-wise latent retrieval figures for the report.

Each figure keeps one corrupted query fixed and compares the five main
reconstruction objectives.  The first row is the clean target plus the five
reconstructions; the second row is the corresponding exact Top-1 retrieval.
Green frames identify an exact retrieval and blue frames a different one.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from experiments.latent_retrieval_comparison import (
    prepare_run,
    read_runs,
    reconstruct,
    select_cases,
    set_seed,
    write_selection,
)


LOSS_ORDER = ("chamfer", "density_aware_chamfer", "hausdorff", "temporal_weighted_chamfer")
LOSS_NAMES = {
    "chamfer": "Chamfer",
    "density_aware_chamfer": "DCD",
    "hausdorff": "Hausdorff",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["dvsgesture", "nmnist", "ncaltech101"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save-to", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gallery-size", type=int, default=400)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--eval-subset-from", default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--encoder-seed", type=int, default=1729)
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--temporal-shuffle-fractions", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    return parser.parse_args()


def loss_label(row):
    loss = row["trained_loss"]
    if loss == "temporal_weighted_chamfer":
        weight = int(float(row["trained_loss_time_weight"]))
        return f"Temporal W. ({weight})"
    return LOSS_NAMES[loss]


def order_configs(case):
    rows = list(case["configs"].values())
    return sorted(rows, key=lambda row: (LOSS_ORDER.index(row["trained_loss"]),
                                         float(row.get("trained_loss_time_weight") or 0)))


def clean_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def frame(axis, color=None):
    for spine in axis.spines.values():
        spine.set_visible(color is not None)
        if color:
            spine.set_color(color)
            spine.set_linewidth(2.2)


def cloud(axis, points, border=None):
    values = points.detach().cpu().numpy()
    axis.scatter(values[:, 0], values[:, 1], c=values[:, 2], cmap="viridis", vmin=0, vmax=1,
                 s=5.2, linewidths=0, alpha=0.95, rasterized=True)
    clean_axis(axis)
    frame(axis, border)


def dataset_title(name):
    return {"dvsgesture": "DVS Gesture", "nmnist": "N-MNIST", "ncaltech101": "N-Caltech101"}[name]


def model_title(name):
    return {"pointnet_ae": "PointNet", "pointnetpp_ae": "PointNet++"}.get(name, name)


def render_case(case_id, case, args, output_dir, prepared):
    rows = order_configs(case)
    if len(rows) != 5:
        return None
    reconstructions = []
    retrieved = []
    true_points = None
    for row in rows:
        label = row["_label"]
        if label not in prepared:
            prepared[label] = prepare_run(row, args)
        reconstruction, target, gallery = reconstruct(row, prepared[label], args)
        reconstructions.append(reconstruction)
        retrieved.append(gallery[0])
        if true_points is None:
            true_points = target

    first = rows[0]
    fig, axes = plt.subplots(2, 6, figsize=(18.5, 7.8), gridspec_kw={"wspace": 0.06, "hspace": 0.24})
    fig.subplots_adjust(left=0.035, right=0.995, top=0.84, bottom=0.075)
    fig.suptitle("Latent retrieval across reconstruction losses", x=0.035, y=0.975,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(0.035, 0.929,
             f"{dataset_title(first['dataset'])} · {model_title(first['model'])} · same query · "
             f"{first['corruption'].replace('_', ' ')} {float(first['corruption_level']):g}",
             ha="left", fontsize=10.5, color="#555555")
    fig.text(0.035, 0.875, "Clean target", ha="left", fontsize=12.5, color="#238636", fontweight="bold")
    fig.text(0.035, 0.474, "Top-1 retrieval", ha="left", fontsize=12.5, color="#2166c2", fontweight="bold")

    cloud(axes[0, 0], true_points, "#238636")
    for column, (row, reconstruction, top1) in enumerate(zip(rows, reconstructions, retrieved), 1):
        axes[0, column].set_title(loss_label(row), fontsize=11.5, fontweight="bold", pad=12)
        cloud(axes[0, column], reconstruction)
        exact = int(row["exact_top1"]) == 1
        cloud(axes[1, column], top1, "#238636" if exact else "#2166c2")
        axes[1, column].set_xlabel(
            f"gallery #{row['top1_gallery_index']} · {'exact' if exact else 'different'}",
            fontsize=8.5, color="#238636" if exact else "#2166c2", labelpad=9,
        )

    axes[1, 0].axis("off")
    legend = FancyBboxPatch((0.08, 0.33), 0.79, 0.3, boxstyle="round,pad=0.03,rounding_size=0.04",
                            linewidth=0.8, edgecolor="#c5ced8", facecolor="#f7f9fc",
                            transform=axes[1, 0].transAxes)
    axes[1, 0].add_patch(legend)
    axes[1, 0].text(0.16, 0.53, "Green = exact target", transform=axes[1, 0].transAxes,
                    fontsize=8.5, color="#238636")
    axes[1, 0].text(0.16, 0.40, "Blue = different Top-1", transform=axes[1, 0].transAxes,
                    fontsize=8.5, color="#2166c2")

    filename = (f"latent__{first['dataset']}__{first['model'].replace('_ae', '').replace('_', '')}"
                f"__losswise_reconstruction_top1__case{case_id:02d}.png")
    path = output_dir / filename
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    set_seed(args.seed)
    runs = read_runs(args.runs_root, args.dataset, [args.model], list(LOSS_ORDER))
    cases = select_cases(runs, args.max_cases)
    output_dir = Path(args.output_dir)
    write_selection(cases, output_dir)
    if not cases:
        print("No contrast cases found.")
        return
    prepared = {}
    written = 0
    for case_id, case in enumerate(cases, 1):
        path = render_case(case_id, case, args, output_dir, prepared)
        if path:
            written += 1
            print(f"Wrote {path}")
    print(f"Rendered {written} compact figures.")


if __name__ == "__main__":
    main()
