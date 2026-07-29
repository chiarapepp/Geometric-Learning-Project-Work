"""Build the compact quantitative figures used by the revised report.

The script only aggregates experiment artifacts already stored in ``outputs``;
it does not rerun training or evaluation.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

COLORS = {
    "chamfer": "#0072B2",
    "density_aware_chamfer": "#D55E00",
    "hausdorff": "#009E73",
    "sinkhorn": "#CC79A7",
}
LABELS = {
    "chamfer": "Chamfer",
    "density_aware_chamfer": "DCD",
    "hausdorff": "Hausdorff",
    "sinkhorn": "Sinkhorn",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def standalone_response() -> None:
    source = ROOT / "outputs" / "standalone_benchmark_rerun"
    core_objectives = [
        "chamfer",
        "density_aware_chamfer",
        "hausdorff",
        "sinkhorn",
    ]
    core_labels = {
        "chamfer": "Chamfer",
        "density_aware_chamfer": "DCD",
        "hausdorff": "Hausdorff",
        "sinkhorn": "Sinkhorn",
    }
    core_colors = {
        "chamfer": "#0072B2",
        "density_aware_chamfer": "#D55E00",
        "hausdorff": "#009E73",
        "sinkhorn": "#CC79A7",
    }
    core_markers = {
        "chamfer": "o",
        "density_aware_chamfer": "s",
        "hausdorff": "^",
        "sinkhorn": "D",
    }
    temporal_colors = {1: "#0072B2", 2: "#E69F00", 5: "#D55E00"}
    temporal_markers = {1: "o", 2: "s", 5: "^"}
    configurations = [
        ("noise", "noise_std", "Gaussian noise", r"Noise std. $\sigma$"),
        (
            "shuffle",
            "temporal_shuffle_fraction",
            "Temporal shuffle",
            r"Shuffled fraction $\rho$",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)

    # Top row: compare response shapes across objective families. Each curve is
    # normalized independently because the raw objective scales are different.
    for column, (suffix, level_col, title, xlabel) in enumerate(configurations):
        axis = axes[0, column]
        values: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for path in sorted(source.glob(f"*_core_tw1_{suffix}.csv")):
            for row in read_csv(path):
                loss = row["loss"]
                if loss not in core_objectives:
                    continue
                values[loss][float(row[level_col])].append(float(row["value"]))

        for loss in core_objectives:
            levels = sorted(values[loss])
            means = np.array([np.mean(values[loss][level]) for level in levels])
            denominator = means[-1] - means[0]
            response = (
                (means - means[0]) / denominator
                if denominator > 0
                else np.zeros_like(means)
            )
            axis.plot(
                levels,
                response,
                marker=core_markers[loss],
                linewidth=1.8,
                markersize=4.5,
                color=core_colors[loss],
                label=core_labels[loss],
            )

        axis.set_title(f"Core objectives: {title}", fontsize=10)
        axis.set_xlabel(xlabel)
        axis.set_ylim(-0.04, 1.06)
        axis.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Normalized response")
    axes[0, 0].legend(loc="upper left", ncol=2, frameon=False, fontsize=8)

    # Bottom row: retain raw scale within the Temporal Chamfer family so the
    # effect of changing lambda_t remains visible.
    for column, (suffix, level_col, title, xlabel) in enumerate(configurations):
        axis = axes[1, column]

        for time_weight in [1, 2, 5]:
            pattern = (
                f"*_core_tw1_{suffix}.csv"
                if time_weight == 1
                else f"*_temporal_tw{time_weight}_{suffix}.csv"
            )
            values: dict[float, list[float]] = defaultdict(list)

            for path in sorted(source.glob(pattern)):
                for row in read_csv(path):
                    if row["loss"] != "temporal_weighted_chamfer":
                        continue
                    values[float(row[level_col])].append(float(row["value"]))

            levels = sorted(values)
            means = np.array([np.mean(values[level]) for level in levels])
            response = means - means[0]
            axis.plot(
                levels,
                response,
                marker=temporal_markers[time_weight],
                linewidth=1.8,
                markersize=4.5,
                color=temporal_colors[time_weight],
                label=rf"$\lambda_t={time_weight}$",
            )

        axis.set_title(f"Temporal weighting: {title}", fontsize=10)
        axis.set_xlabel(xlabel)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(alpha=0.25)

    axes[1, 0].set_ylabel("Clean-subtracted response")
    axes[1, 0].legend(loc="upper left", frameon=False, fontsize=8)

    fig.savefig(OUT / "standalone_corruption_response.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "standalone_corruption_response.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

def reconstruction_robustness() -> None:
    tables = ROOT / "outputs" / "final" / "tables"
    clean_rows = read_csv(tables / "reconstruction_clean_summary.csv")
    corruption_rows = read_csv(tables / "corruption_robustness_summary.csv")

    clean: dict[tuple[str, str, str], float] = {}
    for row in clean_rows:
        if row["metric"] == "chamfer":
            key = (row["dataset"], row["model"], row["trained_loss"])
            clean[key] = float(row["mean_reconstruction_value"])

    objectives = ["chamfer", "density_aware_chamfer", "hausdorff"]
    panels = [
        ("gaussian_noise", r"Gaussian noise ($\sigma$)"),
        ("temporal_shuffle", r"Temporal shuffle ($\rho$)"),
        ("random_drop", r"Random drop ($\delta$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.85), constrained_layout=True)

    for axis, (corruption, xlabel) in zip(axes, panels):
        ratios: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in corruption_rows:
            if (
                row["metric"] != "chamfer"
                or row["corruption"] != corruption
                or row["trained_loss"] not in objectives
            ):
                continue
            objective = row["trained_loss"]
            key = (row["dataset"], row["model"], objective)
            ratio = float(row["mean_reconstruction_value"]) / clean[key]
            ratios[objective][float(row["corruption_level"])].append(ratio)

        for objective in objectives:
            levels = [0.0] + sorted(ratios[objective])
            means = [1.0] + [
                float(np.mean(ratios[objective][level])) for level in levels[1:]
            ]
            axis.plot(
                levels,
                means,
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=COLORS[objective],
                label=LABELS[objective],
            )

        axis.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"$E_{\mathrm{rec}}/E_{\mathrm{clean}}$")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.055),
        fontsize=8,
    )
    fig.savefig(OUT / "reconstruction_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "reconstruction_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    standalone_response()
    reconstruction_robustness()


if __name__ == "__main__":
    main()
