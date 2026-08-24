"""Create normalized validation curves and convergence-profile statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "final_windowed4096"
OUT = ROOT / "report" / "figures"
OBJECTIVES = [
    "chamfer",
    "density_aware_chamfer",
    "hausdorff",
    "projection",
    "temporal_weighted_chamfer",
]
COLORS = {
    "chamfer": "#0072B2",
    "density_aware_chamfer": "#D55E00",
    "hausdorff": "#009E73",
    "projection": "#CC79A7",
    "temporal_weighted_chamfer": "#E69F00",
}
LINESTYLES = {
    "chamfer": "-",
    "density_aware_chamfer": "--",
    "hausdorff": "-.",
    "projection": ":",
    "temporal_weighted_chamfer": (0, (5, 1)),
}
LABELS = {
    "chamfer": "Chamfer",
    "density_aware_chamfer": "DCD",
    "hausdorff": "Hausdorff",
    "projection": "Projection",
    "temporal_weighted_chamfer": "Temporal Chamfer",
}


def read_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_epoch = {int(row["epoch"]): row for row in rows}
    epochs = np.array(sorted(by_epoch), dtype=int)
    train = np.array([float(by_epoch[e]["train_loss"] ) for e in epochs])
    values = np.array([float(by_epoch[e]["val_loss"] ) for e in epochs])
    seconds = np.array([float(by_epoch[e]["epoch_seconds"] ) for e in epochs])
    return epochs, train, values, seconds


def main() -> None:
    datasets = ["dvsgesture", "ncaltech101", "nmnist"]
    models = ["pointnet_ae", "pointnetpp_ae"]
    dataset_labels = {
        "dvsgesture": "DVSGesture",
        "ncaltech101": "N-Caltech101",
        "nmnist": "N-MNIST",
    }
    model_labels = {"pointnet_ae": "PointNet", "pointnetpp_ae": "PointNet++"}
    statistics: list[dict[str, str | float | int]] = []

    fig, axes = plt.subplots(
        2, 3, figsize=(10.5, 5.8), sharex=True, constrained_layout=True
    )
    for column, dataset in enumerate(datasets):
        for row_index, model in enumerate(models):
            axis = axes[row_index, column]
            for objective in OBJECTIVES:
                run_name = f"{dataset}_{model}_{objective}"
                epochs, train, values, seconds = read_history(
                    SOURCE / run_name / f"{run_name}_history.csv"
                )
                initial = values[0]
                best = values.min()
                threshold = initial - 0.9 * (initial - best)
                epoch_90 = int(epochs[np.flatnonzero(values <= threshold)[0]])
                normalized = values / initial
                statistics.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "loss": objective,
                        "relative_reduction_percent": 100
                        * (initial - best)
                        / initial,
                        "epoch_best": int(epochs[values.argmin()]),
                        "epoch_90_percent_improvement": epoch_90,
                        "mean_epoch_seconds": float(np.mean(seconds)),
                        "normalized_auc": float(
                            np.trapz(normalized, epochs) / (epochs[-1] - epochs[0])
                        ),
                    }
                )
                axis.plot(
                    epochs + 1,
                    normalized,
                    linewidth=1.8,
                    color=COLORS[objective],
                    linestyle=LINESTYLES[objective],
                    label=LABELS[objective],
                )

            axis.set_title(
                f"{dataset_labels[dataset]} ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {model_labels[model]}", fontsize=9
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(r"Validation loss / $L_0$")
            axis.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        fontsize=8.5,
    )
    fig.savefig(OUT / "normalized_validation_curves.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "normalized_validation_curves.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    with (OUT / "convergence_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)


if __name__ == "__main__":
    main()
