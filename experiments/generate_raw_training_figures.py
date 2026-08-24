"""Generate objective-specific raw train/validation curves for the report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "outputs" / "final_windowed4096"
OUTPUT_DIR = ROOT / "report" / "figures"

DATASETS = {
    "dvsgesture": "DVSGesture",
    "ncaltech101": "N-Caltech101",
    "nmnist": "N-MNIST",
}
MODELS = {
    "pointnet_ae": "PointNet",
    "pointnetpp_ae": "PointNet++",
}
OBJECTIVES = {
    "chamfer": "Chamfer Distance",
    "density_aware_chamfer": "Density-Aware Chamfer Distance",
    "hausdorff": "Hausdorff Distance",
    "projection": "Projection loss",
    "temporal_weighted_chamfer": "Temporal-Weighted Chamfer Distance",
}


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows.sort(key=lambda row: int(row["epoch"]))
    epochs = np.array([int(row["epoch"]) + 1 for row in rows])
    train = np.array([float(row["train_loss"]) for row in rows])
    validation = np.array([float(row["val_loss"]) for row in rows])
    return epochs, train, validation


def plot_objective(objective: str, title: str) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 5.8),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for column, (dataset, dataset_label) in enumerate(DATASETS.items()):
        for row, (model, model_label) in enumerate(MODELS.items()):
            axis = axes[row, column]
            run = f"{dataset}_{model}_{objective}"
            history_path = HISTORY_ROOT / run / f"{run}_history.csv"

            if not history_path.exists():
                axis.text(
                    0.5,
                    0.5,
                    "Run unavailable",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue

            epochs, train, validation = load_history(history_path)
            best_index = int(np.argmin(validation))
            best_epoch = int(epochs[best_index])

            axis.plot(epochs, train, color="#0072B2", linewidth=1.5, label="Train")
            axis.plot(
                epochs,
                validation,
                color="#D55E00",
                linewidth=1.5,
                label="Validation",
            )
            axis.axvline(
                best_epoch,
                color="#555555",
                linewidth=1,
                linestyle=":",
                label="Best checkpoint",
            )
            axis.scatter(
                best_epoch,
                validation[best_index],
                color="#222222",
                s=18,
                zorder=3,
            )
            axis.set_title(f"{dataset_label} - {model_label}", fontsize=9)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Raw loss")
            axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.suptitle(f"{title}: raw training dynamics", fontsize=11)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"raw_training_curves_{objective}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for objective, title in OBJECTIVES.items():
        plot_objective(objective, title)
    print(f"Generated {len(OBJECTIVES)} figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
