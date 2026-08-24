"""Generate raw training curves for temporal Chamfer time-weight ablations."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "outputs" / "time_weight_ablation_windowed4096"
OUTPUT_DIR = ROOT / "report" / "figures"

DATASETS = {"dvsgesture": "DVSGesture", "ncaltech101": "N-Caltech101", "nmnist": "N-MNIST"}
MODELS = {"pointnet_ae": "PointNet", "pointnetpp_ae": "PointNet++"}


def load_history(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["epoch"]))
    return (
        np.array([int(row["epoch"]) + 1 for row in rows]),
        np.array([float(row["train_loss"]) for row in rows]),
        np.array([float(row["val_loss"]) for row in rows]),
    )


def plot_weight(weight: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), constrained_layout=True)
    suffix = f"tw{weight}"

    for column, (dataset, dataset_label) in enumerate(DATASETS.items()):
        for row, (model, model_label) in enumerate(MODELS.items()):
            axis = axes[row, column]
            run = f"{dataset}_{model}_temporal_weighted_chamfer_{suffix}"
            path = HISTORY_ROOT / run / f"{dataset}_{model}_temporal_weighted_chamfer_history.csv"
            if not path.exists():
                axis.text(0.5, 0.5, "Run unavailable", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue

            epochs, train, validation = load_history(path)
            best = int(np.argmin(validation))
            axis.plot(epochs, train, color="#0072B2", linewidth=1.5, label="Train")
            axis.plot(epochs, validation, color="#D55E00", linewidth=1.5, label="Validation")
            axis.axvline(epochs[best], color="#555555", linewidth=1, linestyle=":", label="Best checkpoint")
            axis.scatter(epochs[best], validation[best], color="#222222", s=18, zorder=3)
            axis.set_title(f"{dataset_label} - {model_label}", fontsize=9)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Raw loss")
            axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    figure.suptitle(f"Temporal-Weighted Chamfer (lambda = {weight}): raw training dynamics", fontsize=11)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"raw_training_curves_temporal_weighted_chamfer_tw{weight}"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    for weight in (2, 5):
        plot_weight(weight)
    print(f"Generated lambda=2 and lambda=5 figures in {OUTPUT_DIR}")
