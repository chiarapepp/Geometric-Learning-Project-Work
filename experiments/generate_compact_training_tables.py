"""Generate compact quantitative LaTeX convergence tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "outputs" / "final_windowed4096"
TEMPORAL = ROOT / "slurm_runs" / "outputs" / "time_weight_ablation_windowed4096"
OUT = ROOT / "report" / "tables"
DATASETS = {"dvsgesture": "DVSGesture", "ncaltech101": "N-Caltech101", "nmnist": "N-MNIST"}
MODELS = {"pointnet_ae": "PointNet", "pointnetpp_ae": "PointNet++"}
LOSSES = {"chamfer": "Chamfer", "density_aware_chamfer": "DCD", "hausdorff": "Hausdorff"}


def summarize(path: Path) -> tuple[int, int, float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["epoch"]))
    values = np.array([float(row["val_loss"]) for row in rows])
    seconds = np.array([float(row["epoch_seconds"]) for row in rows])
    best = int(np.argmin(values))
    return len(rows), int(rows[best]["epoch"]) + 1, float(values[best]), float(np.mean(seconds))


def row(dataset: str, model: str, objective: str, path: Path) -> str:
    epochs, best_epoch, best_loss, mean_seconds = summarize(path)
    return (
        f"{DATASETS[dataset]} & {MODELS[model]} & {objective} & {epochs} & "
        f"{best_epoch} & {best_loss:.6g} & {mean_seconds:.1f} \\\\"
    )


def standard_path(dataset: str, model: str, loss: str) -> Path:
    run = f"{dataset}_{model}_{loss}"
    return STANDARD / run / f"{run}_history.csv"


def temporal_path(dataset: str, model: str, weight: int) -> Path:
    directory = f"{dataset}_{model}_temporal_weighted_chamfer_tw{weight}"
    filename = f"{dataset}_{model}_temporal_weighted_chamfer_history.csv"
    return TEMPORAL / directory / filename


def write_table(path: Path, caption: str, label: str, rows: list[str]) -> None:
    lines = [
        r"\begin{longtable}{lllrrrr}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        r"Dataset & Model & Objective & Epochs & Best epoch & Best val. loss & Time/epoch (s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Model & Objective & Epochs & Best epoch & Best val. loss & Time/epoch (s) \\",
        r"\midrule",
        r"\endhead",
        *rows,
        r"\bottomrule",
        r"\end{longtable}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    primary: list[str] = []
    sinkhorn: list[str] = []
    for dataset in DATASETS:
        for model in MODELS:
            for loss, label in LOSSES.items():
                primary.append(row(dataset, model, label, standard_path(dataset, model, loss)))
            for weight in (2, 5):
                primary.append(
                    row(
                        dataset,
                        model,
                        rf"Temporal Chamfer ($\lambda_t={weight}$)",
                        temporal_path(dataset, model, weight),
                    )
                )
            sinkhorn_path = standard_path(dataset, model, "sinkhorn")
            if sinkhorn_path.exists():
                sinkhorn.append(row(dataset, model, "Sinkhorn", sinkhorn_path))

    OUT.mkdir(parents=True, exist_ok=True)
    write_table(
        OUT / "training_convergence_summary.tex",
        "Quantitative summary of the completed autoencoder training runs.",
        "tab:training-convergence",
        primary,
    )
    write_table(
        OUT / "sinkhorn_reduced_training_summary.tex",
        "Available reduced training runs for the computationally expensive Sinkhorn objective.",
        "tab:sinkhorn-reduced-training",
        sinkhorn,
    )
    print(f"Wrote {len(primary)} main rows and {len(sinkhorn)} Sinkhorn rows.")


if __name__ == "__main__":
    main()
