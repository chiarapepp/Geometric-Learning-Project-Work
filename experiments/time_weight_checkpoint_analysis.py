"""Aggregate common-metric evaluation for temporal-weight checkpoints."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="slurm_runs/outputs/time_weight_ablation_eval_windowed4096",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/time_weight_ablation_common_metrics",
    )
    parser.add_argument(
        "--canonical-eval-time-weight",
        type=float,
        default=2.0,
        help="Evaluation run used for Chamfer and Hausdorff, which are duplicated across eval weights.",
    )
    return parser.parse_args()


def safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def weight_from_text(text: str, prefix: str) -> float | None:
    match = re.search(rf"{prefix}([0-9]+(?:p[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def metric_label(metric: str, eval_weight: float | None) -> str:
    if metric == "temporal_weighted_chamfer":
        return f"temporal_chamfer_eval_tw{eval_weight:g}"
    return metric


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    paths = sorted(root.rglob("*_corruptions.csv"))
    if not paths:
        raise SystemExit(f"No evaluation CSV files found under {root}")

    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    coverage: set[tuple[str, str, float, float]] = set()

    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        file_eval_weight = weight_from_text(path.as_posix(), "evaltw")
        train_weight = weight_from_text(path.as_posix(), "_tw")
        if file_eval_weight is None or train_weight is None:
            raise ValueError(f"Cannot infer temporal weights from {path}")

        first = rows[0]
        coverage.add((first["dataset"], first["model"], train_weight, file_eval_weight))

        for row in rows:
            metric = row["metric"]
            if metric in {"chamfer", "hausdorff"}:
                if file_eval_weight != args.canonical_eval_time_weight:
                    continue
                eval_weight = None
            elif metric == "temporal_weighted_chamfer":
                eval_weight = safe_float(row.get("metric_time_weight"))
                if eval_weight is None:
                    eval_weight = file_eval_weight
            else:
                continue

            key = (
                row["dataset"],
                row["model"],
                train_weight,
                metric_label(metric, eval_weight),
                row["corruption"],
                float(row["corruption_level"]),
            )
            grouped[key].append(float(row["reconstruction_value"]))

    clean_long: list[dict[str, object]] = []
    corruption_long: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        dataset, model, train_weight, metric, corruption, level = key
        value_mean, value_std = stats(values)
        record = {
            "dataset": dataset,
            "model": model,
            "train_time_weight": train_weight,
            "metric": metric,
            "corruption": corruption,
            "corruption_level": level,
            "mean_reconstruction_value": value_mean,
            "std_reconstruction_value": value_std,
            "num_batches": len(values),
        }
        corruption_long.append(record)
        if corruption in {"clean", "gaussian_noise"} and level == 0.0:
            clean_long.append(record)

    clean_by_run: dict[tuple[str, str, float], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in clean_long:
        run_key = (str(row["dataset"]), str(row["model"]), float(row["train_time_weight"]))
        clean_by_run[run_key][str(row["metric"])] = row

    metric_order = [
        "chamfer",
        "hausdorff",
        "temporal_chamfer_eval_tw2",
        "temporal_chamfer_eval_tw5",
    ]
    clean_wide: list[dict[str, object]] = []
    for (dataset, model, train_weight), by_metric in sorted(clean_by_run.items()):
        record: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "train_time_weight": train_weight,
        }
        for metric in metric_order:
            row = by_metric.get(metric)
            record[f"{metric}_mean"] = "" if row is None else row["mean_reconstruction_value"]
            record[f"{metric}_std"] = "" if row is None else row["std_reconstruction_value"]
        clean_wide.append(record)

    pairwise: list[dict[str, object]] = []
    paired: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    for row in clean_long:
        paired[(str(row["dataset"]), str(row["model"]), str(row["metric"]))][
            float(row["train_time_weight"])
        ] = float(row["mean_reconstruction_value"])
    for (dataset, model, metric), values in sorted(paired.items()):
        if 2.0 not in values or 5.0 not in values:
            continue
        tw2 = values[2.0]
        tw5 = values[5.0]
        winner = 2 if tw2 < tw5 else 5 if tw5 < tw2 else "tie"
        relative_difference = 100.0 * (tw5 - tw2) / tw2 if tw2 != 0 else ""
        pairwise.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "train_tw2": tw2,
                "train_tw5": tw5,
                "lower_is_better_winner": winner,
                "tw5_relative_to_tw2_percent": relative_difference,
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "clean_common_metrics_long.csv", clean_long)
    write_csv(output_dir / "clean_common_metrics_wide.csv", clean_wide)
    write_csv(output_dir / "corruption_common_metrics.csv", corruption_long)
    write_csv(output_dir / "pairwise_tw2_vs_tw5.csv", pairwise)

    expected = {
        (dataset, model, train_weight, eval_weight)
        for dataset in ("dvsgesture", "nmnist", "ncaltech101")
        for model in ("pointnet_ae", "pointnetpp_ae")
        for train_weight in (2.0, 5.0)
        for eval_weight in (2.0, 5.0)
    }
    missing = sorted(expected - coverage)
    print(f"Found {len(paths)} evaluation files; coverage {len(coverage)}/{len(expected)}.")
    if missing:
        print("Missing combinations:")
        for item in missing:
            print("  ", item)
    else:
        print("All temporal-weight evaluation combinations are complete.")
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()