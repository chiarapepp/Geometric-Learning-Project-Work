"""Aggregate the 4096-point all-metrics checkpoint evaluation suite."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="slurm_runs/outputs/all_metrics_eval_windowed4096",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/all_metrics_common_eval",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def objective_label(path, first_row):
    run_id = path.parent.name
    if run_id.endswith("_temporal_weighted_chamfer_tw2"):
        return "temporal_weighted_chamfer_tw2"
    if run_id.endswith("_temporal_weighted_chamfer_tw5"):
        return "temporal_weighted_chamfer_tw5"
    return first_row["trained_loss"]


def summarize(values):
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def condition_label(corruption, level):
    if corruption == "clean":
        return "clean"
    if corruption == "gaussian_noise" and level == 0.1:
        return "noise_0p1"
    if corruption == "temporal_shuffle" and level == 1.0:
        return "shuffle_1"
    if corruption == "random_drop" and level == 0.5:
        return "drop_0p5"
    return None


def main():
    args = parse_args()
    root = Path(args.root)
    paths = sorted(root.rglob("*_all_metrics.csv"))
    if not paths:
        raise SystemExit(f"No all-metrics CSV files found under {root}")

    grouped = defaultdict(lambda: {"reconstruction": [], "input": []})
    coverage = set()
    for path in paths:
        rows = read_csv(path)
        if not rows:
            continue
        objective = objective_label(path, rows[0])
        coverage.add((rows[0]["dataset"], rows[0]["model"], objective))
        for row in rows:
            key = (
                row["dataset"],
                row["model"],
                objective,
                row["metric"],
                row["corruption"],
                float(row["corruption_level"]),
            )
            grouped[key]["reconstruction"].append(float(row["reconstruction_value"]))
            grouped[key]["input"].append(float(row["corrupted_input_value"]))

    summary = []
    for key, values in sorted(grouped.items()):
        dataset, model, objective, metric, corruption, level = key
        rec_mean, rec_std = summarize(values["reconstruction"])
        input_mean, input_std = summarize(values["input"])
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "training_objective": objective,
                "metric": metric,
                "metric_direction": "higher" if metric.startswith("fscore_") else "lower",
                "corruption": corruption,
                "corruption_level": level,
                "mean_reconstruction_value": rec_mean,
                "std_reconstruction_value": rec_std,
                "mean_corrupted_input_value": input_mean,
                "std_corrupted_input_value": input_std,
                "num_batches": len(values["reconstruction"]),
            }
        )

    endpoint_groups = defaultdict(list)
    endpoint_input_groups = defaultdict(list)
    for row in summary:
        condition = condition_label(row["corruption"], float(row["corruption_level"]))
        if condition is None:
            continue
        key = (row["training_objective"], row["metric"], condition)
        endpoint_groups[key].append(float(row["mean_reconstruction_value"]))
        endpoint_input_groups[(row["metric"], condition)].append(
            float(row["mean_corrupted_input_value"])
        )

    macro_endpoints = []
    for key, values in sorted(endpoint_groups.items()):
        objective, metric, condition = key
        macro_endpoints.append(
            {
                "training_objective": objective,
                "metric": metric,
                "metric_direction": "higher" if metric.startswith("fscore_") else "lower",
                "condition": condition,
                "mean_reconstruction_value": mean(values),
                "std_across_dataset_model_pairs": stdev(values) if len(values) > 1 else 0.0,
                "num_dataset_model_pairs": len(values),
                "mean_corrupted_input_value": mean(endpoint_input_groups[(metric, condition)]),
            }
        )

    expected = {
        (dataset, model, objective)
        for dataset in ("dvsgesture", "nmnist", "ncaltech101")
        for model in ("pointnet_ae", "pointnetpp_ae")
        for objective in (
            "chamfer",
            "density_aware_chamfer",
            "hausdorff",
            "temporal_weighted_chamfer_tw2",
            "temporal_weighted_chamfer_tw5",
        )
    }
    missing = sorted(expected - coverage)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "all_metrics_summary.csv", summary)
    write_csv(output_dir / "macro_clean_and_endpoints.csv", macro_endpoints)

    print(f"Found {len(paths)} files; coverage {len(coverage)}/{len(expected)}.")
    if missing:
        print("Missing evaluations:")
        for item in missing:
            print("  ", item)
    else:
        print("All 30 checkpoint evaluations are complete.")
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()