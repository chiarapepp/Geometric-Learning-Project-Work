import argparse
import csv
import io
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASETS = ["dvsgesture", "nmnist", "ncaltech101"]
DEFAULT_MODELS = ["pointnet_ae", "pointnetpp_ae"]
DEFAULT_LOSSES = [
    "chamfer",
    "density_aware_chamfer",
    "sinkhorn",
    "temporal_weighted_chamfer",
    "hausdorff",
]
DEFAULT_METRICS = ["chamfer", "temporal_weighted_chamfer", "hausdorff", "mse"]
DEFAULT_TIME_WEIGHTS = [1.0, 2.0, 5.0, 10.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final windowed4096 project-work results into coverage tables, "
            "summary CSV/Markdown files, plots, and Italian report notes."
        )
    )
    parser.add_argument("--root", default="outputs/final_windowed4096")
    parser.add_argument("--benchmark-root", nargs="+", default=["outputs/windows_exp/benchmarks"])
    parser.add_argument("--eval-root", default="outputs/final_windowed4096/eval")
    parser.add_argument("--time-weight-root", default="outputs/time_weight_ablation")
    parser.add_argument("--time-weight-eval-root", default="outputs/time_weight_ablation_eval")
    parser.add_argument("--visual-root", nargs="+", default=["outputs/visual_eval", "outputs/final_windowed4096/visual_eval"])
    parser.add_argument("--output-dir", default="outputs/final_report")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--include-emd-small", action="store_true")
    parser.add_argument("--primary-metric", default="chamfer")
    parser.add_argument("--time-weights", nargs="+", type=float, default=DEFAULT_TIME_WEIGHTS)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if expected coverage is incomplete.")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (csv.Error, UnicodeDecodeError):
        # Some interrupted cluster transfers can leave NUL bytes in CSV files.
        # Keep the final audit usable and let coverage/row counts expose gaps.
        text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        if not text.strip():
            return []
        return list(csv.DictReader(io.StringIO(text)))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value):
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")


def write_markdown_table(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(markdown_escape(row.get(key, "")) for key in fieldnames) + " |\n")


def safe_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value, default=None):
    number = safe_float(value)
    if number is None:
        return default
    return int(number)


def format_number(value):
    number = safe_float(value)
    if number is None:
        return ""
    if number == 0:
        return "0"
    if abs(number) >= 1e4 or abs(number) < 1e-3:
        return f"{number:.4e}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def stats(values):
    numbers = [safe_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return {"mean": "", "std": "", "min": "", "max": "", "n": 0}
    return {
        "mean": mean(numbers),
        "std": pstdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
        "n": len(numbers),
    }


def normalize_dataset(name):
    if name == "mnist":
        return "nmnist"
    return name


def dataset_aliases(dataset):
    if dataset == "nmnist":
        return ["nmnist", "mnist"]
    return [dataset]


def expected_losses(args):
    losses = list(args.losses)
    if args.include_emd_small and "emd" not in losses:
        losses.append("emd")
    return losses


def parse_identity(path, datasets=None, models=None, losses=None):
    name = Path(path).name.lower()

    # Remove common suffixes.
    for suffix in ["_history.csv", "_best.pth", "_corruptions.csv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove epoch suffix, e.g. _epoch_50.pth
    if "_epoch_" in name and name.endswith(".pth"):
        name = name.rsplit("_epoch_", 1)[0]

    if "." in name:
        name = Path(name).stem

    datasets = list(datasets or ["ncaltech101", "dvsgesture", "nmnist"])
    models = list(models or ["pointnetpp_ae", "pointnet_ae"])
    losses = list(
        losses
        or [
            "density_aware_chamfer",
            "temporal_weighted_chamfer",
            "chamfer",
            "hausdorff",
            "sinkhorn",
            "projection",
            "voxel",
            "chamfer_squared",
            "repulsion_chamfer",
        ]
    )

    known_losses = [
        "density_aware_chamfer",
        "temporal_weighted_chamfer",
        "chamfer",
        "hausdorff",
        "sinkhorn",
        "projection",
        "voxel",
        "chamfer_squared",
        "repulsion_chamfer",
        "emd",
    ]
    for loss_name in known_losses:
        if loss_name not in losses:
            losses.append(loss_name)

    for dataset in datasets:
        for alias in dataset_aliases(dataset):
            prefix = alias.lower() + "_"
            if not name.startswith(prefix):
                continue

            rest = name[len(prefix) :]

            for model in sorted(models, key=len, reverse=True):
                model_prefix = model.lower() + "_"
                if not rest.startswith(model_prefix):
                    continue

                loss_part = rest[len(model_prefix) :]

                for loss_name in sorted(losses, key=len, reverse=True):
                    if loss_part == loss_name or loss_part.startswith(loss_name + "_"):
                        return dataset, model, loss_name

    return None, None, None


def compact_path(path):
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def scan_files(root, pattern):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def discover_training_artifacts(args):
    root = Path(args.root)
    losses = expected_losses(args)
    history_files = scan_files(root, "*_history.csv")
    checkpoint_files = scan_files(root, "*_best.pth")
    epoch_checkpoint_files = scan_files(root, "*_epoch_*.pth")

    histories = defaultdict(list)
    checkpoints = defaultdict(list)
    epoch_checkpoints = defaultdict(list)

    for path in history_files:
        identity = parse_identity(path, args.datasets, args.models, losses)
        if identity[0]:
            histories[identity].append(path)
    for path in checkpoint_files:
        identity = parse_identity(path, args.datasets, args.models, losses)
        if identity[0]:
            checkpoints[identity].append(path)
    for path in epoch_checkpoint_files:
        identity = parse_identity(path, args.datasets, args.models, losses)
        if identity[0]:
            epoch_checkpoints[identity].append(path)
    return histories, checkpoints, epoch_checkpoints


def discover_eval_files(args):
    losses = expected_losses(args)
    eval_files = scan_files(args.eval_root, "*_corruptions.csv")
    evals = defaultdict(list)
    for path in eval_files:
        rows = read_csv(path)
        identity = None
        if rows:
            row = rows[0]
            identity = (
                normalize_dataset(row.get("dataset", "")),
                row.get("model", ""),
                row.get("trained_loss", ""),
            )
        if not identity or not identity[0]:
            identity = parse_identity(path, args.datasets, args.models, losses)
        if identity[0] in args.datasets and identity[1] in args.models and identity[2] in losses:
            evals[identity].append(path)
    return evals


def discover_benchmark_files(args):
    paths = []
    for root in args.benchmark_root:
        paths.extend(scan_files(root, "*.csv"))
    return sorted(set(paths))


def benchmark_kind(path):
    name = Path(path).stem.lower()
    if "noise" in name:
        return "gaussian_noise"
    if "temporal_shuffle" in name:
        return "temporal_shuffle"
    if "emd" in name:
        return "emd_small"
    if "loss_comparison" in name:
        return "clean"
    return "benchmark"


def corruption_from_benchmark_row(row, kind):
    noise = safe_float(row.get("noise_std"), 0.0) or 0.0
    shuffle = safe_float(row.get("temporal_shuffle_fraction"), 0.0) or 0.0
    drop = safe_float(row.get("drop_fraction"), 0.0) or 0.0
    if noise > 0:
        return "gaussian_noise", noise
    if shuffle > 0:
        return "temporal_shuffle", shuffle
    if drop > 0:
        return "random_drop", drop
    if kind == "emd_small":
        return "clean_emd_small", 0.0
    return "clean", 0.0


def load_benchmark_rows(args):
    losses = expected_losses(args)
    rows = []
    for path in discover_benchmark_files(args):
        kind = benchmark_kind(path)
        for row in read_csv(path):
            dataset = normalize_dataset(row.get("dataset", ""))
            loss_name = row.get("loss", "")
            if dataset not in args.datasets or loss_name not in losses:
                continue
            corruption, level = corruption_from_benchmark_row(row, kind)
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["loss"] = loss_name
            enriched["benchmark_kind"] = kind
            enriched["corruption"] = corruption
            enriched["corruption_level"] = level
            enriched["source_file"] = compact_path(path)
            rows.append(enriched)
    return rows


def load_eval_rows(args):
    losses = expected_losses(args)
    rows = []
    for path in scan_files(args.eval_root, "*_corruptions.csv"):
        file_rows = read_csv(path)
        for row in file_rows:
            dataset = normalize_dataset(row.get("dataset", ""))
            model = row.get("model", "")
            trained_loss = row.get("trained_loss", "")
            metric = row.get("metric", "")
            if dataset not in args.datasets or model not in args.models:
                continue
            if trained_loss not in losses or metric not in args.metrics:
                continue
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["source_file"] = compact_path(path)
            rows.append(enriched)
    return rows


def first_path(paths):
    if not paths:
        return ""
    return compact_path(sorted(paths)[0])


def build_coverage(args, histories, checkpoints, epoch_checkpoints, evals, benchmark_rows):
    losses = expected_losses(args)
    benchmark_present = set()
    for row in benchmark_rows:
        benchmark_present.add((row["dataset"], row["loss"]))

    rows = []
    for dataset in args.datasets:
        for model in args.models:
            for loss_name in losses:
                identity = (dataset, model, loss_name)
                history_count = len(histories.get(identity, []))
                best_count = len(checkpoints.get(identity, []))
                epoch_count = len(epoch_checkpoints.get(identity, []))
                eval_count = len(evals.get(identity, []))
                benchmark_count = 1 if (dataset, loss_name) in benchmark_present else 0
                status = "complete" if history_count and best_count and eval_count and benchmark_count else "missing"
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "loss": loss_name,
                        "history_present": bool(history_count),
                        "best_checkpoint_present": bool(best_count),
                        "epoch_checkpoints_present": bool(epoch_count),
                        "eval_present": bool(eval_count),
                        "benchmark_present": bool(benchmark_count),
                        "history_count": history_count,
                        "best_checkpoint_count": best_count,
                        "epoch_checkpoint_count": epoch_count,
                        "eval_file_count": eval_count,
                        "status": status,
                        "first_history": first_path(histories.get(identity, [])),
                        "first_checkpoint": first_path(checkpoints.get(identity, [])),
                        "first_eval": first_path(evals.get(identity, [])),
                    }
                )
    return rows


def summarize_histories(args, histories):
    rows = []
    for identity, paths in sorted(histories.items()):
        dataset, model, loss_name = identity
        for path in paths:
            history = read_csv(path)
            if not history:
                continue
            valid = [row for row in history if safe_float(row.get("val_loss")) is not None]
            if not valid:
                continue
            best = min(valid, key=lambda row: safe_float(row.get("val_loss")))
            final = max(valid, key=lambda row: safe_int(row.get("epoch"), -1))
            epoch_seconds = [safe_float(row.get("epoch_seconds")) for row in valid]
            epoch_seconds = [value for value in epoch_seconds if value is not None]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "loss": loss_name,
                    "epochs": len(valid),
                    "best_epoch": safe_int(best.get("epoch")),
                    "best_val_loss": format_number(best.get("val_loss")),
                    "best_train_loss": format_number(best.get("train_loss")),
                    "final_epoch": safe_int(final.get("epoch")),
                    "final_val_loss": format_number(final.get("val_loss")),
                    "final_train_loss": format_number(final.get("train_loss")),
                    "mean_epoch_seconds": format_number(mean(epoch_seconds) if epoch_seconds else None),
                    "total_epoch_seconds": format_number(sum(epoch_seconds) if epoch_seconds else None),
                    "num_points": final.get("num_points", ""),
                    "input_dim": final.get("input_dim", ""),
                    "loss_time_weight": final.get("loss_time_weight", ""),
                    "history_path": compact_path(path),
                }
            )
    return rows


def summarize_loss_benchmarks(benchmark_rows):
    grouped = defaultdict(list)
    for row in benchmark_rows:
        corruption = row.get("corruption")
        if corruption not in ("clean", "clean_emd_small"):
            continue
        key = (row["dataset"], row["loss"], corruption)
        grouped[key].append(row)

    summaries = []
    for (dataset, loss_name, corruption), rows in sorted(grouped.items()):
        value_stats = stats(row.get("value") for row in rows)
        second_stats = stats(row.get("seconds") for row in rows)
        memory_stats = stats(row.get("peak_memory_mb") for row in rows)
        flops_stats = stats(row.get("estimated_flops") for row in rows)
        throughput_stats = stats(row.get("estimated_flops_per_second") for row in rows)
        summaries.append(
            {
                "dataset": dataset,
                "loss": loss_name,
                "protocol": corruption,
                "mean_value": format_number(value_stats["mean"]),
                "std_value": format_number(value_stats["std"]),
                "mean_seconds": format_number(second_stats["mean"]),
                "std_seconds": format_number(second_stats["std"]),
                "mean_peak_memory_mb": format_number(memory_stats["mean"]),
                "mean_estimated_gflops": format_number((flops_stats["mean"] or 0.0) / 1e9),
                "mean_estimated_gflops_per_second": format_number((throughput_stats["mean"] or 0.0) / 1e9),
                "num_rows": value_stats["n"],
            }
        )
    return summaries


def summarize_eval_clean(eval_rows):
    grouped = defaultdict(list)
    for row in eval_rows:
        level = safe_float(row.get("corruption_level"))
        if level != 0.0:
            continue
        # Every corruption family includes a level-zero control. Keep one canonical
        # clean control when available to avoid triplicating the same condition.
        if row.get("corruption") not in ("gaussian_noise", "clean", ""):
            continue
        key = (row["dataset"], row["model"], row["trained_loss"], row["metric"])
        grouped[key].append(row)

    summaries = []
    for (dataset, model, loss_name, metric), rows in sorted(grouped.items()):
        rec_stats = stats(row.get("reconstruction_value") for row in rows)
        input_stats = stats(row.get("corrupted_input_value") for row in rows)
        seconds_stats = stats(row.get("model_seconds") for row in rows)
        memory_stats = stats(row.get("peak_memory_mb") for row in rows)
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "trained_loss": loss_name,
                "metric": metric,
                "mean_reconstruction_value": format_number(rec_stats["mean"]),
                "std_reconstruction_value": format_number(rec_stats["std"]),
                "mean_clean_input_value": format_number(input_stats["mean"]),
                "mean_model_seconds": format_number(seconds_stats["mean"]),
                "mean_peak_memory_mb": format_number(memory_stats["mean"]),
                "num_rows": rec_stats["n"],
            }
        )
    return summaries


def clean_baselines(clean_summary):
    baselines = {}
    for row in clean_summary:
        key = (row["dataset"], row["model"], row["trained_loss"], row["metric"])
        baselines[key] = safe_float(row.get("mean_reconstruction_value"))
    return baselines


def summarize_corruptions(eval_rows, clean_summary):
    baselines = clean_baselines(clean_summary)
    grouped = defaultdict(list)
    for row in eval_rows:
        level = safe_float(row.get("corruption_level"))
        if level is None or level == 0.0:
            continue
        key = (
            row["dataset"],
            row["model"],
            row["trained_loss"],
            row["metric"],
            row.get("corruption", ""),
            level,
        )
        grouped[key].append(row)

    summaries = []
    for (dataset, model, loss_name, metric, corruption, level), rows in sorted(grouped.items()):
        rec_stats = stats(row.get("reconstruction_value") for row in rows)
        input_stats = stats(row.get("corrupted_input_value") for row in rows)
        baseline = baselines.get((dataset, model, loss_name, metric))
        degradation = None
        if baseline is not None and rec_stats["mean"] != "":
            degradation = rec_stats["mean"] - baseline
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "trained_loss": loss_name,
                "metric": metric,
                "corruption": corruption,
                "corruption_level": format_number(level),
                "mean_reconstruction_value": format_number(rec_stats["mean"]),
                "mean_corrupted_input_value": format_number(input_stats["mean"]),
                "absolute_degradation_vs_clean": format_number(degradation),
                "num_rows": rec_stats["n"],
            }
        )
    return summaries


def summarize_models(clean_summary, corruption_summary, primary_metric):
    clean_grouped = defaultdict(list)
    time_grouped = defaultdict(list)
    for row in clean_summary:
        if row["metric"] != primary_metric:
            continue
        key = (row["dataset"], row["model"])
        clean_grouped[key].append(row.get("mean_reconstruction_value"))
        time_grouped[key].append(row.get("mean_model_seconds"))

    robust_grouped = defaultdict(list)
    for row in corruption_summary:
        if row["metric"] != primary_metric:
            continue
        key = (row["dataset"], row["model"])
        robust_grouped[key].append(row.get("absolute_degradation_vs_clean"))

    summaries = []
    for key in sorted(set(clean_grouped) | set(robust_grouped)):
        dataset, model = key
        clean_stats = stats(clean_grouped.get(key, []))
        robust_stats = stats(robust_grouped.get(key, []))
        time_stats = stats(time_grouped.get(key, []))
        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": primary_metric,
                "mean_clean_reconstruction": format_number(clean_stats["mean"]),
                "mean_corruption_degradation": format_number(robust_stats["mean"]),
                "mean_model_seconds": format_number(time_stats["mean"]),
                "num_clean_groups": clean_stats["n"],
                "num_corruption_groups": robust_stats["n"],
            }
        )
    return summaries


def extract_time_weight(path, rows):
    for row in rows:
        value = safe_float(row.get("loss_time_weight"))
        if value is not None:
            return value
    match = re.search(r"_tw([0-9p]+)", str(path))
    if match:
        return safe_float(match.group(1).replace("p", "."))
    if "chamfer" in str(path):
        return 1.0
    return None


def summarize_time_weight(args):
    roots = [Path(args.time_weight_root), Path(args.root) / "time_weight_ablation"]
    histories = []
    for root in roots:
        histories.extend(scan_files(root, "*_history.csv"))

    summary_rows = []
    for path in sorted(set(histories)):
        rows = read_csv(path)
        if not rows:
            continue
        dataset, model, loss_name = parse_identity(
            path,
            args.datasets,
            args.models,
            ["chamfer", "temporal_weighted_chamfer"],
        )
        if not dataset:
            first = rows[0]
            dataset = normalize_dataset(first.get("dataset", ""))
            model = first.get("model", "")
            loss_name = first.get("loss_name", "")
        if dataset not in args.datasets or model not in args.models:
            continue
        valid = [row for row in rows if safe_float(row.get("val_loss")) is not None]
        if not valid:
            continue
        best = min(valid, key=lambda row: safe_float(row.get("val_loss")))
        final = max(valid, key=lambda row: safe_int(row.get("epoch"), -1))
        time_weight = extract_time_weight(path, valid)
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "loss": loss_name,
                "loss_time_weight": format_number(time_weight),
                "best_epoch": safe_int(best.get("epoch")),
                "best_val_loss": format_number(best.get("val_loss")),
                "final_epoch": safe_int(final.get("epoch")),
                "final_val_loss": format_number(final.get("val_loss")),
                "history_path": compact_path(path),
            }
        )

    eval_roots = [Path(args.time_weight_eval_root), Path(args.root) / "time_weight_ablation_eval"]
    eval_files = []
    for root in eval_roots:
        eval_files.extend(scan_files(root, "*_corruptions.csv"))
    eval_rows = []
    for path in sorted(set(eval_files)):
        rows = read_csv(path)
        if not rows:
            continue
        time_weight = extract_time_weight(path, rows)
        for row in rows:
            if row.get("metric") != args.primary_metric:
                continue
            if safe_float(row.get("corruption_level")) != 0.0:
                continue
            if row.get("corruption") not in ("gaussian_noise", "clean", ""):
                continue
            eval_rows.append(
                {
                    "dataset": normalize_dataset(row.get("dataset", "")),
                    "model": row.get("model", ""),
                    "loss": row.get("trained_loss", ""),
                    "loss_time_weight": format_number(time_weight),
                    "metric": row.get("metric", ""),
                    "mean_reconstruction_value": row.get("reconstruction_value", ""),
                }
            )

    if eval_rows:
        grouped = defaultdict(list)
        for row in eval_rows:
            key = (row["dataset"], row["model"], row["loss"], row["loss_time_weight"], row["metric"])
            grouped[key].append(row["mean_reconstruction_value"])
        eval_summary = []
        for (dataset, model, loss_name, time_weight, metric), values in sorted(grouped.items()):
            value_stats = stats(values)
            eval_summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "loss": loss_name,
                    "loss_time_weight": time_weight,
                    "metric": metric,
                    "mean_reconstruction_value": format_number(value_stats["mean"]),
                    "num_rows": value_stats["n"],
                }
            )
        return summary_rows, eval_summary
    return summary_rows, []


def normalized_ranks(rows, key_fields, value_field):
    values = defaultdict(list)
    for row in rows:
        group_key = tuple(row[field] for field in key_fields)
        value = safe_float(row.get(value_field))
        if value is not None:
            values[group_key].append(value)
    means = {key: mean(vals) for key, vals in values.items() if vals}
    if not means:
        return {}
    min_value = min(means.values())
    max_value = max(means.values())
    if max_value == min_value:
        return {key: 0.0 for key in means}
    return {key: (value - min_value) / (max_value - min_value) for key, value in means.items()}


def build_ranking(clean_summary, corruption_summary, loss_benchmark_summary, primary_metric):
    key_fields = ["dataset", "model", "trained_loss"]
    clean_rows = [row for row in clean_summary if row["metric"] == primary_metric]
    robust_rows = [row for row in corruption_summary if row["metric"] == primary_metric]

    clean_scores = normalized_ranks(clean_rows, key_fields, "mean_reconstruction_value")
    robust_scores = normalized_ranks(robust_rows, key_fields, "absolute_degradation_vs_clean")

    cost_by_dataset_loss = normalized_ranks(
        loss_benchmark_summary,
        ["dataset", "loss"],
        "mean_seconds",
    )
    all_keys = set(clean_scores) | set(robust_scores)
    ranking = []
    for key in sorted(all_keys):
        dataset, model, loss_name = key
        cost_score = cost_by_dataset_loss.get((dataset, loss_name))
        components = [
            clean_scores.get(key),
            robust_scores.get(key),
            cost_score,
        ]
        present = [value for value in components if value is not None]
        if not present:
            continue
        score = mean(present)
        ranking.append(
            {
                "dataset": dataset,
                "model": model,
                "trained_loss": loss_name,
                "metric": primary_metric,
                "overall_score_lower_is_better": format_number(score),
                "quality_score": format_number(clean_scores.get(key)),
                "robustness_score": format_number(robust_scores.get(key)),
                "cost_score": format_number(cost_score),
                "available_components": len(present),
            }
        )
    ranking.sort(key=lambda row: safe_float(row["overall_score_lower_is_better"], 1e9))
    for idx, row in enumerate(ranking, start=1):
        row["rank"] = idx
    return ranking


def plot_bar(rows, output_path, x_field, y_field, title, ylabel, group_field=None):
    data = []
    if group_field:
        groups = sorted({row[group_field] for row in rows})
        labels = sorted({row[x_field] for row in rows})
        width = 0.8 / max(1, len(groups))
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
        positions = list(range(len(labels)))
        for group_idx, group in enumerate(groups):
            values = []
            for label in labels:
                candidates = [row for row in rows if row[x_field] == label and row[group_field] == group]
                values.append(safe_float(candidates[0].get(y_field), 0.0) if candidates else 0.0)
            offsets = [pos - 0.4 + width / 2 + group_idx * width for pos in positions]
            ax.bar(offsets, values, width=width, label=group)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
    else:
        for row in rows:
            value = safe_float(row.get(y_field))
            if value is not None:
                data.append((row[x_field], value))
        if not data:
            return
        labels, values = zip(*data)
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_convergence(convergence_summary, histories, output_dir):
    by_dataset_model = defaultdict(list)
    for identity, paths in histories.items():
        dataset, model, loss_name = identity
        by_dataset_model[(dataset, model)].append((loss_name, paths))

    for (dataset, model), loss_paths in sorted(by_dataset_model.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for loss_name, paths in sorted(loss_paths):
            rows = []
            for path in paths:
                rows.extend(read_csv(path))
            points = [
                (safe_int(row.get("epoch")), safe_float(row.get("val_loss")))
                for row in rows
                if safe_int(row.get("epoch")) is not None and safe_float(row.get("val_loss")) is not None
            ]
            if not points:
                continue
            points = sorted(points)
            ax.plot([p[0] for p in points], [p[1] for p in points], label=loss_name)
            plotted = True
        if plotted:
            ax.set_title(f"Convergenza validation - {dataset} - {model}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation loss")
            ax.legend()
            fig.tight_layout()
            path = Path(output_dir) / "convergence" / f"{dataset}_{model}_convergence.png"
            ensure_dir(path.parent)
            fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_benchmark_bars(loss_benchmark_summary, output_dir):
    for dataset in sorted({row["dataset"] for row in loss_benchmark_summary}):
        rows = [row for row in loss_benchmark_summary if row["dataset"] == dataset]
        clean_rows = [row for row in rows if row["protocol"] == "clean"]
        if not clean_rows:
            clean_rows = rows
        plot_bar(
            clean_rows,
            Path(output_dir) / "benchmarks" / f"{dataset}_loss_value.png",
            "loss",
            "mean_value",
            f"Loss media clean - {dataset}",
            "Mean loss value",
        )
        plot_bar(
            clean_rows,
            Path(output_dir) / "benchmarks" / f"{dataset}_runtime.png",
            "loss",
            "mean_seconds",
            f"Tempo medio loss - {dataset}",
            "Seconds",
        )
        plot_bar(
            clean_rows,
            Path(output_dir) / "benchmarks" / f"{dataset}_memory.png",
            "loss",
            "mean_peak_memory_mb",
            f"Memoria media loss - {dataset}",
            "Peak memory MB",
        )
        plot_bar(
            clean_rows,
            Path(output_dir) / "benchmarks" / f"{dataset}_gflops.png",
            "loss",
            "mean_estimated_gflops",
            f"FLOPs stimati - {dataset}",
            "GFLOPs",
        )


def plot_corruption_curves(corruption_summary, output_dir, primary_metric):
    filtered = [row for row in corruption_summary if row["metric"] == primary_metric]
    grouped = defaultdict(list)
    for row in filtered:
        grouped[(row["dataset"], row["model"], row["corruption"])].append(row)

    for (dataset, model, corruption), rows in sorted(grouped.items()):
        by_loss = defaultdict(list)
        for row in rows:
            by_loss[row["trained_loss"]].append(row)
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for loss_name, loss_rows in sorted(by_loss.items()):
            points = [
                (safe_float(row.get("corruption_level")), safe_float(row.get("mean_reconstruction_value")))
                for row in loss_rows
            ]
            points = sorted((x, y) for x, y in points if x is not None and y is not None)
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=loss_name)
            plotted = True
        if plotted:
            ax.set_title(f"{primary_metric} sotto {corruption} - {dataset} - {model}")
            ax.set_xlabel(corruption)
            ax.set_ylabel(primary_metric)
            ax.legend()
            fig.tight_layout()
            path = Path(output_dir) / "robustness" / f"{dataset}_{model}_{corruption}_{primary_metric}.png"
            ensure_dir(path.parent)
            fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_heatmap(clean_summary, output_dir, primary_metric):
    rows = [row for row in clean_summary if row["metric"] == primary_metric]
    if not rows:
        return
    datasets = sorted({row["dataset"] for row in rows})
    losses = sorted({row["trained_loss"] for row in rows})
    matrix = []
    for dataset in datasets:
        matrix_row = []
        for loss_name in losses:
            values = [
                safe_float(row.get("mean_reconstruction_value"))
                for row in rows
                if row["dataset"] == dataset and row["trained_loss"] == loss_name
            ]
            values = [value for value in values if value is not None]
            matrix_row.append(mean(values) if values else float("nan"))
        matrix.append(matrix_row)

    fig, ax = plt.subplots(figsize=(max(7, len(losses) * 1.2), max(4, len(datasets) * 0.8)))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(losses)))
    ax.set_xticklabels(losses, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title(f"Heatmap clean reconstruction - {primary_metric}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = Path(output_dir) / "heatmaps" / f"dataset_loss_{primary_metric}.png"
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_model_comparison(model_summary, output_dir):
    for dataset in sorted({row["dataset"] for row in model_summary}):
        rows = [row for row in model_summary if row["dataset"] == dataset]
        plot_bar(
            rows,
            Path(output_dir) / "models" / f"{dataset}_model_clean_comparison.png",
            "model",
            "mean_clean_reconstruction",
            f"Confronto modelli clean - {dataset}",
            "Mean clean reconstruction",
        )
        plot_bar(
            rows,
            Path(output_dir) / "models" / f"{dataset}_model_robustness_comparison.png",
            "model",
            "mean_corruption_degradation",
            f"Degrado medio sotto corruzione - {dataset}",
            "Mean degradation",
        )


def plot_time_weight(time_weight_summary, time_weight_eval_summary, output_dir):
    rows = time_weight_eval_summary or time_weight_summary
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        if row.get("loss") not in ("temporal_weighted_chamfer", "chamfer"):
            continue
        grouped[(row.get("dataset"), row.get("model"))].append(row)

    for (dataset, model), group_rows in sorted(grouped.items()):
        points = []
        for row in group_rows:
            weight = safe_float(row.get("loss_time_weight"))
            if weight is None:
                continue
            value = safe_float(row.get("mean_reconstruction_value"))
            if value is None:
                value = safe_float(row.get("best_val_loss"))
            if value is not None:
                points.append((weight, value))
        if not points:
            continue
        points = sorted(points)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o")
        ax.set_title(f"Ablation peso temporale - {dataset} - {model}")
        ax.set_xlabel("loss_time_weight")
        ax.set_ylabel("metric value")
        fig.tight_layout()
        path = Path(output_dir) / "time_weight" / f"{dataset}_{model}_time_weight_ablation.png"
        ensure_dir(path.parent)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def discover_visual_panels(args):
    rows = []
    for root in args.visual_root:
        for path in scan_files(root, "*.png"):
            dataset, model, loss_name = parse_identity(
                path,
                args.datasets,
                args.models,
                expected_losses(args),
            )
            rows.append(
                {
                    "dataset": dataset or "",
                    "model": model or "",
                    "loss": loss_name or "",
                    "image_path": compact_path(path),
                }
            )
    return rows


def best_row(rows, value_field, lower_is_better=True):
    candidates = [(safe_float(row.get(value_field)), row) for row in rows]
    candidates = [(value, row) for value, row in candidates if value is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=not lower_is_better)[0][1]


def write_report_notes(
    path,
    coverage,
    loss_benchmark_summary,
    convergence_summary,
    clean_summary,
    corruption_summary,
    model_summary,
    ranking,
    time_weight_summary,
    visual_panels,
    args,
):
    complete = sum(1 for row in coverage if row["status"] == "complete")
    total = len(coverage)
    missing = total - complete
    best_clean = best_row(
        [row for row in clean_summary if row["metric"] == args.primary_metric],
        "mean_reconstruction_value",
    )
    fastest_loss = best_row(loss_benchmark_summary, "mean_seconds")
    best_convergence = best_row(convergence_summary, "best_val_loss")
    best_model = best_row(model_summary, "mean_clean_reconstruction")
    best_rank = ranking[0] if ranking else None

    lines = [
        "# Note finali per relazione",
        "",
        "## Protocollo",
        "",
        (
            "Analisi principale su point cloud neuromorfiche in modalita windowed4096: "
            "finestre da 4096 eventi, stride 4096, 4096 punti, input [x, y, t] e ordine temporale preservato."
        ),
        "",
        "## Copertura",
        "",
        f"- Combinazioni complete: {complete}/{total}.",
        f"- Combinazioni mancanti o parziali: {missing}.",
        "",
        "## Risultati principali",
        "",
    ]
    if best_clean:
        lines.append(
            "- Migliore ricostruzione clean secondo "
            f"{args.primary_metric}: {best_clean['dataset']} / {best_clean['model']} / "
            f"{best_clean['trained_loss']} con valore medio {best_clean['mean_reconstruction_value']}."
        )
    if fastest_loss:
        lines.append(
            "- Loss piu veloce nel benchmark clean: "
            f"{fastest_loss['dataset']} / {fastest_loss['loss']} con "
            f"{fastest_loss['mean_seconds']} secondi medi."
        )
    if best_convergence:
        lines.append(
            "- Migliore convergenza osservata: "
            f"{best_convergence['dataset']} / {best_convergence['model']} / {best_convergence['loss']} "
            f"alla epoca {best_convergence['best_epoch']} con val loss {best_convergence['best_val_loss']}."
        )
    if best_model:
        lines.append(
            "- Modello mediamente migliore sul clean: "
            f"{best_model['dataset']} / {best_model['model']} con valore medio "
            f"{best_model['mean_clean_reconstruction']}."
        )
    if best_rank:
        lines.append(
            "- Ranking complessivo migliore: "
            f"{best_rank['dataset']} / {best_rank['model']} / {best_rank['trained_loss']} "
            f"(score {best_rank['overall_score_lower_is_better']})."
        )

    lines.extend(
        [
            "",
            "## Lettura consigliata",
            "",
            "- Usare `coverage_matrix.md` per dichiarare in modo trasparente quali combinazioni sono complete.",
            "- Usare `loss_benchmark_summary.md` per discutere costo computazionale, memoria e FLOPs stimati.",
            "- Usare `convergence_summary.md` per confrontare velocita e stabilita di addestramento.",
            "- Usare `reconstruction_clean_summary.md` e `corruption_robustness_summary.md` per la qualita finale.",
            "- Usare i plot in `plots/robustness` per mostrare la sensibilita a noise, shuffle temporale e drop.",
            "",
            "## Materiale qualitativo",
            "",
            f"- Pannelli visuali trovati: {len(visual_panels)}.",
        ]
    )
    if time_weight_summary:
        lines.append("- Ablation temporale disponibile: usare `time_weight_ablation_summary.md`.")
    else:
        lines.append("- Ablation temporale non trovata nei percorsi analizzati.")

    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_pair(output_dir, name, rows):
    csv_path = Path(output_dir) / f"{name}.csv"
    md_path = Path(output_dir) / f"{name}.md"
    write_csv(csv_path, rows)
    write_markdown_table(md_path, rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)
    ensure_dir(args.root)

    histories, checkpoints, epoch_checkpoints = discover_training_artifacts(args)
    evals = discover_eval_files(args)
    benchmark_rows = load_benchmark_rows(args)
    eval_rows = load_eval_rows(args)

    coverage = build_coverage(args, histories, checkpoints, epoch_checkpoints, evals, benchmark_rows)
    convergence_summary = summarize_histories(args, histories)
    loss_benchmark_summary = summarize_loss_benchmarks(benchmark_rows)
    clean_summary = summarize_eval_clean(eval_rows)
    corruption_summary = summarize_corruptions(eval_rows, clean_summary)
    model_summary = summarize_models(clean_summary, corruption_summary, args.primary_metric)
    time_weight_summary, time_weight_eval_summary = summarize_time_weight(args)
    ranking = build_ranking(clean_summary, corruption_summary, loss_benchmark_summary, args.primary_metric)
    visual_panels = discover_visual_panels(args)

    write_table_pair(tables_dir, "coverage_matrix", coverage)
    write_table_pair(tables_dir, "loss_benchmark_summary", loss_benchmark_summary)
    write_table_pair(tables_dir, "convergence_summary", convergence_summary)
    write_table_pair(tables_dir, "reconstruction_clean_summary", clean_summary)
    write_table_pair(tables_dir, "corruption_robustness_summary", corruption_summary)
    write_table_pair(tables_dir, "model_comparison_summary", model_summary)
    write_table_pair(tables_dir, "time_weight_ablation_summary", time_weight_summary)
    write_table_pair(tables_dir, "time_weight_ablation_eval_summary", time_weight_eval_summary)
    write_table_pair(tables_dir, "ranking_overall", ranking)
    write_table_pair(tables_dir, "visual_panels_manifest", visual_panels)

    plot_convergence(convergence_summary, histories, plots_dir)
    plot_benchmark_bars(loss_benchmark_summary, plots_dir)
    plot_corruption_curves(corruption_summary, plots_dir, args.primary_metric)
    plot_heatmap(clean_summary, plots_dir, args.primary_metric)
    plot_model_comparison(model_summary, plots_dir)
    plot_time_weight(time_weight_summary, time_weight_eval_summary, plots_dir)

    write_report_notes(
        output_dir / "final_report_notes.md",
        coverage,
        loss_benchmark_summary,
        convergence_summary,
        clean_summary,
        corruption_summary,
        model_summary,
        ranking,
        time_weight_summary,
        visual_panels,
        args,
    )

    missing = [row for row in coverage if row["status"] != "complete"]
    print(f"Wrote final project analysis to {output_dir}")
    print(f"Coverage complete: {len(coverage) - len(missing)}/{len(coverage)}")
    if missing:
        print(f"Missing or partial combinations: {len(missing)}")
    if args.strict and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
