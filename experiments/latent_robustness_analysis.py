"""Merge latent-robustness summaries and generate comparison figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("mean_d_encoder_cosine", r"$d_{enc}$"),
    ("mean_d_target_delta_cosine", r"$\Delta d_{target}$"),
    ("mean_d_end_to_end_cosine", r"$d_{AE}$"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-label", default="")
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def objective_label(row):
    loss = row["trained_loss"]
    if loss == "temporal_weighted_chamfer":
        weight = float(row.get("trained_loss_time_weight") or 1.0)
        return f"TW-{weight:g}"
    return {
        "chamfer": "Chamfer",
        "density_aware_chamfer": "DCD",
        "hausdorff": "Hausdorff",
        "sinkhorn": "Sinkhorn",
    }.get(loss, loss)


def slug(value):
    return value.lower().replace("++", "pp").replace("+", "p").replace(" ", "_")


def make_plots(rows, output_dir, protocol_label):
    clean = {}
    grouped = defaultdict(list)
    for row in rows:
        identity = (row["dataset"], row["model"], objective_label(row))
        if row["corruption"] == "clean":
            clean[identity] = row
        else:
            grouped[(row["dataset"], row["model"], row["corruption"])].append(row)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, model, corruption), group in sorted(grouped.items()):
        by_objective = defaultdict(list)
        for row in group:
            by_objective[objective_label(row)].append(row)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for axis, (metric, ylabel) in zip(axes, METRICS):
            for objective, objective_rows in sorted(by_objective.items()):
                points = []
                clean_row = clean.get((dataset, model, objective))
                if clean_row is not None:
                    points.append((0.0, float(clean_row[metric])))
                points.extend(
                    (float(row["corruption_level"]), float(row[metric]))
                    for row in objective_rows
                )
                points.sort()
                axis.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    marker="o", linewidth=1.7, label=objective,
                )
            axis.set_xlabel(corruption.replace("_", " "))
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize=8)
        title = f"{dataset} | {model} | {corruption.replace('_', ' ')}"
        if protocol_label:
            title += f" | {protocol_label}"
        fig.suptitle(title)
        fig.tight_layout()
        path = plot_dir / f"{slug(dataset)}_{slug(model)}_{slug(corruption)}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    summary_paths = sorted(root.rglob("*_latent_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No *_latent_summary.csv files found below {root}")

    rows = []
    for path in summary_paths:
        for row in read_csv(path):
            row["source_file"] = str(path)
            row["protocol"] = args.protocol_label
            rows.append(row)
    write_csv(output_dir / "latent_robustness_merged.csv", rows)

    diagnostics = []
    for path in sorted(root.rglob("*_latent_diagnostics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["source_file"] = str(path)
        record["protocol"] = args.protocol_label
        diagnostics.append(record)
    write_csv(output_dir / "latent_diagnostics_merged.csv", diagnostics)
    make_plots(rows, output_dir, args.protocol_label)
    print(f"Merged {len(summary_paths)} checkpoint summaries into {output_dir}")


if __name__ == "__main__":
    main()
