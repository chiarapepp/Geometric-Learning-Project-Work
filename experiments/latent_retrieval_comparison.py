"""Compare what several latent-retrieval runs retrieve for the *same* query.

The script has two deliberately separate stages:

1. it reads existing ``*_latent.csv`` files and selects corruption/query tuples
   for which at least one configuration retrieves the exact clean instance and
   at least one retrieves another gallery item;
2. with ``--render`` it reloads only those checkpoints and writes one figure
   per selected case.  Each row is a configuration and every row shares the
   same clean target: ``reconstruction | true clean | Top-1 | Top-2 | Top-3``.

Selection needs no checkpoints, so it can be run locally on copied Slurm
outputs.  Rendering needs the checkpoints and datasets, and is normally run on
the cluster (it performs inference only; it never trains a model).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from experiments.latent_robustness_eval import (
    build_model, config_from_checkpoint, corruption_specs,
    deterministic_rng, encode, gallery_indices, load_fixed_batches,
    make_loader, metadata_label, perturb_points, perturbation_seed,
    plot_cloud, prepare_clean_cache, set_seed,
)
from src.datasets.evaluation_sampling import make_evaluation_loader


KEY_COLUMNS = ("dataset", "split", "sample_index", "corruption", "corruption_level", "repeat")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Directory containing latent retrieval run folders.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=["dvsgesture", "nmnist", "ncaltech101"], default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional model names to compare, e.g. pointnet_ae.")
    parser.add_argument("--losses", nargs="+", default=None,
                        help="Optional loss names to compare, e.g. chamfer hausdorff.")
    parser.add_argument("--save-to", default=None, help="Optional local dataset root override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=1, help="Contrast cases selected per dataset.")
    parser.add_argument("--render", action="store_true", help="Reload checkpoints and render the selected cases.")
    parser.add_argument("--checkpoint-root", default=None,
                        help="Optional directory searched recursively by checkpoint filename when CSV paths are remote.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gallery-size", type=int, default=400)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument(
        "--eval-subset-from", default=None,
        help="Balanced-subset manifest used by latent_robustness_eval; render one dataset per invocation.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--encoder-seed", type=int, default=1729)
    # These defaults mirror latent_robustness_eval.py and identify spec_idx.
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--temporal-shuffle-fractions", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    return parser.parse_args()


def config_label(row):
    loss = str(row["trained_loss"])
    if row.get("trained_loss_time_weight") not in (None, "", "0", "0.0"):
        loss += f" (tw={float(row['trained_loss_time_weight']):g})"
    return f"{row['model']} | {loss}"


def float_key(value):
    return round(float(value), 10)


def case_key(row):
    return tuple(float_key(row[column]) if column == "corruption_level" else str(row[column])
                 for column in KEY_COLUMNS)


def read_runs(root, dataset=None, models=None, losses=None):
    runs = []
    for path in sorted(Path(root).rglob("*_latent.csv")):
        if path.name.endswith("_summary.csv"):
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or (dataset and rows[0]["dataset"] != dataset):
            continue
        if models and rows[0]["model"] not in models:
            continue
        if losses and rows[0]["trained_loss"] not in losses:
            continue
        for row in rows:
            row["_source_csv"] = str(path)
            row["_label"] = config_label(row)
        runs.append((path, rows))
    if not runs:
        raise FileNotFoundError(f"No matching *_latent.csv files under {root}")
    return runs


def select_cases(runs, max_cases):
    grouped = defaultdict(dict)
    for _, rows in runs:
        label = rows[0]["_label"]
        for row in rows:
            if row["corruption"] == "clean":
                continue
            grouped[(row["dataset"], case_key(row))][label] = row

    candidates = defaultdict(list)
    for (dataset, key), by_config in grouped.items():
        if len(by_config) < 2:
            continue
        exact = [int(row["exact_top1"]) for row in by_config.values()]
        if min(exact) == max(exact):
            continue
        margins = [float(row["retrieval_margin"]) for row in by_config.values()]
        # First prefer unanimous contrast (some correct, some wrong), then a
        # visually decisive separation in the nearest-impostor margin.
        score = (sum(exact) * (len(exact) - sum(exact)), max(margins) - min(margins), max(abs(v) for v in margins))
        candidates[dataset].append((score, key, by_config))

    selected = []
    for dataset, values in candidates.items():
        for score, key, by_config in sorted(values, reverse=True)[:max_cases]:
            selected.append({
                "dataset": dataset,
                "key": dict(zip(KEY_COLUMNS, key)),
                "score": score,
                "configs": {label: row for label, row in sorted(by_config.items())},
            })
    return selected


def write_selection(cases, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_selection.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    fields = ["case_id", "config", *KEY_COLUMNS, "exact_top1", "retrieval_margin", "d_true_cosine",
              "target_gallery_index", "top1_gallery_index", "top1_distance", "top1_class",
              "top1_original_sample_id", "top1_window_id", "_source_csv", "checkpoint"]
    with (output_dir / "comparison_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id, case in enumerate(cases, 1):
            for label, row in case["configs"].items():
                values = {key: row.get(key, "") for key in fields if key not in {"case_id", "config"}}
                writer.writerow({"case_id": case_id, "config": label, **values})


def resolve_checkpoint(recorded_path, checkpoint_root):
    path = Path(recorded_path)
    if path.exists():
        return path
    if checkpoint_root:
        matches = list(Path(checkpoint_root).rglob(path.name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Preserve as much of the recorded directory suffix as possible
            # (e.g. outputs/final_windowed4096/<run>/<checkpoint>).
            recorded_parts = path.parts
            ranked = []
            for match in matches:
                local_parts = match.parts
                common_suffix = 0
                for recorded_part, local_part in zip(reversed(recorded_parts), reversed(local_parts)):
                    if recorded_part != local_part:
                        break
                    common_suffix += 1
                ranked.append((common_suffix, match))
            best_suffix = max(score for score, _ in ranked)
            best_matches = [match for score, match in ranked if score == best_suffix]
            if len(best_matches) == 1:
                return best_matches[0]
            # Time-weight ablations deliberately reuse the same checkpoint
            # filename in distinct tw2/tw5 run directories. Prefer a local
            # match whose parent run name agrees with the recorded path.
            same_run = [match for match in best_matches if match.parent.name == path.parent.name]
            if len(same_run) == 1:
                return same_run[0]
            raise RuntimeError(f"Checkpoint filename is ambiguous: {path.name}: {matches}")
    raise FileNotFoundError(f"Checkpoint unavailable: {recorded_path}. Pass --checkpoint-root on the cluster.")


def spec_index(row, args):
    for index, spec in enumerate(corruption_specs(args)):
        if (spec["corruption"] == row["corruption"] and
                math.isclose(float(spec["corruption_level"]), float(row["corruption_level"]))):
            return index, spec
    raise ValueError(f"Cannot find corruption spec for {row['corruption']}={row['corruption_level']}")


def prepare_run(row, args):
    checkpoint_path = resolve_checkpoint(row["checkpoint"], args.checkpoint_root)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    cfg = config_from_checkpoint(checkpoint, args)
    model, model_type = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device).eval()
    base_loader = make_loader(
        dataset_name=cfg.dataset, save_to=args.save_to or cfg.save_to, split=row["split"], num_points=cfg.num_points,
        input_dim=cfg.input_dim, batch_size=args.batch_size or cfg.test_batch_size,
        num_workers=cfg.num_workers if args.num_workers is None else args.num_workers,
        temporal_weight=cfg.temporal_weight, sample_mode=cfg.sample_mode, pad_mode=cfg.pad_mode,
        shuffle_points=cfg.shuffle_points, split_ratio=cfg.split_ratio, split_seed=cfg.split_seed,
        stream_mode=cfg.stream_mode, window_size=cfg.window_size, window_stride=cfg.window_stride,
        window_drop_last=cfg.window_drop_last, max_windows_per_sample=cfg.max_windows_per_sample,
    )
    if args.eval_subset_from:
        fixed_loader, _ = make_evaluation_loader(
            base_loader.dataset, dataset_name=cfg.dataset,
            batch_size=cfg.test_batch_size, num_workers=cfg.num_workers,
            subset_from=args.eval_subset_from,
        )
        batches = load_fixed_batches(fixed_loader, len(fixed_loader), cfg.dataset)
    else:
        fixed_loader = DataLoader(base_loader.dataset, batch_size=cfg.test_batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
        batches = load_fixed_batches(fixed_loader, args.max_batches, cfg.dataset)
    with torch.no_grad():
        cache = prepare_clean_cache(model, model_type, batches, args.device, args.encoder_seed)
    all_points = torch.cat([item["clean_points"] for item in cache], dim=0)
    selected = sorted(gallery_indices(cache, args.gallery_size, args.seed))
    return model, model_type, cfg, batches, all_points, selected


def reconstruct(row, run, args):
    model, model_type, cfg, batches, all_points, selected = run
    sample_index = int(row["sample_index"])
    recorded_batch = int(row["batch"])
    # The CSV stores the batch index used during evaluation but not its batch
    # size. Recover it from the global sample index (all v3 runs used a fixed
    # batch size of four) so rendering remains faithful even when a larger
    # loading batch is requested locally for speed.
    recorded_batch_size = sample_index // recorded_batch if recorded_batch > 0 else 4
    recorded_batch_size = max(recorded_batch_size, 1)
    start = recorded_batch * recorded_batch_size
    end = min(start + recorded_batch_size, len(all_points))
    if not start <= sample_index < end:
        raise IndexError(f"Cannot reconstruct recorded batch for sample {sample_index}, batch {recorded_batch}")
    local_idx = sample_index - start
    clean = all_points[start:end].to(args.device)
    spec_idx, spec = spec_index(row, args)
    seed = perturbation_seed(args.seed, spec_idx, int(row["repeat"]), recorded_batch)
    with torch.no_grad(), deterministic_rng(seed, args.device):
        corrupted = perturb_points(clean, noise_std=spec["noise_std"],
                                   temporal_shuffle_fraction=spec["temporal_shuffle_fraction"],
                                   drop_fraction=spec["drop_fraction"])
        z = encode(model, model_type, corrupted, args.encoder_seed + recorded_batch, args.device)
        reconstruction = model.decode(z)[local_idx].detach().cpu()
    gallery_points = [all_points[selected[int(row[f"top{rank}_gallery_index"])]] for rank in range(1, 4)]
    return reconstruction, all_points[sample_index], gallery_points


def panel_metadata(row, rank=None):
    if rank is None:
        return {key: row[key] for key in ("class", "original_sample_id", "window_id", "window_start", "window_end")}
    return {
        "class": row[f"top{rank}_class"], "original_sample_id": row[f"top{rank}_original_sample_id"],
        "window_id": row[f"top{rank}_window_id"], "window_start": 0, "window_end": None,
    }


def render_case(case_id, case, args, output_dir, prepared=None):
    config_items = list(case["configs"].items())
    prepared = {} if prepared is None else prepared
    fig, axes = plt.subplots(len(config_items), 5, figsize=(15, 3.85 * len(config_items)), squeeze=False)
    for row_index, (label, row) in enumerate(config_items):
        if label not in prepared:
            prepared[label] = prepare_run(row, args)
        reconstruction, true_points, retrieved = reconstruct(row, prepared[label], args)
        panels = [
            ("Query reconstruction", reconstruction, panel_metadata(row), None, None),
            (f"True clean | gallery #{row['target_gallery_index']}", true_points, panel_metadata(row),
             float(row["d_true_cosine"]), "#2e7d32"),
        ]
        for rank in range(1, 4):
            panels.append((f"Top-{rank} | gallery #{row[f'top{rank}_gallery_index']}", retrieved[rank - 1],
                           panel_metadata(row, rank), float(row[f"top{rank}_distance"]),
                           "#1565c0" if rank == 1 else None))
        for axis, (title, points, metadata, distance, highlight) in zip(axes[row_index], panels):
            plot_cloud(axis, points, f"{title}\n{metadata_label(metadata)}", distance, highlight)
        axes[row_index][0].set_ylabel(
            f"{label}\nexact Top-1={int(row['exact_top1'])}\nmargin={float(row['retrieval_margin']):.4f}", fontsize=8
        )
    first = next(iter(case["configs"].values()))
    fig.suptitle(
        f"Same query and clean target across configurations | {first['dataset']} | "
        f"{first['corruption']}={float(first['corruption_level']):g}\n"
        "green=true target; blue=Top-1", fontsize=13
    )
    # Reserve a dedicated header band and generous row spacing: the panel
    # titles and the multi-line configuration labels must never overlap.
    fig.tight_layout(rect=(0.075, 0.015, 0.995, 0.965), h_pad=3.4, w_pad=1.4)
    path = output_dir / f"retrieval_comparison_case{case_id:02d}_{first['dataset']}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    if args.max_cases < 1:
        raise ValueError("--max-cases must be at least 1")
    set_seed(args.seed)
    runs = read_runs(args.runs_root, args.dataset, args.models, args.losses)
    cases = select_cases(runs, args.max_cases)
    output_dir = Path(args.output_dir)
    write_selection(cases, output_dir)
    print(f"Selected {len(cases)} contrast cases; wrote {output_dir / 'comparison_selection.csv'}")
    if not cases:
        return
    if args.render:
        prepared = {}
        for case_id, case in enumerate(cases, 1):
            path = render_case(case_id, case, args, output_dir, prepared)
            print(f"Wrote {path}")
    else:
        print("Selection only. Add --render where checkpoints and datasets are available to create PNG figures.")


if __name__ == "__main__":
    main()
