"""Evaluate latent-space robustness of a trained point-cloud autoencoder.

The script measures three complementary cosine distances for each test sample:

* encoder: E(clean) versus E(corrupted)
* target: E(clean) versus E(AE(corrupted))
* end_to_end: E(AE(clean)) versus E(AE(corrupted))

It also reports the target-distance increase relative to the clean
reconstruction baseline.  Rows are stored per sample so that downstream
statistics do not accidentally give equal weight to unequal batches.

In addition, every reconstructed query is retrieved against a fixed clean
gallery.  The output separates latent fidelity (distance to its own clean
target) from identity preservation (exact-instance Top-1) and records the
nearest-impostor margin, same-recording, and same-class retrieval.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.evaluate import make_loader, perturb_points, unpack_points, write_csv
from src.datasets.evaluation_sampling import IndexedSubsetDataset, make_evaluation_loader, write_csv_rows
from src.train_ae import Config, build_model
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None, choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument(
        "--temporal-shuffle-fractions", nargs="+", type=float,
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Independent realizations for every non-clean corruption level.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument(
        "--eval-sampling", choices=["ordered", "random", "stratified"], default="ordered",
        help="How to choose the evaluation subset. 'stratified' balances classes and recordings.",
    )
    parser.add_argument(
        "--eval-items-per-class", type=int, default=None,
        help="Required for --eval-sampling stratified; target number of windows/items per class.",
    )
    parser.add_argument("--eval-max-windows-per-source", type=int, default=1)
    parser.add_argument("--eval-sampling-seed", type=int, default=13)
    parser.add_argument(
        "--eval-subset-from", default=None,
        help="CSV manifest made by inspect_dataset_composition; reuses exactly those dataset indices.",
    )
    parser.add_argument(
        "--eval-subset-manifest", default=None,
        help="Where to save the selected subset manifest (defaults beside --output).",
    )
    parser.add_argument(
        "--gallery-size", type=int, default=400,
        help="Number of clean samples used both as the retrieval gallery and queries. "
             "Use 0 for every sample loaded by --max-batches.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval-output-dir", default=None,
        help="Directory for gallery and representative-retrieval figures.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/latent_robustness.csv")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--encoder-seed", type=int, default=1729,
        help="Fixed RNG seed for deterministic PointNet++ FPS during each encoding.",
    )
    return parser.parse_args()


def config_from_checkpoint(checkpoint, args):
    checkpoint_config = checkpoint.get("config", {})
    values = {field.name: getattr(Config, field.name) for field in fields(Config)}
    values.update({key: value for key, value in checkpoint_config.items() if key in values})
    if args.dataset is not None:
        values["dataset"] = args.dataset
    if args.save_to is not None:
        values["save_to"] = args.save_to
    values["device"] = args.device
    if args.batch_size is not None:
        values["test_batch_size"] = args.batch_size
    if args.num_workers is not None:
        values["num_workers"] = args.num_workers
    values["seed"] = args.seed
    values["wandb"] = "disabled"
    return Config(**values)


@contextmanager
def deterministic_rng(seed, device):
    """Fork CPU/CUDA RNG state so model-side sampling is reproducible."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def encode(model, model_type, points, seed, device):
    with deterministic_rng(seed, device):
        encoded = model.encode(points)
    # For a VAE, use mu rather than a stochastic latent sample.
    return encoded[0] if model_type == "vae" else encoded


def cosine_distance(left, right):
    return 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)


def corruption_specs(args):
    specs = [{
        "corruption": "clean", "corruption_level": 0.0,
        "noise_std": 0.0, "temporal_shuffle_fraction": 0.0, "drop_fraction": 0.0,
    }]
    for level in args.noise_stds:
        if level > 0:
            specs.append({
                "corruption": "gaussian_noise", "corruption_level": level,
                "noise_std": level, "temporal_shuffle_fraction": 0.0, "drop_fraction": 0.0,
            })
    for level in args.temporal_shuffle_fractions:
        if level > 0:
            specs.append({
                "corruption": "temporal_shuffle", "corruption_level": level,
                "noise_std": 0.0, "temporal_shuffle_fraction": level, "drop_fraction": 0.0,
            })
    for level in args.drop_fractions:
        if level > 0:
            specs.append({
                "corruption": "random_drop", "corruption_level": level,
                "noise_std": 0.0, "temporal_shuffle_fraction": 0.0, "drop_fraction": level,
            })
    return specs


def metadata_for_dataset(dataset, dataset_name, index, target):
    """Return stable, human-readable provenance without changing dataset APIs."""
    record = {
        "dataset": dataset_name,
        "class": int(target) if torch.is_tensor(target) and target.ndim == 0 else target,
        "original_sample_id": int(index),
        "window_id": 0,
        "window_start": 0,
        "window_end": None,
    }
    # WindowedEventDataset keeps exactly the provenance required for retrieval.
    if hasattr(dataset, "windows"):
        original_id, start, end = dataset.windows[index]
        record.update({
            "original_sample_id": int(original_id),
            "window_id": sum(1 for sample, window_start, _ in dataset.windows[:index]
                             if sample == original_id and window_start < start),
            "window_start": int(start), "window_end": int(end),
        })
        base = dataset.dataset
    else:
        base = dataset
    # N-Caltech's local index differs from the source-dataset index.
    if hasattr(base, "indices"):
        record["original_sample_id"] = int(base.indices[record["original_sample_id"]])
    return record


def load_fixed_batches(loader, max_batches, dataset_name):
    batches = []
    source_dataset = loader.dataset.dataset if isinstance(loader.dataset, IndexedSubsetDataset) else loader.dataset
    global_index = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="Loading fixed test subset")):
        if batch_idx >= max_batches:
            break
        points = unpack_points(batch).cpu()
        targets = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else [None] * len(points)
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().tolist()
        if isinstance(batch, (tuple, list)) and len(batch) > 2:
            source_indices = batch[2]
            source_indices = source_indices.detach().cpu().tolist() if torch.is_tensor(source_indices) else list(source_indices)
        else:
            source_indices = list(range(global_index, global_index + len(points)))
        metadata = [metadata_for_dataset(source_dataset, dataset_name, index, target)
                    for index, target in zip(source_indices, targets)]
        batches.append({"points": points, "metadata": metadata})
        global_index += len(points)
    if not batches:
        raise RuntimeError("The selected data loader produced no batches.")
    return batches


def prepare_clean_cache(model, model_type, batches, device, encoder_seed):
    cache = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="Encoding clean references")):
        clean_cpu = batch["points"]
        clean = clean_cpu.to(device)
        seed = encoder_seed + batch_idx
        z_clean = encode(model, model_type, clean, seed, device)
        clean_reconstruction = model.decode(z_clean)
        z_clean_reconstruction = encode(model, model_type, clean_reconstruction, seed, device)
        z_clean_alternate = encode(model, model_type, clean, seed + 1_000_000, device)
        cache.append({
            "z_clean": z_clean.detach(),
            "z_clean_reconstruction": z_clean_reconstruction.detach(),
            "clean_target_distance": cosine_distance(z_clean, z_clean_reconstruction).detach(),
            "fps_noise_floor": cosine_distance(z_clean, z_clean_alternate).detach(),
            "clean_points": clean_cpu,
            "metadata": batch["metadata"],
        })
    return cache


def perturbation_seed(base_seed, spec_idx, repeat, batch_idx):
    # Stable arithmetic mapping; independent of RNG consumed by model forwards.
    return base_seed + 10_000_000 * spec_idx + 100_000 * repeat + batch_idx


def gallery_indices(clean_cache, gallery_size, seed):
    total = sum(item["z_clean"].shape[0] for item in clean_cache)
    if gallery_size <= 0 or gallery_size >= total:
        return set(range(total))
    generator = torch.Generator().manual_seed(seed)
    return set(torch.randperm(total, generator=generator)[:gallery_size].tolist())


def retrieval_metrics(query, gallery, query_metadata, gallery_metadata, top_k):
    distances = 1.0 - F.normalize(query, dim=-1, eps=1e-8) @ F.normalize(gallery, dim=-1, eps=1e-8).T
    order = distances.argsort(dim=1)
    top = order[:, :min(top_k, gallery.shape[0])]
    values = distances.gather(1, top)
    rows = []
    for index in range(query.shape[0]):
        meta = query_metadata[index]
        exact = [pos for pos, other in enumerate(gallery_metadata)
                 if other["original_sample_id"] == meta["original_sample_id"]
                 and other["window_id"] == meta["window_id"]]
        if len(exact) != 1:
            raise RuntimeError("Every query must occur exactly once in the gallery.")
        target_pos = exact[0]
        d_true = float(distances[index, target_pos])
        impostor = distances[index].clone()
        impostor[target_pos] = float("inf")
        d_wrong = float(impostor.min())
        retrieved = [int(item) for item in top[index].detach().cpu().tolist()]
        result = {
            "d_true_cosine": d_true, "d_nearest_wrong_cosine": d_wrong,
            "retrieval_margin": d_wrong - d_true,
            "exact_top1": int(retrieved[0] == target_pos),
            "same_recording_top1": int(gallery_metadata[retrieved[0]]["original_sample_id"] == meta["original_sample_id"]),
            "same_class_top1": int(gallery_metadata[retrieved[0]]["class"] == meta["class"]),
            "target_gallery_index": target_pos,
        }
        for rank, (pos, distance) in enumerate(zip(retrieved, values[index].detach().cpu().tolist()), 1):
            other = gallery_metadata[pos]
            result.update({
                f"top{rank}_gallery_index": pos, f"top{rank}_distance": float(distance),
                f"top{rank}_class": other["class"],
                f"top{rank}_original_sample_id": other["original_sample_id"],
                f"top{rank}_window_id": other["window_id"],
            })
        rows.append(result)
    return rows


def evaluate(model, model_type, batches, clean_cache, specs, args, cfg, selected_indices):
    rows = []
    examples = {"easy_correct": None, "hard_correct": None, "wrong_retrieval": None}
    sample_offsets = []
    offset = 0
    for batch in batches:
        sample_offsets.append(offset)
        offset += batch["points"].shape[0]

    gallery_embeddings = torch.cat([cache["z_clean"] for cache in clean_cache], dim=0)
    gallery_points_all = torch.cat([cache["clean_points"] for cache in clean_cache], dim=0)
    gallery_metadata_all = [meta for cache in clean_cache for meta in cache["metadata"]]
    selected = sorted(selected_indices)
    gallery_embeddings = gallery_embeddings[selected]
    gallery_metadata = [gallery_metadata_all[index] for index in selected]

    with torch.no_grad():
        for spec_idx, spec in enumerate(tqdm(specs, desc="Latent robustness")):
            repeat_count = 1 if spec["corruption"] == "clean" else args.repeats
            for repeat in range(repeat_count):
                for batch_idx, batch in enumerate(batches):
                    clean_cpu = batch["points"]
                    clean = clean_cpu.to(cfg.device)
                    perturb_seed = perturbation_seed(args.seed, spec_idx, repeat, batch_idx)
                    with deterministic_rng(perturb_seed, cfg.device):
                        corrupted = perturb_points(
                            clean,
                            noise_std=spec["noise_std"],
                            temporal_shuffle_fraction=spec["temporal_shuffle_fraction"],
                            drop_fraction=spec["drop_fraction"],
                        )

                    encode_seed = args.encoder_seed + batch_idx
                    z_corrupted = encode(model, model_type, corrupted, encode_seed, cfg.device)
                    corrupted_reconstruction = model.decode(z_corrupted)
                    z_corrupted_reconstruction = encode(
                        model, model_type, corrupted_reconstruction, encode_seed, cfg.device
                    )

                    reference = clean_cache[batch_idx]
                    d_encoder = cosine_distance(reference["z_clean"], z_corrupted)
                    d_target = cosine_distance(reference["z_clean"], z_corrupted_reconstruction)
                    d_end_to_end = cosine_distance(
                        reference["z_clean_reconstruction"], z_corrupted_reconstruction
                    )
                    d_target_delta = d_target - reference["clean_target_distance"]
                    clean_norm = reference["z_clean"].norm(dim=-1).clamp_min(1e-8)
                    corrupted_norm_ratio = z_corrupted.norm(dim=-1) / clean_norm
                    reconstructed_norm_ratio = z_corrupted_reconstruction.norm(dim=-1) / clean_norm
                    local_global = [sample_offsets[batch_idx] + i for i in range(clean.shape[0])]
                    keep = [i for i, index in enumerate(local_global) if index in selected_indices]
                    retrieval = retrieval_metrics(
                        z_corrupted_reconstruction[keep], gallery_embeddings,
                        [reference["metadata"][i] for i in keep], gallery_metadata,
                        args.retrieval_top_k,
                    ) if keep else []

                    tensors = [
                        d_encoder, d_target, d_target_delta, d_end_to_end,
                        reference["clean_target_distance"], reference["fps_noise_floor"],
                        corrupted_norm_ratio, reconstructed_norm_ratio,
                    ]
                    arrays = [tensor.detach().cpu().tolist() for tensor in tensors]
                    for metric_idx, local_idx in enumerate(keep):
                        row = {
                            "sample_index": sample_offsets[batch_idx] + local_idx,
                            "batch": batch_idx,
                            "repeat": repeat,
                            "dataset": cfg.dataset,
                            "split": args.split,
                            "model": cfg.model_name,
                            "trained_loss": cfg.loss_name,
                            "trained_loss_time_weight": cfg.loss_time_weight,
                            "checkpoint": args.checkpoint,
                            "corruption": spec["corruption"],
                            "corruption_level": spec["corruption_level"],
                            "noise_std": spec["noise_std"],
                            "temporal_shuffle_fraction": spec["temporal_shuffle_fraction"],
                            "drop_fraction": spec["drop_fraction"],
                            "d_encoder_cosine": arrays[0][local_idx],
                            "d_target_cosine": arrays[1][local_idx],
                            "d_target_delta_cosine": arrays[2][local_idx],
                            "d_end_to_end_cosine": arrays[3][local_idx],
                            "clean_target_baseline_cosine": arrays[4][local_idx],
                            "fps_noise_floor_cosine": arrays[5][local_idx],
                            "corrupted_latent_norm_ratio": arrays[6][local_idx],
                            "reconstructed_latent_norm_ratio": arrays[7][local_idx],
                        }
                        row.update(reference["metadata"][local_idx])
                        row.update(retrieval[metric_idx])
                        rows.append(row)
                        metric = row["retrieval_margin"]
                        category = None
                        if row["exact_top1"] and (examples["easy_correct"] is None or metric > examples["easy_correct"]["margin"]):
                            category = "easy_correct"
                        elif row["exact_top1"] and metric > 0 and (examples["hard_correct"] is None or metric < examples["hard_correct"]["margin"]):
                            category = "hard_correct"
                        elif not row["exact_top1"] and (examples["wrong_retrieval"] is None or metric < examples["wrong_retrieval"]["margin"]):
                            category = "wrong_retrieval"
                        if category:
                            retrieved = [row[f"top{rank}_gallery_index"] for rank in range(1, args.retrieval_top_k + 1)
                                         if f"top{rank}_gallery_index" in row]
                            examples[category] = {
                                "margin": metric, "row": row,
                                "query": corrupted_reconstruction[local_idx].detach().cpu(),
                                "true": clean_cpu[local_idx],
                                "retrieved": [gallery_points_all[selected[pos]]
                                              for pos in retrieved],
                                "retrieved_metadata": [gallery_metadata[pos] for pos in retrieved],
                            }
    return rows, examples


def summarize(rows):
    metrics = [
        "d_encoder_cosine", "d_target_cosine", "d_target_delta_cosine",
        "d_end_to_end_cosine", "clean_target_baseline_cosine",
        "fps_noise_floor_cosine", "corrupted_latent_norm_ratio",
        "reconstructed_latent_norm_ratio",
        "d_true_cosine", "d_nearest_wrong_cosine", "retrieval_margin",
        "exact_top1", "same_recording_top1", "same_class_top1",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["corruption"], float(row["corruption_level"]))].append(row)

    summary = []
    for (corruption, level), group in sorted(grouped.items()):
        # Average repeats per sample first, then summarize across samples.
        per_sample = defaultdict(lambda: defaultdict(list))
        for row in group:
            for metric in metrics:
                per_sample[row["sample_index"]][metric].append(float(row[metric]))
        sample_means = {
            metric: [statistics.fmean(values[metric]) for values in per_sample.values()]
            for metric in metrics
        }
        record = {
            "dataset": group[0]["dataset"],
            "model": group[0]["model"],
            "trained_loss": group[0]["trained_loss"],
            "trained_loss_time_weight": group[0]["trained_loss_time_weight"],
            "corruption": corruption,
            "corruption_level": level,
            "num_samples": len(per_sample),
            "repeats": max(len(values[metrics[0]]) for values in per_sample.values()),
        }
        for metric in metrics:
            values = sample_means[metric]
            record[f"mean_{metric}"] = statistics.fmean(values)
            record[f"median_{metric}"] = statistics.median(values)
            record[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary


def clean_diagnostics(clean_cache):
    embeddings = torch.cat([item["z_clean"].detach().cpu() for item in clean_cache], dim=0)
    normalized = F.normalize(embeddings, dim=-1, eps=1e-8)
    similarity = normalized @ normalized.T
    mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    pairwise_distances = (1.0 - similarity)[mask]
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    probabilities = variance / variance.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    fps_values = torch.cat([item["fps_noise_floor"].detach().cpu() for item in clean_cache])
    return {
        "num_samples": int(embeddings.shape[0]),
        "latent_dim": int(embeddings.shape[1]),
        "mean_clean_pairwise_cosine_distance": float(pairwise_distances.mean()),
        "median_clean_pairwise_cosine_distance": float(pairwise_distances.median()),
        "mean_feature_variance": float(centered.var(dim=0, unbiased=False).mean()),
        "effective_rank": float(effective_rank),
        "mean_fps_noise_floor_cosine": float(fps_values.mean()),
        "max_fps_noise_floor_cosine": float(fps_values.max()),
    }


def derived_path(output, suffix):
    path = Path(output)
    return path.with_name(f"{path.stem}{suffix}")


def metadata_label(metadata):
    """Compact provenance label shared by gallery and retrieval panels."""
    window = f"win={metadata['window_id']}"
    if metadata.get("window_end") is not None:
        window += f" [{metadata['window_start']}:{metadata['window_end']}]"
    return f"class={metadata['class']}\nsample={metadata['original_sample_id']}, {window}"


def plot_cloud(axis, points, title, distance=None, highlight=None):
    points = points.detach().cpu()
    axis.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1, cmap="viridis")
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title(title + (f"\nd={distance:.4f}" if distance is not None else ""), fontsize=8)
    if highlight is not None:
        for spine in axis.spines.values():
            spine.set_color(highlight)
            spine.set_linewidth(2.5)


def make_gallery_figure(clean_cache, selected, output_dir, max_items=25):
    points = torch.cat([item["clean_points"] for item in clean_cache], dim=0)
    metadata = [meta for item in clean_cache for meta in item["metadata"]]
    if not selected:
        return
    # Evenly distributed gallery positions avoid showing only adjacent windows.
    shown_positions = torch.linspace(0, len(selected) - 1, steps=min(max_items, len(selected))).round().long().tolist()
    shown = [(position, selected[position]) for position in shown_positions]
    cols = 5; rows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows), squeeze=False)
    for axis in axes.flat: axis.axis("off")
    for axis, (gallery_index, source_index) in zip(axes.flat, shown):
        meta = metadata[source_index]
        plot_cloud(axis, points[source_index], f"gallery #{gallery_index} | {metadata_label(meta)}")
    fig.suptitle("Clean retrieval gallery (x,y coloured by time)")
    fig.tight_layout()
    fig.savefig(output_dir / "gallery_overview_5x5.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def retrieval_panels(example):
    """Return five labelled panels in the order used by all qualitative plots."""
    row = example["row"]
    query_meta = {
        key: row[key] for key in ("class", "original_sample_id", "window_id", "window_start", "window_end")
    }
    panels = [
        ("Query reconstruction", example["query"], query_meta, None, None),
        (f"True clean | gallery #{row['target_gallery_index']}", example["true"], query_meta,
         row["d_true_cosine"], "#2e7d32"),
    ]
    for rank, (points, meta) in enumerate(zip(example["retrieved"], example["retrieved_metadata"]), 1):
        highlight = "#1565c0" if rank == 1 else None
        panels.append((f"Top-{rank} | gallery #{row[f'top{rank}_gallery_index']}", points, meta,
                       row.get(f"top{rank}_distance"), highlight))
    return panels


def draw_retrieval_row(axes, category, example):
    if example is None:
        for axis in axes:
            axis.axis("off")
        axes[0].set_title(f"{category.replace('_', ' ')} unavailable", fontsize=9)
        return
    row = example["row"]
    panels = retrieval_panels(example)
    for axis, (title, points, metadata, distance, highlight) in zip(axes, panels):
        plot_cloud(axis, points, f"{title}\n{metadata_label(metadata)}", distance, highlight)
    for axis in axes[len(panels):]:
        axis.axis("off")


def make_retrieval_examples(examples, output_dir):
    for category, example in examples.items():
        if example is None:
            continue
        row = example["row"]
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
        draw_retrieval_row(axes, category, example)
        fig.suptitle(
            f"{category.replace('_', ' ')} | {row['corruption']}={row['corruption_level']:g} | "
            f"margin={row['retrieval_margin']:.4f}\n"
            f"green=true target; blue=Top-1"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"retrieval_{category}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_retrieval_triptych(examples, output_dir):
    fig, axes = plt.subplots(3, 5, figsize=(15, 10.5))
    for row_axes, category in zip(axes, ("easy_correct", "hard_correct", "wrong_retrieval")):
        draw_retrieval_row(row_axes, category, examples[category])
        example = examples[category]
        if example is not None:
            row = example["row"]
            row_axes[0].set_ylabel(
                f"{category.replace('_', ' ')}\n{row['corruption']}={row['corruption_level']:g}\n"
                f"margin={row['retrieval_margin']:.4f}", fontsize=9
            )
    fig.suptitle("Latent retrieval qualitative cases | green=true target; blue=Top-1", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_triptych.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.retrieval_top_k < 1:
        raise ValueError("--retrieval-top-k must be at least 1")
    if args.gallery_size == 1:
        raise ValueError("--gallery-size must be 0 or at least 2 to define a nearest impostor.")
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = config_from_checkpoint(checkpoint, args)
    model, model_type = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()

    base_loader = make_loader(
        dataset_name=cfg.dataset, save_to=cfg.save_to, split=args.split,
        num_points=cfg.num_points, input_dim=cfg.input_dim,
        batch_size=cfg.test_batch_size, num_workers=cfg.num_workers,
        temporal_weight=cfg.temporal_weight, sample_mode=cfg.sample_mode,
        pad_mode=cfg.pad_mode, shuffle_points=cfg.shuffle_points,
        split_ratio=cfg.split_ratio, split_seed=cfg.split_seed,
        stream_mode=cfg.stream_mode, window_size=cfg.window_size,
        window_stride=cfg.window_stride, window_drop_last=cfg.window_drop_last,
        max_windows_per_sample=cfg.max_windows_per_sample,
    )
    loader, subset_rows = make_evaluation_loader(
        base_loader.dataset,
        dataset_name=cfg.dataset,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
        mode=args.eval_sampling,
        max_items=args.max_batches * cfg.test_batch_size,
        items_per_class=args.eval_items_per_class,
        max_windows_per_source=args.eval_max_windows_per_source,
        seed=args.eval_sampling_seed,
        subset_from=args.eval_subset_from,
    )
    subset_manifest = (
        Path(args.eval_subset_manifest) if args.eval_subset_manifest else
        Path(args.output).with_name(f"{Path(args.output).stem}_subset.csv")
    )
    write_csv_rows(subset_manifest, subset_rows)
    batches = load_fixed_batches(loader, len(loader), cfg.dataset)
    with torch.no_grad():
        clean_cache = prepare_clean_cache(
            model, model_type, batches, cfg.device, args.encoder_seed
        )
        selected = gallery_indices(clean_cache, args.gallery_size, args.seed)
        rows, examples = evaluate(
            model, model_type, batches, clean_cache, corruption_specs(args), args, cfg, selected
        )

    summary_output = args.summary_output or derived_path(args.output, "_summary.csv")
    diagnostics_output = args.diagnostics_output or derived_path(args.output, "_diagnostics.json")
    write_csv(Path(args.output), rows)
    print(f"Wrote {len(subset_rows)} selected evaluation items to {subset_manifest}")
    write_csv(Path(summary_output), summarize(rows))
    diagnostics = clean_diagnostics(clean_cache)
    diagnostics.update({
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "trained_loss": cfg.loss_name,
        "trained_loss_time_weight": cfg.loss_time_weight,
        "checkpoint": args.checkpoint,
    })
    diagnostics_path = Path(diagnostics_output)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    retrieval_output_dir = Path(args.retrieval_output_dir or derived_path(args.output, "_retrieval"))
    retrieval_output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata = [meta for cache in clean_cache for meta in cache["metadata"]]
    gallery_manifest = [
        {"gallery_index": gallery_index, "source_subset_index": source_index, **all_metadata[source_index]}
        for gallery_index, source_index in enumerate(sorted(selected))
    ]
    write_csv(retrieval_output_dir / "gallery_metadata.csv", gallery_manifest)
    make_gallery_figure(clean_cache, sorted(selected), retrieval_output_dir)
    make_retrieval_examples(examples, retrieval_output_dir)
    make_retrieval_triptych(examples, retrieval_output_dir)
    print(f"Wrote {len(rows)} per-sample rows to {args.output}")
    print(f"Wrote aggregate results to {summary_output}")
    print(f"Wrote clean-space diagnostics to {diagnostics_output}")
    print(f"Wrote retrieval gallery, metadata, examples, and triptych to {retrieval_output_dir}")


if __name__ == "__main__":
    main()
