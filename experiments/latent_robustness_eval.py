"""Evaluate latent-space robustness of a trained point-cloud autoencoder.

The script measures three complementary cosine distances for each test sample:

* encoder: E(clean) versus E(corrupted)
* target: E(clean) versus E(AE(corrupted))
* end_to_end: E(AE(clean)) versus E(AE(corrupted))

It also reports the target-distance increase relative to the clean
reconstruction baseline.  Rows are stored per sample so that downstream
statistics do not accidentally give equal weight to unequal batches.
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

from src.evaluate import make_loader, perturb_points, unpack_points, write_csv
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


def load_fixed_batches(loader, max_batches):
    batches = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Loading fixed test subset")):
        if batch_idx >= max_batches:
            break
        batches.append(unpack_points(batch).cpu())
    if not batches:
        raise RuntimeError("The selected data loader produced no batches.")
    return batches


def prepare_clean_cache(model, model_type, batches, device, encoder_seed):
    cache = []
    for batch_idx, clean_cpu in enumerate(tqdm(batches, desc="Encoding clean references")):
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
        })
    return cache


def perturbation_seed(base_seed, spec_idx, repeat, batch_idx):
    # Stable arithmetic mapping; independent of RNG consumed by model forwards.
    return base_seed + 10_000_000 * spec_idx + 100_000 * repeat + batch_idx


def evaluate(model, model_type, batches, clean_cache, specs, args, cfg):
    rows = []
    sample_offsets = []
    offset = 0
    for batch in batches:
        sample_offsets.append(offset)
        offset += batch.shape[0]

    with torch.no_grad():
        for spec_idx, spec in enumerate(tqdm(specs, desc="Latent robustness")):
            repeat_count = 1 if spec["corruption"] == "clean" else args.repeats
            for repeat in range(repeat_count):
                for batch_idx, clean_cpu in enumerate(batches):
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

                    tensors = [
                        d_encoder, d_target, d_target_delta, d_end_to_end,
                        reference["clean_target_distance"], reference["fps_noise_floor"],
                        corrupted_norm_ratio, reconstructed_norm_ratio,
                    ]
                    arrays = [tensor.detach().cpu().tolist() for tensor in tensors]
                    for local_idx in range(clean.shape[0]):
                        rows.append({
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
                        })
    return rows


def summarize(rows):
    metrics = [
        "d_encoder_cosine", "d_target_cosine", "d_target_delta_cosine",
        "d_end_to_end_cosine", "clean_target_baseline_cosine",
        "fps_noise_floor_cosine", "corrupted_latent_norm_ratio",
        "reconstructed_latent_norm_ratio",
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


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = config_from_checkpoint(checkpoint, args)
    model, model_type = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()

    loader = make_loader(
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
    batches = load_fixed_batches(loader, args.max_batches)
    with torch.no_grad():
        clean_cache = prepare_clean_cache(
            model, model_type, batches, cfg.device, args.encoder_seed
        )
        rows = evaluate(
            model, model_type, batches, clean_cache, corruption_specs(args), args, cfg
        )

    summary_output = args.summary_output or derived_path(args.output, "_summary.csv")
    diagnostics_output = args.diagnostics_output or derived_path(args.output, "_diagnostics.json")
    write_csv(Path(args.output), rows)
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
    print(f"Wrote {len(rows)} per-sample rows to {args.output}")
    print(f"Wrote aggregate results to {summary_output}")
    print(f"Wrote clean-space diagnostics to {diagnostics_output}")


if __name__ == "__main__":
    main()
