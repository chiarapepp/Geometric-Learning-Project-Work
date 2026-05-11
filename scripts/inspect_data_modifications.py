import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dataset_factory import (  # noqa: E402
    build_pointcloud_transform,
    get_dataset,
    get_sensor_size,
)
from src.datasets.transforms import (  # noqa: E402
    AddUniformNoise,
    DropEventByTime,
    DropEventRandom,
    SpatialJitter,
    TemporalShuffle,
    TimeJitter,
    TimeSkew,
)
from src.evaluate import perturb_points  # noqa: E402


def load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def as_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def print_structured_events(title, events, rows=8):
    print(f"\n=== {title} ===")
    print(f"type: {type(events).__name__}")
    print(f"len: {len(events)}")
    print(f"dtype: {events.dtype}")

    if len(events) == 0:
        print("empty event stream")
        return

    names = events.dtype.names or ()
    for name in names:
        values = events[name]
        print(f"{name} min/max: {values.min()} / {values.max()}")

    print(f"\nfirst {min(rows, len(events))} events:")
    print(events[:rows])


def print_points(title, points, rows=8):
    array = as_numpy(points)
    print(f"\n=== {title} ===")
    print(f"type: {type(points).__name__}")
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")

    if array.size == 0:
        print("empty point cloud")
        return

    dim_names = ["x", "y", "t", "p"]
    for dim in range(array.shape[1]):
        name = dim_names[dim] if dim < len(dim_names) else f"dim{dim}"
        values = array[:, dim]
        print(f"{name} min/max: {values.min():.6f} / {values.max():.6f}")

    print(f"\nfirst {min(rows, len(array))} points:")
    print(array[:rows])


def print_time_comparison(title, before, after, rows=12):
    before_t = before["t"][:rows]
    after_t = after["t"][:rows]
    changed = before_t != after_t

    print(f"\n=== {title}: first {len(before_t)} timestamp positions ===")
    print("idx | before_t | after_t | changed")
    for idx, (old_t, new_t, is_changed) in enumerate(zip(before_t, after_t, changed)):
        print(f"{idx:>3} | {old_t:>8} | {new_t:>7} | {bool(is_changed)}")


def describe_point_delta(title, clean_points, corrupted_points):
    clean = as_numpy(clean_points)
    corrupted = as_numpy(corrupted_points)
    delta = corrupted - clean

    print(f"\n=== {title}: delta summary ===")
    print(f"mean abs delta: {np.mean(np.abs(delta), axis=(0, 1))}")
    print(f"max abs delta:  {np.max(np.abs(delta), axis=(0, 1))}")
    if clean.shape[-1] >= 3:
        changed_t = np.count_nonzero(np.abs(delta[..., 2]) > 1e-12)
        print(f"changed t entries: {changed_t}/{delta[..., 2].size}")


def sample_indices(length, max_points, seed):
    if length <= max_points:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_points, replace=False))


def slug_value(value):
    return str(value).replace("-", "m").replace(".", "p")


def select_levels(plural_values, singular_value, default_values):
    if plural_values is not None:
        return plural_values
    if singular_value is not None:
        return [singular_value]
    return default_values


def event_time_limits(event_variants):
    mins = []
    maxs = []
    for _, events in event_variants:
        if len(events) and events.dtype.names and "t" in events.dtype.names:
            mins.append(float(events["t"].min()))
            maxs.append(float(events["t"].max()))
    if not mins:
        return 0.0, 1.0
    low = min(mins)
    high = max(maxs)
    if high <= low:
        high = low + 1.0
    return low, high


def plot_event_variants(event_variants, sensor_size, output_path, max_points, seed, dpi):
    plt = load_pyplot()

    event_variants = [(title, events) for title, events in event_variants if events is not None]
    ncols = 3
    nrows = int(np.ceil(len(event_variants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    t_min, t_max = event_time_limits(event_variants)

    for ax in axes.ravel():
        ax.axis("off")

    scatter = None
    for panel_idx, (title, events) in enumerate(event_variants):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        ax.axis("on")

        if len(events) == 0:
            ax.set_title(f"{title}\nempty")
            continue

        idx = sample_indices(len(events), max_points, seed + panel_idx)
        sampled = events[idx]
        scatter = ax.scatter(
            sampled["x"],
            sampled["y"],
            c=sampled["t"],
            cmap="viridis",
            vmin=t_min,
            vmax=t_max,
            s=3,
            alpha=0.75,
            linewidths=0,
        )
        ax.set_title(f"{title}\nshown={len(sampled)}/{len(events)}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if sensor_size is not None:
            ax.set_xlim(0, sensor_size[0] - 1)
            ax.set_ylim(sensor_size[1] - 1, 0)
        else:
            ax.invert_yaxis()
        ax.grid(True, linewidth=0.25, alpha=0.35)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78, label="timestamp")
    fig.suptitle("Event variants: x-y projection colored by timestamp", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_comparison(clean_points, corrupted_points, output_path, max_points, seed, dpi):
    plt = load_pyplot()

    clean = as_numpy(clean_points)
    corrupted = as_numpy(corrupted_points)
    clouds = [("clean", clean), ("corrupted", corrupted)]
    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 11.0), squeeze=False)

    t_max = max(float(clean[:, 2].max()), float(corrupted[:, 2].max()), 1.0)
    projections = [
        ("xy projection", 0, 1, "x", "y", 1.0),
        ("xt projection", 0, 2, "x", "t", t_max),
        ("yt projection", 1, 2, "y", "t", t_max),
    ]

    for col, (cloud_name, points) in enumerate(clouds):
        idx = sample_indices(len(points), max_points, seed + col)
        sampled = points[idx]
        for row, (title, x_dim, y_dim, x_label, y_label, y_max) in enumerate(projections):
            ax = axes[row][col]
            color = sampled[:, 2] if sampled.shape[1] >= 3 else None
            ax.scatter(
                sampled[:, x_dim],
                sampled[:, y_dim],
                c=color,
                cmap="viridis",
                vmin=0.0,
                vmax=t_max,
                s=8,
                alpha=0.8,
                linewidths=0,
            )
            ax.set_title(f"{cloud_name}: {title}", fontsize=9)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, y_max)
            ax.grid(True, linewidth=0.25, alpha=0.35)

    fig.suptitle("Point cloud clean vs corrupted", fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_variants(title, point_variants, output_path, max_points, seed, dpi):
    plt = load_pyplot()

    point_variants = [(name, as_numpy(points)) for name, points in point_variants]
    ncols = min(4, len(point_variants))
    nrows_per_projection = int(np.ceil(len(point_variants) / ncols))
    projections = [
        ("xy", 0, 1, "x", "y"),
        ("xt", 0, 2, "x", "t"),
        ("yt", 1, 2, "y", "t"),
    ]
    total_rows = len(projections) * nrows_per_projection
    fig, axes = plt.subplots(
        total_rows,
        ncols,
        figsize=(4.2 * ncols, 3.4 * total_rows),
        squeeze=False,
    )

    t_max = 1.0
    for _, points in point_variants:
        if len(points) and points.shape[1] >= 3:
            t_max = max(t_max, float(points[:, 2].max()))

    for ax in axes.ravel():
        ax.axis("off")

    for projection_idx, (projection_name, x_dim, y_dim, x_label, y_label) in enumerate(projections):
        row_offset = projection_idx * nrows_per_projection
        for variant_idx, (variant_name, points) in enumerate(point_variants):
            row = row_offset + variant_idx // ncols
            col = variant_idx % ncols
            ax = axes[row][col]
            ax.axis("on")
            if len(points) == 0:
                ax.set_title(f"{variant_name}\nempty")
                continue

            idx = sample_indices(len(points), max_points, seed + projection_idx * 100 + variant_idx)
            sampled = points[idx]
            color = sampled[:, 2] if sampled.shape[1] >= 3 else None
            ax.scatter(
                sampled[:, x_dim],
                sampled[:, y_dim],
                c=color,
                cmap="viridis",
                vmin=0.0,
                vmax=t_max,
                s=7,
                alpha=0.8,
                linewidths=0,
            )
            ax.set_title(f"{variant_name} - {projection_name}", fontsize=9)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, t_max if y_dim == 2 else 1.0)
            ax.grid(True, linewidth=0.25, alpha=0.35)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Print one event-camera sample before and after the same temporal/noise/drop "
            "corruptions used in the project experiments."
        )
    )
    parser.add_argument("--dataset", default="dvsgesture", choices=["dvsgesture", "nmnist", "ncaltech101", "all"])
    parser.add_argument("--save-to", default="./data")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=4, choices=[3, 4])
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--shuffle-fraction", type=float, default=None)
    parser.add_argument("--shuffle-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--noise-stds", nargs="+", type=float, default=None)
    parser.add_argument("--drop-fraction", type=float, default=None)
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--time-jitter-std", type=float, default=None)
    parser.add_argument("--time-jitter-stds", nargs="+", type=float, default=None)
    parser.add_argument("--time-skew", type=float, default=None)
    parser.add_argument("--time-skews", nargs="+", type=float, default=None)
    parser.add_argument("--stream-mode", default="sample", choices=["sample", "windowed"])
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--max-windows-per-sample", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/images")
    parser.add_argument("--max-plot-events", type=int, default=20000)
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-images", action="store_true")
    return parser


def inspect_dataset(args, dataset_name):
    train = args.split == "train"
    shuffle_fractions = select_levels(args.shuffle_fractions, args.shuffle_fraction, [0.1, 0.25, 0.5, 1.0])
    noise_stds = select_levels(args.noise_stds, args.noise_std, [0.01, 0.03, 0.05, 0.1])
    drop_fractions = select_levels(args.drop_fractions, args.drop_fraction, [0.1, 0.25, 0.5])
    time_jitter_stds = select_levels(args.time_jitter_stds, args.time_jitter_std, [500.0, 1000.0, 5000.0])
    time_skews = select_levels(args.time_skews, args.time_skew, [0.75, 1.25, 1.5])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_dataset = get_dataset(
        dataset_name=dataset_name,
        save_to=args.save_to,
        train=train,
        transform=None,
        stream_mode=args.stream_mode,
        window_size=args.window_size,
        window_stride=args.window_stride,
        max_windows_per_sample=args.max_windows_per_sample,
    )

    events, label = raw_dataset[args.index]

    print(f"\n##############################")
    print(f"dataset: {dataset_name}")
    print(f"split: {args.split}")
    print(f"dataset length: {len(raw_dataset)}")
    print(f"sample index: {args.index}")
    print(f"label: {label}")
    print(f"seed: {args.seed}")

    print_structured_events("RAW EVENTS", events, rows=args.rows)
    event_variants = [("raw", events)]

    for fraction in shuffle_fractions:
        np.random.seed(args.seed)
        temporal_unsorted = TemporalShuffle(
            fraction=fraction,
            sort_timestamps=False,
        )(events)
        print_structured_events(
            f"RAW + TEMPORAL SHUFFLE fraction={fraction}, sort=False",
            temporal_unsorted,
            rows=args.rows,
        )
        print_time_comparison(
            f"RAW vs TEMPORAL SHUFFLE fraction={fraction}, sort=False",
            events,
            temporal_unsorted,
            rows=max(args.rows, 12),
        )
        event_variants.append((f"temporal shuffle\nf={fraction}", temporal_unsorted))

    for std in time_jitter_stds:
        np.random.seed(args.seed)
        jittered = TimeJitter(std=std)(events)
        print_structured_events(f"RAW + TIME JITTER std={std}", jittered, rows=args.rows)
        event_variants.append((f"time jitter\nstd={std:g}", jittered))

    for coefficient in time_skews:
        skewed = TimeSkew(coefficient=coefficient, offset=0.0)(events)
        print_structured_events(f"RAW + TIME SKEW coefficient={coefficient}", skewed, rows=args.rows)
        event_variants.append((f"time skew\nx{coefficient:g}", skewed))

    for fraction in drop_fractions:
        np.random.seed(args.seed)
        dropped_random = DropEventRandom(drop_probability=fraction)(events)
        print_structured_events(f"RAW + RANDOM DROP drop_probability={fraction}", dropped_random, rows=args.rows)
        event_variants.append((f"random drop\np={fraction}", dropped_random))

        np.random.seed(args.seed)
        dropped_time = DropEventByTime(duration_ratio=fraction)(events)
        print_structured_events(f"RAW + TIME-INTERVAL DROP duration_ratio={fraction}", dropped_time, rows=args.rows)
        event_variants.append((f"time drop\nr={fraction}", dropped_time))

    sensor_size = get_sensor_size(dataset_name)
    if sensor_size is not None:
        np.random.seed(args.seed)
        spatial = SpatialJitter(sensor_size=sensor_size, var_x=1.0, var_y=1.0)(events)
        print_structured_events("RAW + SPATIAL JITTER var_x=1.0 var_y=1.0", spatial, rows=args.rows)
        event_variants.append(("spatial jitter\nvar=1", spatial))

        np.random.seed(args.seed)
        uniform_noise = AddUniformNoise(sensor_size=sensor_size, n=min(100, max(1, len(events) // 20)))(events)
        print_structured_events("RAW + UNIFORM NOISE EVENTS", uniform_noise, rows=args.rows)
        event_variants.append(("uniform noise\nevents", uniform_noise))

    transform = build_pointcloud_transform(
        dataset_name=dataset_name,
        num_points=args.num_points,
        input_dim=args.input_dim,
        temporal_weight=args.temporal_weight,
        sample_mode="uniform",
        pad_mode="repeat",
        shuffle=False,
    )
    clean_points = transform(events)
    print_points("POINT CLOUD CLEAN normalized/sampled", clean_points, rows=args.rows)

    clean_batch = clean_points.unsqueeze(0)
    temporal_point_variants = [("clean", clean_points)]
    noise_point_variants = [("clean", clean_points)]
    drop_point_variants = [("clean", clean_points)]

    for fraction in shuffle_fractions:
        torch.manual_seed(args.seed)
        points = perturb_points(
            clean_batch,
            noise_std=0.0,
            temporal_shuffle_fraction=fraction,
            drop_fraction=0.0,
        )
        squeezed = points.squeeze(0)
        print_points(f"POINT CLOUD TEMPORAL SHUFFLE fraction={fraction}", squeezed, rows=args.rows)
        describe_point_delta(f"POINT CLOUD CLEAN vs TEMPORAL SHUFFLE fraction={fraction}", clean_batch, points)
        temporal_point_variants.append((f"shuffle f={fraction}", squeezed))

    for std in noise_stds:
        torch.manual_seed(args.seed)
        points = perturb_points(
            clean_batch,
            noise_std=std,
            temporal_shuffle_fraction=0.0,
            drop_fraction=0.0,
        )
        squeezed = points.squeeze(0)
        print_points(f"POINT CLOUD GAUSSIAN NOISE std={std}", squeezed, rows=args.rows)
        describe_point_delta(f"POINT CLOUD CLEAN vs GAUSSIAN NOISE std={std}", clean_batch, points)
        noise_point_variants.append((f"noise std={std}", squeezed))

    for fraction in drop_fractions:
        torch.manual_seed(args.seed)
        points = perturb_points(
            clean_batch,
            noise_std=0.0,
            temporal_shuffle_fraction=0.0,
            drop_fraction=fraction,
        )
        squeezed = points.squeeze(0)
        print_points(f"POINT CLOUD RANDOM DROP fraction={fraction}", squeezed, rows=args.rows)
        describe_point_delta(f"POINT CLOUD CLEAN vs RANDOM DROP fraction={fraction}", clean_batch, points)
        drop_point_variants.append((f"drop f={fraction}", squeezed))

    if not args.no_images:
        output_dir = Path(args.output_dir)
        sample_dir = output_dir / dataset_name / f"{args.split}_idx{args.index:04d}_seed{args.seed}"
        events_path = sample_dir / "events_separate_modifications.png"
        temporal_points_path = sample_dir / "pointcloud_temporal_shuffle_levels.png"
        noise_points_path = sample_dir / "pointcloud_gaussian_noise_levels.png"
        drop_points_path = sample_dir / "pointcloud_random_drop_levels.png"

        plot_event_variants(
            event_variants=event_variants,
            sensor_size=sensor_size,
            output_path=events_path,
            max_points=args.max_plot_events,
            seed=args.seed,
            dpi=args.dpi,
        )
        plot_pointcloud_variants(
            title=f"{dataset_name}: clean vs temporal shuffle levels",
            point_variants=temporal_point_variants,
            output_path=temporal_points_path,
            max_points=args.max_plot_points,
            seed=args.seed,
            dpi=args.dpi,
        )
        plot_pointcloud_variants(
            title=f"{dataset_name}: clean vs gaussian noise levels",
            point_variants=noise_point_variants,
            output_path=noise_points_path,
            max_points=args.max_plot_points,
            seed=args.seed,
            dpi=args.dpi,
        )
        plot_pointcloud_variants(
            title=f"{dataset_name}: clean vs random drop levels",
            point_variants=drop_point_variants,
            output_path=drop_points_path,
            max_points=args.max_plot_points,
            seed=args.seed,
            dpi=args.dpi,
        )
        print("\n=== SAVED IMAGES ===")
        print(events_path)
        print(temporal_points_path)
        print(noise_points_path)
        print(drop_points_path)


def main():
    args = build_parser().parse_args()
    dataset_names = ["dvsgesture", "nmnist", "ncaltech101"] if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        try:
            inspect_dataset(args, dataset_name)
        except Exception as exc:
            if len(dataset_names) == 1:
                raise
            print(f"\nERROR while inspecting {dataset_name}: {exc}")


if __name__ == "__main__":
    main()
