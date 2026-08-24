"""Inspect class and recording composition, and create a reproducible subset.

Examples
--------
Inspect the entire windowed test split::

    python -m experiments.inspect_dataset_composition --dataset nmnist \
      --save-to /path/to/data --split test --stream-mode windowed \
      --window-size 4096 --window-stride 4096 --output-dir outputs/composition

Create a balanced subset with at most one window from each source recording::

    python -m experiments.inspect_dataset_composition --dataset ncaltech101 \
      --save-to /path/to/data --split test --stream-mode windowed \
      --window-size 4096 --window-stride 4096 --output-dir outputs/composition \
      --items-per-class 4 --max-windows-per-source 1
"""

import argparse
import json
from pathlib import Path

from src.datasets.dataset_factory import get_dataset
from src.datasets.evaluation_sampling import (
    build_dataset_manifest,
    composition_by_class,
    composition_by_source,
    select_manifest_rows,
    write_csv_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--stream-mode", default="windowed", choices=["sample", "windowed"])
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--window-stride", type=int, default=4096)
    parser.add_argument("--keep-last-window", action="store_true")
    parser.add_argument("--max-windows-per-sample", type=int, default=None)
    parser.add_argument("--items-per-class", type=int, default=None,
                        help="Also write a balanced subset manifest with this many items per class.")
    parser.add_argument("--max-windows-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = get_dataset(
        dataset_name=args.dataset, save_to=args.save_to, train=args.split == "train", split=args.split,
        split_ratio=args.split_ratio, split_seed=args.split_seed, stream_mode=args.stream_mode,
        window_size=args.window_size if args.stream_mode == "windowed" else None,
        window_stride=args.window_stride if args.stream_mode == "windowed" else None,
        window_drop_last=not args.keep_last_window,
        max_windows_per_sample=args.max_windows_per_sample,
    )
    manifest = build_dataset_manifest(dataset, args.dataset)
    output_dir = Path(args.output_dir)
    prefix = f"{args.dataset}_{args.split}_{args.stream_mode}"
    write_csv_rows(output_dir / f"{prefix}_classes.csv", composition_by_class(manifest))
    write_csv_rows(output_dir / f"{prefix}_sources.csv", composition_by_source(manifest))

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "stream_mode": args.stream_mode,
        "dataset_items": len(manifest),
        "classes": len(composition_by_class(manifest)),
        "source_samples": len({row["source_index"] for row in manifest}),
        "class_composition": composition_by_class(manifest),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.items_per_class is not None:
        selected = select_manifest_rows(
            manifest, mode="stratified", items_per_class=args.items_per_class,
            max_windows_per_source=args.max_windows_per_source, seed=args.seed,
        )
        subset_path = output_dir / f"{prefix}_balanced_subset.csv"
        write_csv_rows(subset_path, selected)
        write_csv_rows(output_dir / f"{prefix}_balanced_subset_classes.csv", composition_by_class(selected))
        print(f"Wrote balanced subset ({len(selected)} items) to {subset_path}")

    print(f"Wrote composition for {len(manifest)} items to {output_dir}")


if __name__ == "__main__":
    main()
