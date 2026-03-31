from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import DatasetConfig
from experiments import BenchmarkConfig, run_loss_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark point-cloud reconstruction losses.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name: synthetic, dvsgesture, nmnist, shd.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--num-points", type=int, default=256, help="Number of points sampled per cloud.")
    parser.add_argument("--point-dim", type=int, default=4, help="Point dimensionality.")
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["chamfer", "density_aware_chamfer", "sinkhorn_emd"],
        help="Losses to benchmark.",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["identity", "gaussian_noise", "temporal_shuffle"],
        help="Corruptions to apply before the comparison.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.05],
        help="Corruption severity values.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Benchmark batch size.")
    parser.add_argument("--num-batches", type=int, default=8, help="Maximum number of batches to evaluate.")
    parser.add_argument("--temporal-weight", type=float, default=1.0, help="Scaling applied to the time axis.")
    parser.add_argument("--download", action="store_true", help="Download tonic datasets if missing.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="Where to store CSV and JSON outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        name=args.dataset,
        root=args.root,
        split=args.split,
        num_points=args.num_points,
        point_dim=args.point_dim,
        temporal_weight=args.temporal_weight,
        download=args.download,
    )
    benchmark = BenchmarkConfig(
        dataset=dataset_config,
        losses=tuple(args.losses),
        corruptions=tuple(args.corruptions),
        severities=tuple(args.severities),
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        device=args.device,
        output_dir=args.output_dir,
    )
    results = run_loss_benchmark(benchmark)
    print(results)


if __name__ == "__main__":
    main()
