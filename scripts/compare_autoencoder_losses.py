from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import DatasetConfig
from experiments import AutoencoderExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare autoencoder convergence across several losses.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name: synthetic, dvsgesture, nmnist, shd.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--num-points", type=int, default=256, help="Number of points sampled per cloud.")
    parser.add_argument("--point-dim", type=int, default=4, help="Point dimensionality.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding size.")
    parser.add_argument("--model", default="pointnet", help="Model name: pointnet or pointnet++.")
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["chamfer", "density_aware_chamfer", "sinkhorn_emd"],
        help="Losses to compare.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--temporal-weight", type=float, default=1.0, help="Scaling applied to the time axis.")
    parser.add_argument("--download", action="store_true", help="Download tonic datasets if missing.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--output-dir", default="outputs/autoencoder", help="Where to store logs and plots.")
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

    runs = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for loss_name in args.losses:
        experiment = AutoencoderExperiment(
            dataset=dataset_config,
            model_name=args.model,
            loss_name=loss_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            latent_dim=args.latent_dim,
            device=args.device,
            output_dir=str(output_dir),
        )
        history = experiment.run()
        history["loss_name"] = loss_name
        runs.append(history)

    all_histories = pd.concat(runs, ignore_index=True)
    safe_model_name = args.model.replace("+", "plus")
    csv_path = output_dir / f"{args.dataset}_{safe_model_name}_{args.split}_convergence_comparison.csv"
    png_path = output_dir / f"{args.dataset}_{safe_model_name}_{args.split}_convergence_comparison.png"
    all_histories.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for loss_name, group in all_histories.groupby("loss_name"):
        ax.plot(group["epoch"], group["loss"], marker="o", label=loss_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(f"{args.model} convergence on {args.dataset}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    print(all_histories)
    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
