from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import DatasetConfig
from experiments import AutoencoderExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PointNet autoencoder on point-cloud data.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name: synthetic, dvsgesture, nmnist, shd.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--num-points", type=int, default=256, help="Number of points sampled per cloud.")
    parser.add_argument("--point-dim", type=int, default=4, help="Point dimensionality.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding size.")
    parser.add_argument("--model", default="pointnet", help="Model name: pointnet or pointnet++.")
    parser.add_argument("--loss", default="chamfer", help="Loss name.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--temporal-weight", type=float, default=1.0, help="Scaling applied to the time axis.")
    parser.add_argument("--download", action="store_true", help="Download tonic datasets if missing.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--output-dir", default="outputs/autoencoder", help="Where to store checkpoints and logs.")
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
    experiment = AutoencoderExperiment(
        dataset=dataset_config,
        model_name=args.model,
        loss_name=args.loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        device=args.device,
        output_dir=args.output_dir,
    )
    history = experiment.run()
    print(history.tail())


if __name__ == "__main__":
    main()
