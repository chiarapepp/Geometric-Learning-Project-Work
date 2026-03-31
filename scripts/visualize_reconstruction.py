from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import DatasetConfig, build_dataset
from model import ModelConfig, build_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize target and reconstructed point clouds.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pt checkpoint.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name: synthetic, dvsgesture, nmnist, shd.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--num-points", type=int, default=256, help="Number of points sampled per cloud.")
    parser.add_argument("--point-dim", type=int, default=4, help="Point dimensionality.")
    parser.add_argument("--model", default="pointnet", help="Model name: pointnet or pointnet++.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent embedding size.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index inside the first batch.")
    parser.add_argument("--temporal-weight", type=float, default=1.0, help="Scaling applied to the time axis.")
    parser.add_argument("--download", action="store_true", help="Download tonic datasets if missing.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--output", default="outputs/autoencoder/reconstruction_preview.png", help="Output figure path.")
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
    dataset = build_dataset(dataset_config)
    loader = DataLoader(dataset, batch_size=max(args.sample_index + 1, 1), shuffle=False)
    batch = next(iter(loader))

    device = torch.device(args.device)
    model = build_autoencoder(
        ModelConfig(
            name=args.model,
            point_dim=args.point_dim,
            latent_dim=args.latent_dim,
            num_points=args.num_points,
        )
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    points = batch["points"].to(device)
    with torch.no_grad():
        reconstruction, _ = model(points)

    target = points[args.sample_index].cpu().numpy()
    pred = reconstruction[args.sample_index].cpu().numpy()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(target[:, 0], target[:, 1], target[:, 2], c=target[:, 2], s=8, cmap="viridis")
    ax1.set_title("Target")
    ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=pred[:, 2], s=8, cmap="viridis")
    ax2.set_title("Reconstruction")

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t/z")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved reconstruction preview to {output_path}")


if __name__ == "__main__":
    main()
