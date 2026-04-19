import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluate import DEFAULT_LOSSES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the same autoencoder configuration across multiple reconstruction losses."
    )
    parser.add_argument("--dataset", default="dvsgesture", choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", default="./data")
    parser.add_argument("--model-name", default="pointnet_ae", choices=["pointnet_ae", "pointnet_vae", "pointnetpp_ae"])
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=4, choices=[3, 4])
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--loss-time-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", default="disabled", choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default="geometric-learning-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="autoencoder_convergence")
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/autoencoder_convergence")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=13)
    return parser.parse_args()


def main():
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []

    for loss_name in tqdm(args.losses, desc="AE losses"):
        run_output_dir = base_output_dir / f"{args.dataset}_{args.model_name}_{loss_name}"
        base_run_name = args.wandb_run_name or f"convergence_{args.dataset}_{args.model_name}"
        run_name = f"{base_run_name}_{loss_name}"
        command = [
            sys.executable,
            "-m",
            "src.train_ae",
            "--dataset",
            args.dataset,
            "--save-to",
            args.save_to,
            "--model-name",
            args.model_name,
            "--loss-name",
            loss_name,
            "--num-points",
            str(args.num_points),
            "--input-dim",
            str(args.input_dim),
            "--temporal-weight",
            str(args.temporal_weight),
            "--loss-time-weight",
            str(args.loss_time_weight),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--test-batch-size",
            str(args.test_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--latent-dim",
            str(args.latent_dim),
            "--decoder-hidden-dim",
            str(args.decoder_hidden_dim),
            "--kl-weight",
            str(args.kl_weight),
            "--device",
            args.device,
            "--wandb",
            args.wandb,
            "--wandb-project",
            args.wandb_project,
            "--wandb-run-name",
            run_name,
            "--wandb-group",
            args.wandb_group,
            "--wandb-job-type",
            "autoencoder_convergence",
            "--output-dir",
            str(run_output_dir),
            "--save-every",
            str(args.save_every),
            "--seed",
            str(args.seed),
            "--split-ratio",
            str(args.split_ratio),
            "--split-seed",
            str(args.split_seed),
        ]
        if args.wandb_entity is not None:
            command.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_tags:
            command.append("--wandb-tags")
            command.extend(args.wandb_tags)
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        history_path = run_output_dir / f"{args.dataset}_{args.model_name}_{loss_name}_history.csv"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                aggregate_rows.extend(csv.DictReader(handle))

    if aggregate_rows:
        aggregate_path = base_output_dir / f"{args.dataset}_{args.model_name}_convergence.csv"
        with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"Wrote aggregate convergence CSV to {aggregate_path}")


if __name__ == "__main__":
    main()
