import argparse
from pathlib import Path

import torch

from src.evaluate import DEFAULT_LOSSES, benchmark_losses, make_loader, write_csv
from src.utils import set_seed
from src.wandb_util import WandbHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark point-cloud reconstruction losses.")
    parser.add_argument("--dataset", default="dvsgesture", choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", default="./data")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=4, choices=[3, 4])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/benchmarks/loss_comparison.csv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--loss-time-weight", type=float, default=1.0)
    parser.add_argument("--wandb", default="disabled", choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default="geometric-learning-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wandb_run_name is None:
        args.wandb_run_name = f"loss_comparison_{args.dataset}"
    if args.wandb_group is None:
        args.wandb_group = "loss_comparison"
    logger = WandbHandler(
        vars(args),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="loss_comparison",
    )
    set_seed(args.seed)
    loader = make_loader(
        dataset_name=args.dataset,
        save_to=args.save_to,
        split=args.split,
        num_points=args.num_points,
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )
    loss_kwargs = {
        "temporal_weighted_chamfer": {"time_weight": args.loss_time_weight},
    }
    rows = benchmark_losses(
        loader=loader,
        losses=args.losses,
        device=args.device,
        dataset_name=args.dataset,
        split=args.split,
        max_batches=args.max_batches,
        repeats=args.repeats,
        loss_kwargs=loss_kwargs,
    )
    write_csv(Path(args.output), rows)
    logger.log({"benchmark/num_rows": len(rows)})
    logger.log_benchmark_results(rows, csv_path=args.output)
    logger.finish()
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
