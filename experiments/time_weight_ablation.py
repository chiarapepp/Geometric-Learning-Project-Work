import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["dvsgesture", "nmnist"]
DEFAULT_MODELS = ["pointnet_ae"]
DEFAULT_TIME_WEIGHTS = [1.0, 2.0, 5.0, 10.0]
DEFAULT_METRICS = ["chamfer", "temporal_weighted_chamfer", "hausdorff", "mse"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused ablation for temporal_weighted_chamfer with multiple "
            "time weights, plus an optional Chamfer baseline."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--time-weights", nargs="+", type=float, default=DEFAULT_TIME_WEIGHTS)
    parser.add_argument("--skip-chamfer-baseline", action="store_true")
    parser.add_argument("--save-to", default="./data")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=4, choices=[3, 4])
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--sample-mode", default="random", choices=["random", "uniform", "first"])
    parser.add_argument("--pad-mode", default="repeat", choices=["repeat", "zeros"])
    parser.add_argument("--no-shuffle-points", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--stream-mode", default="sample", choices=["sample", "windowed"])
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--keep-last-window", action="store_true")
    parser.add_argument("--max-windows-per-sample", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/time_weight_ablation")
    parser.add_argument("--eval-output-dir", default="outputs/time_weight_ablation_eval")
    parser.add_argument("--visual-output-dir", default="outputs/time_weight_ablation_visual")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume an interrupted run from the latest periodic checkpoint or best checkpoint.",
    )
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--run-visual", action="store_true")
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--eval-metric-time-weight", type=float, default=5.0)
    parser.add_argument("--eval-metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--visual-sample-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb", default="disabled", choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default="geometric-learning-project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="time_weight_ablation")
    return parser.parse_args()


def weight_tag(weight):
    return str(weight).replace(".", "p")


def run_id(dataset, model, loss_name, time_weight=None):
    if loss_name == "chamfer":
        return f"{dataset}_{model}_chamfer"
    return f"{dataset}_{model}_temporal_weighted_chamfer_tw{weight_tag(time_weight)}"


def checkpoint_path(output_dir, dataset, model, loss_name, time_weight=None):
    rid = run_id(dataset, model, loss_name, time_weight)
    return (
        Path(output_dir)
        / rid
        / f"{dataset}_{model}_{loss_name}_best.pth"
    )


def history_path(output_dir, dataset, model, loss_name, time_weight=None):
    rid = run_id(dataset, model, loss_name, time_weight)
    return (
        Path(output_dir)
        / rid
        / f"{dataset}_{model}_{loss_name}_history.csv"
    )


def completed_run(args, dataset, model, loss_name, time_weight=None):
    ckpt = checkpoint_path(args.output_dir, dataset, model, loss_name, time_weight)
    hist = history_path(args.output_dir, dataset, model, loss_name, time_weight)
    if not ckpt.exists() or not hist.exists():
        return False
    try:
        with hist.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return False
    if not rows:
        return False
    max_epoch = max(int(float(row["epoch"])) for row in rows if row.get("epoch") not in (None, ""))
    return max_epoch + 1 >= args.epochs


def latest_resume_checkpoint(output_dir, dataset, model, loss_name, time_weight=None):
    rid = run_id(dataset, model, loss_name, time_weight)
    run_dir = Path(output_dir) / rid
    prefix = f"{dataset}_{model}_{loss_name}_epoch_"
    candidates = []
    for path in run_dir.glob(f"{prefix}*.pth"):
        try:
            epoch_text = path.stem.replace(prefix, "")
            candidates.append((int(epoch_text), path))
        except ValueError:
            continue
    if candidates:
        return sorted(candidates)[-1][1]

    best = checkpoint_path(output_dir, dataset, model, loss_name, time_weight)
    if best.exists():
        return best
    return None


def command_to_string(command):
    return " ".join(str(part) for part in command)


def run_command(command):
    print("Running:", command_to_string(command), flush=True)
    subprocess.run(command, check=True)


def wandb_args(args, run_name):
    command = [
        "--wandb",
        args.wandb,
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        run_name,
        "--wandb-group",
        args.wandb_group,
    ]
    if args.wandb_entity is not None:
        command.extend(["--wandb-entity", args.wandb_entity])
    return command


def training_specs(args):
    specs = []
    if not args.skip_chamfer_baseline:
        specs.append({"loss_name": "chamfer", "time_weight": None})
    for time_weight in args.time_weights:
        specs.append(
            {
                "loss_name": "temporal_weighted_chamfer",
                "time_weight": float(time_weight),
            }
        )
    return specs


def train_one(args, dataset, model, spec):
    loss_name = spec["loss_name"]
    time_weight = spec["time_weight"]
    rid = run_id(dataset, model, loss_name, time_weight)
    out_dir = Path(args.output_dir) / rid
    ckpt = checkpoint_path(args.output_dir, dataset, model, loss_name, time_weight)

    if args.skip_existing and completed_run(args, dataset, model, loss_name, time_weight):
        print(f"Skipping completed run: {rid}", flush=True)
        return ckpt

    resume_from = None
    if args.resume_existing:
        resume_from = latest_resume_checkpoint(args.output_dir, dataset, model, loss_name, time_weight)
        if resume_from is not None:
            print(f"Resuming {rid} from {resume_from}", flush=True)

    command = [
        sys.executable,
        "-m",
        "src.train_ae",
        "--dataset",
        dataset,
        "--save-to",
        args.save_to,
        "--model-name",
        model,
        "--loss-name",
        loss_name,
        "--num-points",
        str(args.num_points),
        "--input-dim",
        str(args.input_dim),
        "--temporal-weight",
        str(args.temporal_weight),
        "--sample-mode",
        args.sample_mode,
        "--pad-mode",
        args.pad_mode,
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
        "--output-dir",
        str(out_dir),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
        "--split-ratio",
        str(args.split_ratio),
        "--split-seed",
        str(args.split_seed),
        "--stream-mode",
        args.stream_mode,
    ]
    if args.window_size is not None:
        command.extend(["--window-size", str(args.window_size)])
    if args.window_stride is not None:
        command.extend(["--window-stride", str(args.window_stride)])
    if args.keep_last_window:
        command.append("--keep-last-window")
    if args.max_windows_per_sample is not None:
        command.extend(["--max-windows-per-sample", str(args.max_windows_per_sample)])
    if args.no_shuffle_points:
        command.append("--no-shuffle-points")
    if loss_name == "temporal_weighted_chamfer":
        command.extend(["--loss-time-weight", str(time_weight)])
    else:
        command.extend(["--loss-time-weight", "1.0"])
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    command.extend(wandb_args(args, f"tw_ablation_{rid}"))

    run_command(command)
    return ckpt


def eval_one(args, dataset, model, spec, ckpt):
    loss_name = spec["loss_name"]
    time_weight = spec["time_weight"]
    rid = run_id(dataset, model, loss_name, time_weight)
    out_dir = Path(args.eval_output_dir) / rid
    out_csv = out_dir / f"{rid}_corruptions.csv"

    if args.skip_existing and out_csv.exists():
        print(f"Skipping existing eval CSV: {out_csv}", flush=True)
        return out_csv

    command = [
        sys.executable,
        "-m",
        "experiments.reconstruction_corruption_eval",
        "--checkpoint",
        str(ckpt),
        "--split",
        "test",
        "--metrics",
        *args.eval_metrics,
        "--metric-time-weight",
        str(args.eval_metric_time_weight),
        "--noise-stds",
        "0.0",
        "0.05",
        "0.1",
        "--temporal-shuffle-fractions",
        "0.0",
        "0.5",
        "1.0",
        "--drop-fractions",
        "0.0",
        "0.25",
        "0.5",
        "--max-batches",
        str(args.eval_max_batches),
        "--device",
        args.device,
        "--output",
        str(out_csv),
        "--plot-dir",
        str(out_dir / "plots"),
        "--wandb",
        args.wandb,
        "--wandb-project",
        args.wandb_project,
        "--wandb-run-name",
        f"tw_eval_{rid}",
        "--wandb-group",
        f"{args.wandb_group}_eval",
    ]
    if args.wandb_entity is not None:
        command.extend(["--wandb-entity", args.wandb_entity])

    run_command(command)
    return out_csv


def visual_one(args, dataset, model, spec, ckpt):
    loss_name = spec["loss_name"]
    time_weight = spec["time_weight"]
    rid = run_id(dataset, model, loss_name, time_weight)
    out_dir = Path(args.visual_output_dir) / rid
    manifest = out_dir / "manifest.csv"

    if args.skip_existing and manifest.exists():
        print(f"Skipping existing visual manifest: {manifest}", flush=True)
        return manifest

    command = [
        sys.executable,
        "-m",
        "experiments.visual_reconstruction_eval",
        "--checkpoints",
        str(ckpt),
        "--sample-indices",
        *[str(index) for index in args.visual_sample_indices],
        "--batch-size",
        str(args.test_batch_size),
        "--num-workers",
        "0",
        "--max-plot-points",
        str(args.max_plot_points),
        "--metric-time-weight",
        str(args.eval_metric_time_weight),
        "--noise-stds",
        "0.0",
        "0.1",
        "--temporal-shuffle-fractions",
        "1.0",
        "--drop-fractions",
        "0.5",
        "--device",
        args.device,
        "--output-dir",
        str(out_dir),
    ]
    run_command(command)
    return manifest


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        for model in args.models:
            for spec in training_specs(args):
                ckpt = train_one(args, dataset, model, spec)
                eval_csv = ""
                visual_manifest = ""
                if args.run_eval:
                    eval_csv = str(eval_one(args, dataset, model, spec, ckpt))
                if args.run_visual:
                    visual_manifest = str(visual_one(args, dataset, model, spec, ckpt))
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "loss_name": spec["loss_name"],
                        "loss_time_weight": spec["time_weight"] if spec["time_weight"] is not None else 1.0,
                        "checkpoint": str(ckpt),
                        "eval_csv": eval_csv,
                        "visual_manifest": visual_manifest,
                    }
                )

    manifest_path = Path(args.output_dir) / "time_weight_ablation_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
