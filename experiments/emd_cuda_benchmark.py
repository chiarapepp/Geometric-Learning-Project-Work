import argparse
from pathlib import Path

import torch

from src.evaluate import write_csv
from src.losses.loss_factory import get_loss
from src.utils import cuda_peak_memory, peak_memory_mb, set_seed, time_call


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark for the optional CUDA approximate EMD backend."
    )
    parser.add_argument("--losses", nargs="+", default=["emd_cuda", "emd", "sinkhorn"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--point-dim", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="outputs/benchmarks/emd_cuda_synthetic.csv")
    parser.add_argument("--max-hungarian-points", type=int, default=256)
    parser.add_argument("--emd-cuda-eps", type=float, default=0.005)
    parser.add_argument("--emd-cuda-iterations", type=int, default=300)
    parser.add_argument("--emd-cuda-no-sqrt", action="store_true")
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    parser.add_argument("--sinkhorn-scaling", type=float, default=0.9)
    return parser.parse_args()


def make_clouds(args):
    device = torch.device(args.device)
    target = torch.rand(args.batch_size, args.num_points, args.point_dim, device=device)
    prediction = target + args.noise_std * torch.randn_like(target)
    return prediction, target


def run_timed_loss(loss_fn, prediction, target, args):
    def run_once():
        cur_prediction = prediction.detach().clone().requires_grad_(True)
        loss = loss_fn(cur_prediction, target)
        loss.backward()
        grad_norm = float(cur_prediction.grad.detach().norm().cpu().item())
        return loss.detach(), grad_norm

    with cuda_peak_memory(args.device):
        result, seconds = time_call(
            run_once,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
        )
        memory = peak_memory_mb(args.device)
    return result[0], result[1], seconds, memory


def make_row(args, loss_name, status, value=None, seconds=None, memory=None, grad_norm=None, error=""):
    return {
        "loss": loss_name,
        "status": status,
        "error": error,
        "value": "" if value is None else float(value),
        "seconds": "" if seconds is None else float(seconds),
        "peak_memory_mb": "" if memory is None else float(memory),
        "grad_norm": "" if grad_norm is None else float(grad_norm),
        "batch_size": args.batch_size,
        "num_points": args.num_points,
        "point_dim": args.point_dim,
        "device": args.device,
        "noise_std": args.noise_std,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "emd_cuda_eps": args.emd_cuda_eps,
        "emd_cuda_iterations": args.emd_cuda_iterations,
        "emd_cuda_sqrt": not args.emd_cuda_no_sqrt,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    prediction, target = make_clouds(args)
    loss_kwargs = {
        "emd_cuda": {
            "eps": args.emd_cuda_eps,
            "iterations": args.emd_cuda_iterations,
            "sqrt": not args.emd_cuda_no_sqrt,
        },
        "sinkhorn": {
            "blur": args.sinkhorn_blur,
            "scaling": args.sinkhorn_scaling,
        },
    }

    rows = []
    for loss_name in args.losses:
        if loss_name == "emd" and args.num_points > args.max_hungarian_points:
            message = (
                f"skipped Hungarian EMD above {args.max_hungarian_points} points; "
                "raise --max-hungarian-points to force it"
            )
            rows.append(make_row(args, loss_name, "skipped", error=message))
            print(f"{loss_name}: skipped ({message})")
            continue

        try:
            loss_fn = get_loss(loss_name, **loss_kwargs.get(loss_name, {}))
            value, grad_norm, seconds, memory = run_timed_loss(loss_fn, prediction, target, args)
            rows.append(
                make_row(
                    args,
                    loss_name,
                    "ok",
                    value=value.cpu().item(),
                    seconds=seconds,
                    memory=memory,
                    grad_norm=grad_norm,
                )
            )
            print(
                f"{loss_name}: value={value.cpu().item():.6f} "
                f"time={seconds:.6f}s grad_norm={grad_norm:.6f} memory={memory:.2f}MB"
            )
        except Exception as exc:
            rows.append(make_row(args, loss_name, "error", error=str(exc)))
            print(f"{loss_name}: ERROR: {exc}")

    write_csv(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
