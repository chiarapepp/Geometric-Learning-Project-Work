"""Aggregate locally rendered clean panels into report-ready loss comparisons."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


TASK_RE = re.compile(r"^(dvsgesture|ncaltech101|nmnist)__(pointnetpp|pointnet)__sample(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("report/clean_panels/by_task"))
    parser.add_argument("--output", type=Path, default=Path("report_figures/clean_losswise_all_models"))
    args = parser.parse_args()

    tasks = sorted(path for path in args.input.iterdir() if path.is_dir() and TASK_RE.match(path.name))
    if not tasks:
        raise FileNotFoundError(f"No task directories found under {args.input}")

    builder = Path("report/qualitative_figures/build_loss_comparison.py")
    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    for task in tasks:
        match = TASK_RE.match(task.name)
        assert match is not None
        dataset, architecture, sample = match.groups()
        output = args.output / dataset / architecture / f"sample{sample}.png"
        command = [
            sys.executable, str(builder),
            "--root", str(task / "main"),
            "--tw2-root", str(task / "tw2"),
            "--tw5-root", str(task / "tw5"),
            "--dataset", dataset,
            "--architecture", architecture,
            "--sample", sample,
            "--corruption", "gaussian_noise_0p0",
            "--clean-only",
            "--output", str(output),
        ]
        subprocess.run(command, check=True)
        written.append(output)

    print(f"Aggregated {len(written)} clean loss-wise figures into {args.output}")


if __name__ == "__main__":
    main()
