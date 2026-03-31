from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import DatasetConfig
from experiments import AutoencoderExperiment, BenchmarkConfig, run_loss_benchmark


def main() -> None:
    dataset = DatasetConfig(name="synthetic", num_samples=64, num_points=64)
    benchmark = BenchmarkConfig(dataset=dataset, batch_size=8, num_batches=2)
    benchmark_frame = run_loss_benchmark(benchmark)
    print("Benchmark rows:", len(benchmark_frame))

    experiment = AutoencoderExperiment(
        dataset=dataset,
        model_name="pointnet",
        loss_name="chamfer",
        batch_size=8,
        epochs=2,
        latent_dim=32,
    )
    history = experiment.run()
    print("Final training loss:", float(history["loss"].iloc[-1]))

    experiment_pp = AutoencoderExperiment(
        dataset=dataset,
        model_name="pointnet++",
        loss_name="chamfer",
        batch_size=8,
        epochs=1,
        latent_dim=32,
    )
    history_pp = experiment_pp.run()
    print("Final PointNet++ training loss:", float(history_pp["loss"].iloc[-1]))



if __name__ == "__main__":
    main()
