from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import DatasetConfig, build_dataset, make_corruption
from losses import build_loss
from model import ModelConfig, build_autoencoder


@dataclass(slots=True)
class BenchmarkConfig:
    dataset: DatasetConfig
    losses: tuple[str, ...] = ("chamfer", "density_aware_chamfer", "sinkhorn_emd")
    corruptions: tuple[str, ...] = ("identity", "gaussian_noise", "temporal_shuffle")
    severities: tuple[float, ...] = (0.0, 0.02, 0.05)
    batch_size: int = 16
    num_batches: int = 8
    device: str = "cpu"
    output_dir: str = "outputs/benchmarks"


def run_loss_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    dataset = build_dataset(config.dataset)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    device = torch.device(config.device)

    rows: list[dict[str, float | str | int]] = []
    for corruption in config.corruptions:
        for severity in config.severities:
            transform = make_corruption(corruption, severity=severity, seed=config.dataset.seed)
            for loss_name in config.losses:
                criterion = build_loss(loss_name)
                batch_times = []
                batch_losses = []
                for batch_index, batch in enumerate(loader):
                    if batch_index >= config.num_batches:
                        break
                    target = batch["target"].to(device)
                    source = transform(target.clone())
                    start = time.perf_counter()
                    loss_value = criterion(source, target)
                    batch_times.append(time.perf_counter() - start)
                    batch_losses.append(loss_value.detach().cpu().item())

                rows.append(
                    {
                        "dataset": config.dataset.name,
                        "split": config.dataset.split,
                        "corruption": corruption,
                        "severity": severity,
                        "loss_name": loss_name,
                        "mean_loss": float(sum(batch_losses) / max(len(batch_losses), 1)),
                        "mean_runtime_sec": float(sum(batch_times) / max(len(batch_times), 1)),
                        "num_batches": min(len(loader), config.num_batches),
                    }
                )

    frame = pd.DataFrame(rows)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{config.dataset.name}_{config.dataset.split}_benchmark.csv"
    json_path = output_dir / f"{config.dataset.name}_{config.dataset.split}_benchmark.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"config": asdict(config), "rows": rows}, indent=2), encoding="utf-8")
    return frame


@dataclass(slots=True)
class AutoencoderExperiment:
    dataset: DatasetConfig
    model_name: str = "pointnet"
    loss_name: str = "chamfer"
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    latent_dim: int = 128
    device: str = "cpu"
    output_dir: str = "outputs/autoencoder"

    def run(self) -> pd.DataFrame:
        dataset = build_dataset(self.dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        device = torch.device(self.device)

        model = build_autoencoder(
            ModelConfig(
                name=self.model_name,
                point_dim=self.dataset.point_dim,
                latent_dim=self.latent_dim,
                num_points=self.dataset.num_points,
            )
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = build_loss(self.loss_name)

        history: list[dict[str, float | int]] = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            running_loss = 0.0
            sample_count = 0
            start = time.perf_counter()

            for batch in loader:
                points = batch["points"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(points)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                batch_size = points.shape[0]
                running_loss += loss.detach().cpu().item() * batch_size
                sample_count += batch_size

            epoch_loss = running_loss / max(sample_count, 1)
            history.append(
                {
                    "epoch": epoch,
                    "model_name": self.model_name,
                    "loss_name": self.loss_name,
                    "loss": epoch_loss,
                    "epoch_time_sec": time.perf_counter() - start,
                }
            )

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_model_name = self.model_name.replace("+", "plus")
        prefix = f"{self.dataset.name}_{safe_model_name}_{self.loss_name}_{self.dataset.split}"
        model_path = output_dir / f"{prefix}.pt"
        history_path = output_dir / f"{prefix}.csv"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset_config": asdict(self.dataset),
                "experiment": asdict(self),
            },
            model_path,
        )
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(history_path, index=False)
        return history_frame


def compare_autoencoders(
    dataset: DatasetConfig,
    model_names: tuple[str, ...],
    loss_names: tuple[str, ...],
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    latent_dim: int = 128,
    device: str = "cpu",
    output_dir: str = "outputs/autoencoder",
) -> pd.DataFrame:
    runs = []
    for model_name in model_names:
        for loss_name in loss_names:
            history = AutoencoderExperiment(
                dataset=dataset,
                model_name=model_name,
                loss_name=loss_name,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                latent_dim=latent_dim,
                device=device,
                output_dir=output_dir,
            ).run()
            runs.append(history)

    frame = pd.concat(runs, ignore_index=True)
    output_path = Path(output_dir) / f"{dataset.name}_{dataset.split}_model_loss_comparison.csv"
    frame.to_csv(output_path, index=False)
    return frame
