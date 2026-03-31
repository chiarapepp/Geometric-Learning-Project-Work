from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class DatasetConfig:
    name: str = "synthetic"
    root: str = "data"
    split: str = "train"
    num_points: int = 256
    point_dim: int = 4
    num_samples: int = 512
    temporal_weight: float = 1.0
    download: bool = False
    seed: int = 42


def _normalise_points(points: np.ndarray, temporal_weight: float) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    if points.shape[0] == 0:
        return points

    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    scale = np.where((maxs - mins) > 1e-8, maxs - mins, 1.0)
    points = (points - mins) / scale
    if points.shape[1] >= 3:
        points[:, 2] *= temporal_weight
    if points.shape[1] >= 4:
        points[:, 3] = np.where(points[:, 3] > 0, 1.0, 0.0)
    return points


def sample_point_cloud(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((num_points, points.shape[1]), dtype=np.float32)
    if points.shape[0] >= num_points:
        indices = rng.choice(points.shape[0], size=num_points, replace=False)
    else:
        indices = rng.choice(points.shape[0], size=num_points, replace=True)
    return points[indices]


def event_array_to_point_cloud(
    events: np.ndarray,
    num_points: int,
    point_dim: int = 4,
    temporal_weight: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    names = set(events.dtype.names or [])
    if {"x", "y", "t", "p"}.issubset(names):
        points = np.stack([events["x"], events["y"], events["t"], events["p"]], axis=1)
    elif {"x", "y", "t"}.issubset(names):
        polarity = np.zeros_like(events["x"], dtype=np.float32)
        points = np.stack([events["x"], events["y"], events["t"], polarity], axis=1)
    else:
        points = np.asarray(events, dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected 2D event array, received shape={points.shape}.")
    if point_dim < points.shape[1]:
        points = points[:, :point_dim]
    elif point_dim > points.shape[1]:
        padding = np.zeros((points.shape[0], point_dim - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, padding], axis=1)

    points = _normalise_points(points, temporal_weight=temporal_weight)
    return sample_point_cloud(points, num_points=num_points, rng=rng)


def add_gaussian_noise(points: torch.Tensor, sigma: float, seed: int | None = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=points.device)
        generator.manual_seed(seed)
    noise = torch.randn(
        points.shape,
        generator=generator,
        device=points.device,
        dtype=points.dtype,
    ) * sigma
    return (points + noise).clamp_(0.0, 1.0)


def shuffle_temporal_axis(points: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if points.shape[-1] < 3:
        return points
    generator = None
    if seed is not None:
        generator = torch.Generator(device=points.device)
        generator.manual_seed(seed)
    indices = torch.randperm(points.shape[-2], generator=generator, device=points.device)
    shuffled = points.clone()
    shuffled[..., 2] = points[..., indices, 2]
    return shuffled


class SyntheticEventPointCloudDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return self.config.num_samples

    def _make_sample(self, index: int) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(self.config.seed + index)
        label = index % 5
        num_raw_points = rng.integers(self.config.num_points, self.config.num_points * 3)

        if label == 0:
            xyz = rng.normal(loc=(0.5, 0.5, 0.5), scale=0.12, size=(num_raw_points, 3))
        elif label == 1:
            t = rng.uniform(0, 2 * np.pi, size=(num_raw_points, 1))
            z = rng.uniform(0.0, 1.0, size=(num_raw_points, 1))
            xyz = np.concatenate([0.25 * np.cos(t) + 0.5, 0.25 * np.sin(t) + 0.5, z], axis=1)
        elif label == 2:
            xyz = rng.uniform(0.15, 0.85, size=(num_raw_points, 3))
            xyz[:, 2] = 0.3 * xyz[:, 0] + 0.4 * xyz[:, 1]
        elif label == 3:
            corners = rng.choice(
                np.array(
                    [
                        [0.2, 0.2, 0.2],
                        [0.8, 0.2, 0.2],
                        [0.2, 0.8, 0.2],
                        [0.8, 0.8, 0.8],
                    ]
                ),
                size=num_raw_points,
            )
            xyz = corners + rng.normal(scale=0.05, size=(num_raw_points, 3))
        else:
            phi = rng.uniform(0, np.pi, size=(num_raw_points, 1))
            theta = rng.uniform(0, 2 * np.pi, size=(num_raw_points, 1))
            xyz = np.concatenate(
                [
                    0.3 * np.sin(phi) * np.cos(theta) + 0.5,
                    0.3 * np.sin(phi) * np.sin(theta) + 0.5,
                    0.3 * np.cos(phi) + 0.5,
                ],
                axis=1,
            )

        t = np.linspace(0.0, 1.0, num_raw_points, dtype=np.float32)[:, None]
        polarity = rng.integers(0, 2, size=(num_raw_points, 1)).astype(np.float32)
        points = np.concatenate([xyz, t, polarity], axis=1)[:, : self.config.point_dim]
        points = _normalise_points(points, temporal_weight=self.config.temporal_weight)
        points = sample_point_cloud(points, self.config.num_points, rng)
        return points.astype(np.float32), label

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        points, label = self._make_sample(index)
        tensor = torch.from_numpy(points)
        return {"points": tensor, "target": tensor.clone(), "label": torch.tensor(label, dtype=torch.long)}


class TonicPointCloudDataset(Dataset):
    DATASET_MAP = {
        "dvsgesture": "DVSGesture",
        "nmnist": "NMNIST",
        "shd": "SHD",
    }

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.root = Path(config.root)
        self.rng = np.random.default_rng(config.seed)
        try:
            import tonic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'tonic' package is required for neuromorphic datasets. "
                "Install it with 'python -m pip install tonic'."
            ) from exc

        dataset_name = config.name.lower()
        if dataset_name not in self.DATASET_MAP:
            available = ", ".join(sorted(self.DATASET_MAP))
            raise ValueError(f"Unsupported tonic dataset '{config.name}'. Available: {available}.")

        dataset_cls = getattr(tonic.datasets, self.DATASET_MAP[dataset_name])
        train = config.split.lower() == "train"
        self.dataset = dataset_cls(save_to=str(self.root), train=train, download=config.download)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        events, label = self.dataset[index]
        rng = np.random.default_rng(self.config.seed + index)
        points = event_array_to_point_cloud(
            events=events,
            num_points=self.config.num_points,
            point_dim=self.config.point_dim,
            temporal_weight=self.config.temporal_weight,
            rng=rng,
        )
        tensor = torch.from_numpy(points.astype(np.float32))
        return {"points": tensor, "target": tensor.clone(), "label": torch.tensor(int(label), dtype=torch.long)}


def build_dataset(config: DatasetConfig) -> Dataset:
    name = config.name.lower()
    if name == "synthetic":
        return SyntheticEventPointCloudDataset(config)
    return TonicPointCloudDataset(config)


def make_corruption(name: str, severity: float, seed: int | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "identity":
        return lambda points: points
    if name == "gaussian_noise":
        return lambda points: add_gaussian_noise(points, sigma=severity, seed=seed)
    if name == "temporal_shuffle":
        return lambda points: shuffle_temporal_axis(points, seed=seed)
    raise ValueError(f"Unknown corruption '{name}'.")
