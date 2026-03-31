from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2)


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = _pairwise_distance(x, y)
    x_to_y = dist.min(dim=-1).values
    y_to_x = dist.min(dim=-2).values
    return (x_to_y.mean(dim=-1) + y_to_x.mean(dim=-1)).mean()


def _knn_density(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    if points.shape[1] <= 1:
        return torch.ones(points.shape[:2], device=points.device, dtype=points.dtype)
    dist = torch.cdist(points, points, p=2)
    knn = dist.topk(k=min(k + 1, points.shape[1]), largest=False).values[..., 1:]
    density = 1.0 / (knn.mean(dim=-1) + 1e-6)
    return density / density.mean(dim=-1, keepdim=True)


def density_aware_chamfer_distance(x: torch.Tensor, y: torch.Tensor, k: int = 8) -> torch.Tensor:
    dist = _pairwise_distance(x, y)
    x_density = _knn_density(x, k=k)
    y_density = _knn_density(y, k=k)
    x_match = dist.min(dim=-1)
    y_match = dist.min(dim=-2)
    x_loss = (x_match.values * x_density).mean(dim=-1)
    y_loss = (y_match.values * y_density).mean(dim=-1)
    return (x_loss + y_loss).mean()


def sinkhorn_emd_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    iterations: int = 50,
) -> torch.Tensor:
    cost = _pairwise_distance(x, y).pow(2)
    batch_size, num_x, num_y = cost.shape
    mu = torch.full((batch_size, num_x), 1.0 / num_x, device=x.device, dtype=x.dtype)
    nu = torch.full((batch_size, num_y), 1.0 / num_y, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-cost / epsilon).clamp_min_(1e-9)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(iterations):
        u = mu / (kernel @ v.unsqueeze(-1)).squeeze(-1).clamp_min_(1e-9)
        v = nu / (kernel.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min_(1e-9)

    transport = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)
    return (transport * cost).sum(dim=(-1, -2)).mean()


@dataclass(slots=True)
class CombinedLoss:
    alpha: float = 1.0
    beta: float = 0.5

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * chamfer_distance(prediction, target) + self.beta * density_aware_chamfer_distance(
            prediction, target
        )


def reconstruction_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


def build_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name in {"chamfer", "cd"}:
        return chamfer_distance
    if name in {"density_aware_chamfer", "dcd", "dac"}:
        return density_aware_chamfer_distance
    if name in {"sinkhorn_emd", "emd", "sinkhorn"}:
        return sinkhorn_emd_distance
    if name in {"combined", "hybrid"}:
        return CombinedLoss()
    if name in {"mse", "l2"}:
        return reconstruction_mse
    raise ValueError(f"Unknown loss '{name}'.")
