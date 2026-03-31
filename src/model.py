from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ModelConfig:
    name: str = "pointnet"
    point_dim: int = 4
    latent_dim: int = 128
    num_points: int = 256


def _pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2)


def _farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    batch_size, num_points, _ = points.shape
    num_samples = min(num_samples, num_points)
    centroids = torch.zeros(batch_size, num_samples, dtype=torch.long, device=points.device)
    distance = torch.full((batch_size, num_points), float("inf"), device=points.device)
    farthest = torch.randint(0, num_points, (batch_size,), device=points.device)
    batch_indices = torch.arange(batch_size, device=points.device)

    for sample_index in range(num_samples):
        centroids[:, sample_index] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        dist = ((points - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices

    return centroids


def _gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size = points.shape[0]
    if indices.ndim == 2:
        batch_indices = torch.arange(batch_size, device=points.device)[:, None]
        return points[batch_indices, indices]
    batch_indices = torch.arange(batch_size, device=points.device)[:, None, None]
    return points[batch_indices, indices]


class SharedMLP(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class PointNetEncoder(nn.Module):
    def __init__(self, point_dim: int = 4, latent_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(point_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, latent_dim, kernel_size=1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.mlp(points.transpose(1, 2))
        return features.max(dim=-1).values


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int], num_samples: int, num_neighbors: int):
        super().__init__()
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP([in_channels + 3, *out_channels])

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        sample_indices = _farthest_point_sample(xyz, self.num_samples)
        centroids = _gather_points(xyz, sample_indices)

        distances = _pairwise_distance(centroids, xyz)
        neighbor_indices = distances.topk(k=min(self.num_neighbors, xyz.shape[1]), largest=False).indices

        grouped_xyz = _gather_points(xyz, neighbor_indices)
        relative_xyz = grouped_xyz - centroids.unsqueeze(2)

        if features is None:
            grouped_features = relative_xyz
            mlp_input = grouped_features.permute(0, 3, 1, 2)
        else:
            grouped_features = _gather_points(features, neighbor_indices)
            mlp_input = torch.cat([relative_xyz, grouped_features], dim=-1).permute(0, 3, 1, 2)

        local_features = self.mlp(mlp_input).max(dim=-1).values
        return centroids, local_features.transpose(1, 2)


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, point_dim: int = 4, latent_dim: int = 128):
        super().__init__()
        self.spatial_dim = min(3, point_dim)
        self.sa1 = PointNetSetAbstraction(
            in_channels=point_dim,
            out_channels=[64, 64, 128],
            num_samples=64,
            num_neighbors=16,
        )
        self.sa2 = PointNetSetAbstraction(
            in_channels=128,
            out_channels=[128, 128, 256],
            num_samples=16,
            num_neighbors=16,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., : self.spatial_dim]
        if self.spatial_dim < 3:
            pad = torch.zeros(*points.shape[:2], 3 - self.spatial_dim, device=points.device, dtype=points.dtype)
            xyz = torch.cat([xyz, pad], dim=-1)
        centroids_1, features_1 = self.sa1(xyz, points)
        _, features_2 = self.sa2(centroids_1, features_1)
        global_features = features_2.max(dim=1).values
        return self.head(global_features)


class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, num_points: int = 256, point_dim: int = 4):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * point_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.mlp(latent)
        output = output.view(latent.shape[0], self.num_points, self.point_dim)
        return torch.sigmoid(output)


class PointNetAutoEncoder(nn.Module):
    def __init__(self, point_dim: int = 4, latent_dim: int = 128, num_points: int = 256):
        super().__init__()
        self.encoder = PointNetEncoder(point_dim=point_dim, latent_dim=latent_dim)
        self.decoder = PointCloudDecoder(latent_dim=latent_dim, num_points=num_points, point_dim=point_dim)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(points)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class PointNetPlusPlusAutoEncoder(nn.Module):
    def __init__(self, point_dim: int = 4, latent_dim: int = 128, num_points: int = 256):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(point_dim=point_dim, latent_dim=latent_dim)
        self.decoder = PointCloudDecoder(latent_dim=latent_dim, num_points=num_points, point_dim=point_dim)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(points)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def build_autoencoder(config: ModelConfig) -> nn.Module:
    name = config.name.lower()
    if name == "pointnet":
        return PointNetAutoEncoder(
            point_dim=config.point_dim,
            latent_dim=config.latent_dim,
            num_points=config.num_points,
        )
    if name in {"pointnet++", "pointnet2", "pointnetpp"}:
        return PointNetPlusPlusAutoEncoder(
            point_dim=config.point_dim,
            latent_dim=config.latent_dim,
            num_points=config.num_points,
        )
    raise ValueError(f"Unknown model '{config.name}'.")
