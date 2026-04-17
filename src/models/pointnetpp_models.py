import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnetpp_utils import PointNetSetAbstraction


class LinearDecoder(nn.Module):
    """
    Linear decoder baseline.
    """

    def __init__(self, latent_dim: int, output_dim: tuple[int, int]):
        super().__init__()
        self.output_dim = output_dim
        self.output_size = int(np.prod(output_dim))
        self.linear = nn.Linear(latent_dim, self.output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.linear(z)
        return x_hat.view(-1, *self.output_dim)


class MLPDecoder(nn.Module):
    """
    MLP decoder for point cloud reconstruction.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: tuple[int, int],
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.output_size = int(np.prod(output_dim))

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        x_hat = self.fc3(z)
        return x_hat.view(-1, *self.output_dim)


class PointNetPPEncoder(nn.Module):
    """
    Simplified PointNet++ encoder with hierarchical local aggregation.

    Input:
        x: (B, N, D), with first 3 dims interpreted as xyz

    Output:
        z: (B, latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 256,
        sa1_npoint: int = 256,
        sa1_nsample: int = 32,
        sa2_npoint: int = 64,
        sa2_nsample: int = 32,
        group_type: str = "knn",
        sa1_radius: float = 0.1,
        sa2_radius: float = 0.2,
    ):
        super().__init__()

        if input_dim < 3:
            raise ValueError("PointNetPPEncoder requires input_dim >= 3")

        extra_dim = input_dim - 3

        self.sa1 = PointNetSetAbstraction(
            npoint=sa1_npoint,
            nsample=sa1_nsample,
            in_channel=extra_dim,
            mlp_channels=[64, 64, 128],
            group_type=group_type,
            radius=sa1_radius,
            use_xyz=True,
            bn=True,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=sa2_npoint,
            nsample=sa2_nsample,
            in_channel=128,
            mlp_channels=[128, 128, 256],
            group_type=group_type,
            radius=sa2_radius,
            use_xyz=True,
            bn=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, N, D), got {tuple(x.shape)}")
        if x.shape[-1] < 3:
            raise ValueError("Input must contain at least xyz coordinates")

        xyz = x[:, :, :3]
        points = x[:, :, 3:] if x.shape[-1] > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, points)     # (B, S1, 3), (B, S1, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, S2, 3), (B, S2, 256)

        z = torch.max(l2_points, dim=1)[0]            # (B, 256)
        z = self.fc(z)                                # (B, latent_dim)
        return z


class PointNetPPAE(nn.Module):
    """
    PointNet++-style autoencoder.

    Input:
        x: (B, N, D)

    Output:
        x_hat: (B, N, D)
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_points: int = 1024,
        latent_dim: int = 256,
        decoder_type: str = "mlp",
        decoder_hidden_dim: int = 512,
        sa1_npoint: int = 256,
        sa1_nsample: int = 32,
        sa2_npoint: int = 64,
        sa2_nsample: int = 32,
        group_type: str = "knn",
        sa1_radius: float = 0.1,
        sa2_radius: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.output_dim = (num_points, input_dim)

        self.encoder = PointNetPPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            sa1_npoint=sa1_npoint,
            sa1_nsample=sa1_nsample,
            sa2_npoint=sa2_npoint,
            sa2_nsample=sa2_nsample,
            group_type=group_type,
            sa1_radius=sa1_radius,
            sa2_radius=sa2_radius,
        )

        if decoder_type == "linear":
            self.decoder = LinearDecoder(
                latent_dim=latent_dim,
                output_dim=self.output_dim,
            )
        elif decoder_type == "mlp":
            self.decoder = MLPDecoder(
                latent_dim=latent_dim,
                output_dim=self.output_dim,
                hidden_dim=decoder_hidden_dim,
            )
        else:
            raise ValueError(
                f"Unknown decoder_type '{decoder_type}'. Use 'linear' or 'mlp'."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
