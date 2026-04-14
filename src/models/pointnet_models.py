import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Simple PointNet-style encoder.

    Input:
        x: (B, N, D)

    Output:
        z: (B, latent_dim)
    """

    def __init__(self, input_dim: int = 3, latent_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, latent_dim, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (B, N, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got last dim={x.shape[-1]}"
            )

        x = x.transpose(1, 2)  # (B, D, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, latent_dim, N)

        x = torch.max(x, dim=2)[0]   # global max pooling -> (B, latent_dim)
        return x


class LinearDecoder(nn.Module):
    """
    Linear decoder baseline.
    """

    def __init__(self, latent_dim: int, output_dim: tuple[int, int]):
        super().__init__()
        self.latent_dim = latent_dim
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
        self.latent_dim = latent_dim
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


class PointNetAE(nn.Module):
    """
    PointNet-based autoencoder.

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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.output_dim = (num_points, input_dim)

        self.encoder = PointNetEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
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


class PointNetVAE(nn.Module):
    """
    PointNet-based variational autoencoder.

    Input:
        x: (B, N, D)

    Output:
        x_hat: (B, N, D)
        mu:    (B, latent_dim)
        logvar:(B, latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_points: int = 1024,
        latent_dim: int = 128,
        encoder_dim: int = 256,
        decoder_type: str = "mlp",
        decoder_hidden_dim: int = 512,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_points = num_points
        self.latent_dim = latent_dim
        self.encoder_dim = encoder_dim
        self.output_dim = (num_points, input_dim)

        self.encoder_backbone = PointNetEncoder(
            input_dim=input_dim,
            latent_dim=encoder_dim,
        )

        self.fc_mu = nn.Linear(encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim, latent_dim)

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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar