import torch

from models.pointnet_models import PointNetAE, PointNetVAE
from models.pointnetpp_models import PointNetPPAE


def test_pointnet_ae(device="cpu"):
    print("\n===== Testing PointNetAE =====")
    batch_size = 4
    num_points = 1024
    input_dim = 3

    x = torch.randn(batch_size, num_points, input_dim, device=device)

    model = PointNetAE(
        input_dim=input_dim,
        num_points=num_points,
        latent_dim=256,
        decoder_type="mlp",
    ).to(device)

    x_hat = model(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(x_hat.shape)}")

    assert x_hat.shape == x.shape, "PointNetAE output shape mismatch"

    loss = ((x_hat - x) ** 2).mean()
    loss.backward()

    print(f"loss: {loss.item():.6f}")
    print("OK: PointNetAE forward/backward works.")


def test_pointnet_vae(device="cpu"):
    print("\n===== Testing PointNetVAE =====")
    batch_size = 4
    num_points = 1024
    input_dim = 3
    latent_dim = 128

    x = torch.randn(batch_size, num_points, input_dim, device=device)

    model = PointNetVAE(
        input_dim=input_dim,
        num_points=num_points,
        latent_dim=latent_dim,
        encoder_dim=256,
        decoder_type="mlp",
    ).to(device)

    x_hat, mu, logvar = model(x)

    print(f"input shape:   {tuple(x.shape)}")
    print(f"output shape:  {tuple(x_hat.shape)}")
    print(f"mu shape:      {tuple(mu.shape)}")
    print(f"logvar shape:  {tuple(logvar.shape)}")

    assert x_hat.shape == x.shape, "PointNetVAE output shape mismatch"
    assert mu.shape == (batch_size, latent_dim), "PointNetVAE mu shape mismatch"
    assert logvar.shape == (batch_size, latent_dim), "PointNetVAE logvar shape mismatch"

    recon_loss = ((x_hat - x) ** 2).mean()
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + 1e-3 * kl_loss
    loss.backward()

    print(f"recon_loss: {recon_loss.item():.6f}")
    print(f"kl_loss:    {kl_loss.item():.6f}")
    print(f"total_loss: {loss.item():.6f}")
    print("OK: PointNetVAE forward/backward works.")


def test_pointnetpp_ae(device="cpu"):
    print("\n===== Testing PointNetPPAE =====")
    batch_size = 2
    num_points = 1024
    input_dim = 3

    x = torch.randn(batch_size, num_points, input_dim, device=device)

    model = PointNetPPAE(
        input_dim=input_dim,
        num_points=num_points,
        latent_dim=256,
        decoder_type="mlp",
        sa1_npoint=256,
        sa1_nsample=32,
        sa2_npoint=64,
        sa2_nsample=32,
        group_type="knn",
    ).to(device)

    x_hat = model(x)

    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(x_hat.shape)}")

    assert x_hat.shape == x.shape, "PointNetPPAE output shape mismatch"

    loss = ((x_hat - x) ** 2).mean()
    loss.backward()

    print(f"loss: {loss.item():.6f}")
    print("OK: PointNetPPAE forward/backward works.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(42)

    test_pointnet_ae(device=device)
    test_pointnet_vae(device=device)
    test_pointnetpp_ae(device=device)

    print("\nAll model tests passed.")


if __name__ == "__main__":
    main()