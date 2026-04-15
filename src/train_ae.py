import os
import torch
from torch.utils.data import DataLoader

from datasets.dataset_factory import get_dataset
from models.pointnet_models import PointNetAE, PointNetVAE
from models.pointnetpp_models import PointNetPPAE

from losses.loss_factory import get_loss
from train_ae_functions import do_epoch_ae
from utils.wandb_handler import WandbHandler


class Config:
    dataset = "dvsgesture"
    save_to = "./data"

    train_batch_size = 16
    test_batch_size = 16
    num_workers = 4
    num_epochs = 50
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "pointnet_ae"   # pointnet_ae | pointnet_vae | pointnetpp_ae
    input_dim = 4                # (x, y, t, p) oppure 3
    num_points = 1024
    latent_dim = 256
    decoder_hidden_dim = 512

    loss_name = "chamfer"
    kl_weight = 1e-4

    wandb = "online"             # online | disabled
    wandb_project = "geometric-learning-project"
    wandb_entity = None
    log_every = 1

    output_dir = "./outputs"


def build_model(cfg):
    if cfg.model_name == "pointnet_ae":
        return PointNetAE(
            input_dim=cfg.input_dim,
            num_points=cfg.num_points,
            latent_dim=cfg.latent_dim,
            decoder_type="mlp",
            decoder_hidden_dim=cfg.decoder_hidden_dim,
        ), "ae"

    elif cfg.model_name == "pointnet_vae":
        return PointNetVAE(
            input_dim=cfg.input_dim,
            num_points=cfg.num_points,
            latent_dim=128,
            encoder_dim=cfg.latent_dim,
            decoder_type="mlp",
            decoder_hidden_dim=cfg.decoder_hidden_dim,
        ), "vae"

    elif cfg.model_name == "pointnetpp_ae":
        return PointNetPPAE(
            input_dim=cfg.input_dim,
            num_points=cfg.num_points,
            latent_dim=cfg.latent_dim,
            decoder_type="mlp",
            decoder_hidden_dim=cfg.decoder_hidden_dim,
        ), "ppae"

    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = WandbHandler(cfg)

    train_dataset = get_dataset(
        dataset_name=cfg.dataset,
        save_to=cfg.save_to,
        train=True,
        transform=None,
    )

    val_dataset = get_dataset(
        dataset_name=cfg.dataset,
        save_to=cfg.save_to,
        train=False,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model, model_type = build_model(cfg)
    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = get_loss(cfg.loss_name)

    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        train_metrics = do_epoch_ae(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=cfg.device,
            epoch=epoch,
            train=True,
            logger=logger,
            model_type=model_type,
            kl_weight=cfg.kl_weight,
        )

        val_metrics = do_epoch_ae(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=cfg.device,
            epoch=epoch,
            train=False,
            logger=logger,
            model_type=model_type,
            kl_weight=cfg.kl_weight,
        )

        print(f"Epoch {epoch}")
        print("train:", train_metrics)
        print("val:  ", val_metrics)

        current_val = val_metrics["loss"]
        if current_val < best_val:
            best_val = current_val
            save_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "config": cfg.__dict__,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

    logger.finish()


if __name__ == "__main__":
    main()