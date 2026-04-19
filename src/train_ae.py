import os
import argparse
import csv
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.datasets.dataset_factory import build_pointcloud_transform, get_dataset
from src.models.pointnet_models import PointNetAE, PointNetVAE
from src.models.pointnetpp_models import PointNetPPAE

from src.losses.loss_factory import get_loss
from src.train_ae_functions import do_epoch_ae
from src.utils import ensure_dir, set_seed
from src.wandb_util import WandbHandler


@dataclass
class Config:
    dataset: str = "dvsgesture"
    save_to: str = "./data"

    train_batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 50
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_name: str = "pointnet_ae"   # pointnet_ae | pointnet_vae | pointnetpp_ae
    input_dim: int = 4                # (x, y, t, p) oppure 3
    num_points: int = 1024
    latent_dim: int = 256
    decoder_hidden_dim: int = 512
    temporal_weight: float = 1.0

    loss_name: str = "chamfer"
    loss_time_weight: float = 1.0
    kl_weight: float = 1e-4

    wandb: str = "disabled"             # online | disabled
    wandb_project: str = "geometric-learning-project"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_tags: list[str] | None = None
    log_every: int = 1

    output_dir: str = "./outputs/autoencoder"
    save_every: int = 0
    resume_from: str | None = None
    seed: int = 13
    split_ratio: float = 0.8
    split_seed: int = 13


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
    parser = argparse.ArgumentParser(description="Train PointNet/PointNet++ AE or VAE on neuromorphic point clouds.")
    parser.add_argument("--dataset", default=Config.dataset, choices=["dvsgesture", "nmnist", "ncaltech101"])
    parser.add_argument("--save-to", default=Config.save_to)
    parser.add_argument("--model-name", default=Config.model_name, choices=["pointnet_ae", "pointnet_vae", "pointnetpp_ae"])
    parser.add_argument("--loss-name", default=Config.loss_name)
    parser.add_argument("--num-points", type=int, default=Config.num_points)
    parser.add_argument("--input-dim", type=int, default=Config.input_dim, choices=[3, 4])
    parser.add_argument("--temporal-weight", type=float, default=Config.temporal_weight)
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Config.train_batch_size)
    parser.add_argument("--test-batch-size", type=int, default=Config.test_batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--decoder-hidden-dim", type=int, default=Config.decoder_hidden_dim)
    parser.add_argument("--loss-time-weight", type=float, default=Config.loss_time_weight)
    parser.add_argument("--kl-weight", type=float, default=Config.kl_weight)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--wandb", default=Config.wandb, choices=["online", "disabled"])
    parser.add_argument("--wandb-project", default=Config.wandb_project)
    parser.add_argument("--wandb-entity", default=Config.wandb_entity)
    parser.add_argument("--wandb-run-name", default=Config.wandb_run_name)
    parser.add_argument("--wandb-group", default=Config.wandb_group)
    parser.add_argument("--wandb-job-type", default=Config.wandb_job_type)
    parser.add_argument("--wandb-tags", nargs="*", default=Config.wandb_tags)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--save-every", type=int, default=Config.save_every)
    parser.add_argument("--resume-from", default=Config.resume_from)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--split-ratio", type=float, default=Config.split_ratio)
    parser.add_argument("--split-seed", type=int, default=Config.split_seed)
    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        save_to=args.save_to,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_name=args.model_name,
        input_dim=args.input_dim,
        num_points=args.num_points,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        temporal_weight=args.temporal_weight,
        loss_name=args.loss_name,
        loss_time_weight=args.loss_time_weight,
        kl_weight=args.kl_weight,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_tags=args.wandb_tags,
        output_dir=args.output_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        seed=args.seed,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
    )
    if cfg.wandb_run_name is None:
        cfg.wandb_run_name = f"convergence_{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}"
    if cfg.wandb_group is None:
        cfg.wandb_group = "autoencoder_convergence"
    if cfg.wandb_job_type is None:
        cfg.wandb_job_type = "train_ae"

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    logger = WandbHandler(cfg)
    transform = build_pointcloud_transform(
        cfg.dataset,
        num_points=cfg.num_points,
        input_dim=cfg.input_dim,
        temporal_weight=cfg.temporal_weight,
    )

    train_dataset = get_dataset(
        dataset_name=cfg.dataset,
        save_to=cfg.save_to,
        train=True,
        transform=transform,
        split_ratio=cfg.split_ratio,
        split_seed=cfg.split_seed,
    )

    val_dataset = get_dataset(
        dataset_name=cfg.dataset,
        save_to=cfg.save_to,
        train=False,
        transform=transform,
        split_ratio=cfg.split_ratio,
        split_seed=cfg.split_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, model_type = build_model(cfg)
    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_kwargs = {}
    if cfg.loss_name == "temporal_weighted_chamfer":
        loss_kwargs["time_weight"] = cfg.loss_time_weight
    loss_fn = get_loss(cfg.loss_name, **loss_kwargs)

    best_val = float("inf")
    best_checkpoint_path = None
    start_epoch = 0
    if cfg.resume_from is not None:
        checkpoint = torch.load(cfg.resume_from, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))
        best_checkpoint_path = cfg.resume_from
        print(f"Resumed from {cfg.resume_from} at epoch {start_epoch}")

    history_path = os.path.join(
        cfg.output_dir,
        f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}_history.csv",
    )
    history_rows = []
    if cfg.resume_from is not None and os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))

    logger.log({
        "model/num_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.perf_counter()
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
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "dataset": cfg.dataset,
            "model": cfg.model_name,
            "loss_name": cfg.loss_name,
            "train_loss": train_metrics.get("loss"),
            "val_loss": val_metrics.get("loss"),
            "train_recon_loss": train_metrics.get("recon_loss"),
            "val_recon_loss": val_metrics.get("recon_loss"),
            "train_kl_loss": train_metrics.get("kl_loss"),
            "val_kl_loss": val_metrics.get("kl_loss"),
            "epoch_seconds": epoch_seconds,
            "num_points": cfg.num_points,
            "input_dim": cfg.input_dim,
            "temporal_weight": cfg.temporal_weight,
            "loss_time_weight": cfg.loss_time_weight,
        }
        history_rows.append(row)
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        logger.log({
            "epoch": epoch,
            "train/epoch_seconds": epoch_seconds,
        })

        if current_val < best_val:
            best_val = current_val
            save_path = os.path.join(
                cfg.output_dir,
                f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}_best.pth",
            )
            best_checkpoint_path = save_path
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
            logger.log({
                "epoch": epoch,
                "val/best_loss": best_val,
            })

        if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
            checkpoint_path = os.path.join(
                cfg.output_dir,
                f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}_epoch_{epoch + 1}.pth",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "config": cfg.__dict__,
                },
                checkpoint_path,
            )
            print(f"Saved periodic checkpoint to {checkpoint_path}")
            logger.log_checkpoint(
                checkpoint_path,
                artifact_name=f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}_epoch_{epoch + 1}",
            )

    logger.log_artifact(history_path, artifact_type="training-history")
    if best_checkpoint_path is not None:
        logger.log_checkpoint(
            best_checkpoint_path,
            artifact_name=f"{cfg.dataset}_{cfg.model_name}_{cfg.loss_name}_best",
        )

    logger.finish()


if __name__ == "__main__":
    main()
