from collections import defaultdict
from tqdm import tqdm
import torch


def do_epoch_ae(
    loader,
    model,
    loss_fn,
    optimizer,
    device,
    epoch,
    train=True,
    logger=None,
    model_type="ae",
    kl_weight=1e-4,
):
    """
    Run one epoch for AE/VAE reconstruction.

    Args:
        loader: DataLoader
        model: reconstruction model
        loss_fn: callable, usually from get_loss(...) or build_composite_loss(...)
        optimizer: torch optimizer
        device: torch device
        epoch: current epoch
        train: bool
        logger: WandbHandler or None
        model_type: "ae", "vae", or "ppae"
        kl_weight: KL weight for VAE

    Returns:
        dict with averaged metrics
    """
    if train:
        model.train()
    else:
        model.eval()

    metrics_epoch = defaultdict(float)

    for i, batch in enumerate(tqdm(loader, desc=f"{'Train' if train else 'Val'} Epoch {epoch}")):
        # expected batch format: (events, target) or similar
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                data, _ = batch
            elif len(batch) == 3:
                _, data, _ = batch
            else:
                data = batch[0]
        else:
            data = batch

        if data.ndim == 4:
            data = data.squeeze(1)

        data = data.to(device).float()

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            if model_type == "vae":
                output, mu, logvar = model(data)
                recon_loss = loss_fn(output, data)
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = recon_loss + kl_weight * kl_loss

                metrics_epoch["recon_loss"] += recon_loss.item()
                metrics_epoch["kl_loss"] += kl_loss.item()
            else:
                output = model(data)
                loss = loss_fn(output, data)

            metrics_epoch["loss"] += loss.item()

            if train:
                loss.backward()
                optimizer.step()

        if logger is not None and i == 0:
            global_step = epoch * len(loader) + i
            logger.log({
                f"{'train' if train else 'val'}/loss_step": loss.item(),
                "epoch": epoch,
            })

    num_batches = len(loader)
    metrics_epoch = {k: v / num_batches for k, v in metrics_epoch.items()}

    if logger is not None:
        for k, v in metrics_epoch.items():
            logger.log({
                f"{'train' if train else 'val'}/{k}": v,
                "epoch": epoch,
            })

    return metrics_epoch