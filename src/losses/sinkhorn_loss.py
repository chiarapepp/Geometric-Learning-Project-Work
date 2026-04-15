import torch
from geomloss import SamplesLoss


def sinkhorn_loss(
    y_hat,
    y,
    p=2,
    blur=0.05,
    scaling=0.9,
    diameter=None,
    reduction="mean",
):
    """
    Sinkhorn loss for point clouds using geomloss.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        p (int): Ground cost exponent. Usually 1 or 2.
        blur (float): Entropic regularization scale.
        scaling (float): Tradeoff for Sinkhorn iterations speed/accuracy.
        diameter (float | None): Optional estimate of max cloud diameter.
        reduction (str): "mean", "sum", or "none"

    Returns:
        torch.Tensor:
            - scalar if reduction is "mean" or "sum"
            - tensor of shape (B,) if reduction is "none"
    """
    if y_hat.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"Expected tensors of shape (B, N, D), got {y_hat.shape} and {y.shape}"
        )

    if y_hat.shape[0] != y.shape[0]:
        raise ValueError(
            f"Batch sizes must match, got {y_hat.shape[0]} and {y.shape[0]}"
        )

    if y_hat.shape[2] != y.shape[2]:
        raise ValueError(
            f"Point dimensions must match, got {y_hat.shape[2]} and {y.shape[2]}"
        )

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
        diameter=diameter,
        backend="tensorized",
    )

    batch_losses = []
    for b in range(y_hat.shape[0]):
        cur_loss = loss_fn(y_hat[b], y[b])
        batch_losses.append(cur_loss)

    batch_losses = torch.stack(batch_losses)

    if reduction == "mean":
        return batch_losses.mean()
    elif reduction == "sum":
        return batch_losses.sum()
    elif reduction == "none":
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")