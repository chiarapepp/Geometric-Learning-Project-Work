import torch


def temporal_weighted_chamfer_loss(
    y_hat,
    y,
    time_weight=1.0,
    reduction="mean",
):
    """
    Temporal-weighted Chamfer loss for point clouds.

    Useful when points are represented as (x, y, t) or (x, y, t, ...),
    and the temporal axis should have a different importance.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        time_weight (float): Weight applied to the temporal dimension.
                             Assumes time is the 3rd coordinate (index 2).
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

    if y_hat.shape[2] < 3:
        raise ValueError(
            "Temporal-weighted Chamfer requires at least 3 coordinates per point "
            "(expected time at index 2)."
        )

    if time_weight <= 0:
        raise ValueError("time_weight must be > 0")

    y_hat = y_hat.float()
    y = y.float()

    weights = torch.ones(y_hat.shape[2], device=y_hat.device, dtype=y_hat.dtype)
    weights[2] = time_weight

    y_hat_w = y_hat * weights.view(1, 1, -1)
    y_w = y * weights.view(1, 1, -1)

    dist = torch.cdist(y_hat_w, y_w, p=2) ** 2  # (B, N, M)

    min_pred_to_gt = dist.min(dim=2)[0]  # (B, N)
    min_gt_to_pred = dist.min(dim=1)[0]  # (B, M)

    loss_per_batch = min_pred_to_gt.mean(dim=1) + min_gt_to_pred.mean(dim=1)

    if reduction == "mean":
        return loss_per_batch.mean()
    elif reduction == "sum":
        return loss_per_batch.sum()
    elif reduction == "none":
        return loss_per_batch
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")