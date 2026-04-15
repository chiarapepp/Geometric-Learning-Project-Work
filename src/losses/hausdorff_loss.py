import torch


def hausdorff_loss(y_hat, y, reduction="mean", squared=False):
    """
    Symmetric Hausdorff loss for point clouds.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        reduction (str): "mean", "sum", or "none"
        squared (bool): If True, use squared Euclidean distances

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

    y_hat = y_hat.float()
    y = y.float()

    dist = torch.cdist(y_hat, y, p=2)  # (B, N, M)
    if squared:
        dist = dist ** 2

    # Directed Hausdorff distances
    pred_to_gt = dist.min(dim=2)[0].max(dim=1)[0]  # (B,)
    gt_to_pred = dist.min(dim=1)[0].max(dim=1)[0]  # (B,)

    loss_per_batch = torch.maximum(pred_to_gt, gt_to_pred)

    if reduction == "mean":
        return loss_per_batch.mean()
    elif reduction == "sum":
        return loss_per_batch.sum()
    elif reduction == "none":
        return loss_per_batch
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")