import torch


def chamfer_loss(y_hat, y, squared=True, reduction="mean"):
    """
    Symmetric Chamfer distance implemented with torch.cdist.

    Args:
        y_hat: Predicted points, shape (B, N, D).
        y: Ground-truth points, shape (B, M, D).
        squared: If True use squared Euclidean distances.
        reduction: "mean", "sum", or "none".
    """
    if y_hat.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"Expected tensors of shape (B, N, D), got {y_hat.shape} and {y.shape}"
        )
    if y_hat.shape[0] != y.shape[0]:
        raise ValueError(f"Batch sizes must match, got {y_hat.shape[0]} and {y.shape[0]}")
    if y_hat.shape[2] != y.shape[2]:
        raise ValueError(f"Point dimensions must match, got {y_hat.shape[2]} and {y.shape[2]}")

    distances = torch.cdist(y_hat, y, p=2)
    if squared:
        distances = distances.pow(2)

    pred_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_pred = distances.min(dim=1).values.mean(dim=1)
    losses = pred_to_target + target_to_pred

    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"Unknown reduction '{reduction}'")
