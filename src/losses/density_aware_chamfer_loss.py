import torch


def density_aware_chamfer_loss(
    y_hat,
    y,
    alpha=1000,
    n_lambda=1,
    non_reg=False,
    reduction="mean",
):
    """
    Density-aware Chamfer Distance (DCD) implemented with pure PyTorch.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        alpha (float): Exponential scaling factor
        n_lambda (float): Exponent for frequency-based reweighting
        non_reg (bool): If True, use max(1, ratio) for cardinality compensation
        reduction (str): "mean", "sum", or "none"

    Returns:
        torch.Tensor
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

    batch_size, n_x, _ = y_hat.shape
    _, n_gt, _ = y.shape
    device = y_hat.device

    if non_reg:
        frac_12 = max(1.0, n_x / n_gt)
        frac_21 = max(1.0, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    # Pairwise squared distances: (B, N, M)
    dist_mat = torch.cdist(y_hat, y, p=2) ** 2

    # gt -> pred
    dist1, idx1 = dist_mat.transpose(1, 2).min(dim=2)  # (B, M), (B, M)

    # pred -> gt
    dist2, idx2 = dist_mat.min(dim=2)  # (B, N), (B, N)

    exp_dist1 = torch.exp(-alpha * dist1)
    exp_dist2 = torch.exp(-alpha * dist2)

    # Count how many gt points map to each pred point
    count1 = torch.zeros(batch_size, n_x, device=device, dtype=y_hat.dtype)
    count1.scatter_add_(1, idx1, torch.ones_like(dist1, dtype=y_hat.dtype))
    weight1 = count1.gather(1, idx1).detach().pow(n_lambda)
    weight1 = (weight1 + 1e-6).pow(-1) * frac_21
    loss1 = (1.0 - exp_dist1 * weight1).mean(dim=1)

    # Count how many pred points map to each gt point
    count2 = torch.zeros(batch_size, n_gt, device=device, dtype=y_hat.dtype)
    count2.scatter_add_(1, idx2, torch.ones_like(dist2, dtype=y_hat.dtype))
    weight2 = count2.gather(1, idx2).detach().pow(n_lambda)
    weight2 = (weight2 + 1e-6).pow(-1) * frac_12
    loss2 = (1.0 - exp_dist2 * weight2).mean(dim=1)

    loss = 0.5 * (loss1 + loss2)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")