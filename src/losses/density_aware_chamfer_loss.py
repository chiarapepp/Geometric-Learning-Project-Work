import torch
from chamferdist import knn_points


def density_aware_chamfer_loss(y_hat, y, alpha=1000, n_lambda=1, non_reg=False):
    """
    Density-aware Chamfer Distance (DCD).

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, 3).
        y (torch.Tensor): Ground-truth points, shape (B, M, 3).
        alpha (float): Exponential scaling factor applied to distances.
        n_lambda (float): Exponent used in frequency-based reweighting.
        non_reg (bool): If True, use max(1, ratio) for cardinality compensation.

    Returns:
        torch.Tensor: Scalar DCD loss.
    """
    y_hat = y_hat.float()
    y = y.float()

    batch_size, n_x, _ = y_hat.shape
    _, n_gt, _ = y.shape
    assert batch_size == y.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    # gt -> pred
    knn_1 = knn_points(y, y_hat, K=1)
    dist1 = knn_1.dists.squeeze(-1)   # (B, n_gt)
    idx1 = knn_1.idx.squeeze(-1)      # (B, n_gt)

    # pred -> gt
    knn_2 = knn_points(y_hat, y, K=1)
    dist2 = knn_2.dists.squeeze(-1)   # (B, n_x)
    idx2 = knn_2.idx.squeeze(-1)      # (B, n_x)

    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    count1 = torch.zeros(
        batch_size, n_x, device=y_hat.device, dtype=idx1.dtype
    )
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6).pow(-1) * frac_21
    loss1 = (1.0 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros(
        batch_size, n_gt, device=y_hat.device, dtype=idx2.dtype
    )
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6).pow(-1) * frac_12
    loss2 = (1.0 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2
    return loss.mean() 