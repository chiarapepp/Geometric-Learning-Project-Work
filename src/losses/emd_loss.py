import torch
from scipy.optimize import linear_sum_assignment


def emd_loss(y_hat, y, reduction="mean"):
    """
    Earth Mover's Distance (EMD) approximation via Hungarian matching.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, N, D)
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

    if y_hat.shape != y.shape:
        raise ValueError(
            f"y_hat and y must have the same shape, got {y_hat.shape} and {y.shape}"
        )

    batch_size, num_points, dim = y_hat.shape

    losses = []

    for b in range(batch_size):
        pred = y_hat[b]   # (N, D)
        target = y[b]     # (N, D)

        # Pairwise distance matrix: (N, N)
        cost_matrix = torch.cdist(pred, target, p=2)

        # Solve the Hungarian assignment on the CPU.
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

        row_ind = torch.as_tensor(row_ind, device=y_hat.device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind, device=y_hat.device, dtype=torch.long)

        matched_cost = cost_matrix[row_ind, col_ind].mean()
        losses.append(matched_cost)

    losses = torch.stack(losses)  # (B,)

    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "none":
        return losses
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")