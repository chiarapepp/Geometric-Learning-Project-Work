import torch
import torch.nn.functional as F


def _normalize_minmax(points, eps=1e-8):
    """
    Normalize each batch independently in [0, 1].
    Args:
        points: (B, N, 2)
    Returns:
        (B, N, 2)
    """
    mins = points.min(dim=1, keepdim=True)[0]
    maxs = points.max(dim=1, keepdim=True)[0]
    scale = (maxs - mins).clamp_min(eps)
    return (points - mins) / scale


def _soft_project(points_2d, grid_size=32, sigma=0.04):
    """
    Soft rasterization of 2D points into an occupancy map.

    Args:
        points_2d: (B, N, 2), assumed normalized in [0, 1]
        grid_size: output image size
        sigma: gaussian kernel width in normalized coordinates

    Returns:
        proj: (B, 1, H, W)
    """
    device = points_2d.device
    dtype = points_2d.dtype
    B, N, _ = points_2d.shape

    xs = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    grid = torch.stack([xx, yy], dim=-1)            # (H, W, 2)
    grid = grid.view(1, 1, grid_size, grid_size, 2) # (1, 1, H, W, 2)

    pts = points_2d.view(B, N, 1, 1, 2)             # (B, N, 1, 1, 2)
    sqdist = ((pts - grid) ** 2).sum(dim=-1)        # (B, N, H, W)

    proj = torch.exp(-sqdist / (2 * sigma * sigma)) # (B, N, H, W)
    proj = proj.max(dim=1, keepdim=True)[0]         # (B, 1, H, W)

    return proj.clamp(0.0, 1.0)


def _project_view(points, dims, grid_size=32, sigma=0.04, normalize=True):
    """
    Project 3D/4D/... points to a 2D plane using selected coordinates.

    Args:
        points: (B, N, D)
        dims: tuple(int, int), coordinates to keep
    """
    proj_2d = points[:, :, list(dims)]
    if normalize:
        proj_2d = _normalize_minmax(proj_2d)
    return _soft_project(proj_2d, grid_size=grid_size, sigma=sigma)


def projection_loss(
    y_hat,
    y,
    grid_size=32,
    sigma=0.04,
    reduction="mean",
    normalize=True,
    views=None,
    loss_type="mse",
):
    """
    Multi-view projection loss for point clouds.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        grid_size (int): Size of the soft projection image
        sigma (float): Gaussian kernel width for soft projection
        reduction (str): "mean", "sum", or "none"
        normalize (bool): Whether to normalize each 2D projection in [0, 1]
        views (list[tuple[int, int]] | None): list of 2D projections to use.
            Default for D >= 3: [(0,1), (0,2), (1,2)]
        loss_type (str): "mse", "l1", or "bce"

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

    if y_hat.shape[2] < 2:
        raise ValueError("projection_loss requires point_dim >= 2")

    if views is None:
        if y_hat.shape[2] >= 3:
            views = [(0, 1), (0, 2), (1, 2)]
        else:
            views = [(0, 1)]

    valid_losses = {"mse", "l1", "bce"}
    if loss_type not in valid_losses:
        raise ValueError(f"loss_type must be one of {valid_losses}, got {loss_type}")

    y_hat = y_hat.float()
    y = y.float()

    per_view_losses = []

    for dims in views:
        pred_proj = _project_view(
            y_hat,
            dims=dims,
            grid_size=grid_size,
            sigma=sigma,
            normalize=normalize,
        )
        gt_proj = _project_view(
            y,
            dims=dims,
            grid_size=grid_size,
            sigma=sigma,
            normalize=normalize,
        )

        if loss_type == "mse":
            cur = F.mse_loss(pred_proj, gt_proj, reduction="none")
        elif loss_type == "l1":
            cur = F.l1_loss(pred_proj, gt_proj, reduction="none")
        else:
            cur = F.binary_cross_entropy(pred_proj, gt_proj, reduction="none")

        cur = cur.flatten(start_dim=1).mean(dim=1)  # (B,)
        per_view_losses.append(cur)

    loss_per_batch = torch.stack(per_view_losses, dim=0).mean(dim=0)  # (B,)

    if reduction == "mean":
        return loss_per_batch.mean()
    elif reduction == "sum":
        return loss_per_batch.sum()
    elif reduction == "none":
        return loss_per_batch
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")