import torch
import torch.nn.functional as F


def _normalize_minmax_3d(points, eps=1e-8):
    """
    Normalize each batch independently in [0, 1].

    Args:
        points: (B, N, 3)

    Returns:
        (B, N, 3)
    """
    mins = points.min(dim=1, keepdim=True)[0]
    maxs = points.max(dim=1, keepdim=True)[0]
    scale = (maxs - mins).clamp_min(eps)
    return (points - mins) / scale


def _soft_voxelize(points, grid_size=32, sigma=0.04):
    """
    Soft voxelization of 3D points into an occupancy volume.

    Args:
        points: (B, N, 3), assumed normalized in [0, 1]
        grid_size: voxel grid size
        sigma: gaussian kernel width in normalized coordinates

    Returns:
        voxels: (B, 1, G, G, G)
    """
    device = points.device
    dtype = points.dtype
    B, N, _ = points.shape

    coords = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)              # (G, G, G, 3)
    grid = grid.view(1, 1, grid_size, grid_size, grid_size, 3)

    pts = points.view(B, N, 1, 1, 1, 3)                   # (B, N, 1, 1, 1, 3)
    sqdist = ((pts - grid) ** 2).sum(dim=-1)              # (B, N, G, G, G)

    voxels = torch.exp(-sqdist / (2 * sigma * sigma))     # (B, N, G, G, G)
    voxels = voxels.max(dim=1, keepdim=True)[0]           # (B, 1, G, G, G)

    return voxels.clamp(0.0, 1.0)


def voxel_loss(
    y_hat,
    y,
    grid_size=32,
    sigma=0.04,
    reduction="mean",
    normalize=True,
    loss_type="mse",
    dims=(0, 1, 2),
):
    """
    Soft voxel loss for point clouds.

    Args:
        y_hat (torch.Tensor): Predicted points, shape (B, N, D)
        y (torch.Tensor): Ground-truth points, shape (B, M, D)
        grid_size (int): Voxel grid size
        sigma (float): Gaussian kernel width for soft voxelization
        reduction (str): "mean", "sum", or "none"
        normalize (bool): Whether to normalize selected coordinates in [0, 1]
        loss_type (str): "mse", "l1", or "bce"
        dims (tuple[int, int, int]): Which 3 coordinates to use for voxelization

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

    if len(dims) != 3:
        raise ValueError("dims must contain exactly 3 coordinate indices")

    if max(dims) >= y_hat.shape[2]:
        raise ValueError(
            f"dims={dims} incompatible with point dimension {y_hat.shape[2]}"
        )

    valid_losses = {"mse", "l1", "bce"}
    if loss_type not in valid_losses:
        raise ValueError(f"loss_type must be one of {valid_losses}, got {loss_type}")

    y_hat = y_hat.float()[:, :, list(dims)]
    y = y.float()[:, :, list(dims)]

    if normalize:
        y_hat = _normalize_minmax_3d(y_hat)
        y = _normalize_minmax_3d(y)

    pred_vox = _soft_voxelize(y_hat, grid_size=grid_size, sigma=sigma)
    gt_vox = _soft_voxelize(y, grid_size=grid_size, sigma=sigma)

    if loss_type == "mse":
        loss = F.mse_loss(pred_vox, gt_vox, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(pred_vox, gt_vox, reduction="none")
    else:
        loss = F.binary_cross_entropy(pred_vox, gt_vox, reduction="none")

    loss_per_batch = loss.flatten(start_dim=1).mean(dim=1)  # (B,)

    if reduction == "mean":
        return loss_per_batch.mean()
    elif reduction == "sum":
        return loss_per_batch.sum()
    elif reduction == "none":
        return loss_per_batch
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")