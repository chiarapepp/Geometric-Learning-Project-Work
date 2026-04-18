from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlopEstimate:
    """
    Lightweight analytic FLOP estimate for one forward loss evaluation.

    The estimates intentionally focus on the dominant tensor operations. They
    are useful for comparing configurations with the same implementation, not
    for exact hardware accounting.
    """

    flops: int
    method: str
    batch_size: int

    @property
    def flops_per_sample(self) -> float:
        return float(self.flops) / float(max(self.batch_size, 1))

    @classmethod
    def make(cls, flops: int | float, method: str, batch_size: int) -> "FlopEstimate":
        return cls(flops=max(int(flops), 0), method=method, batch_size=int(batch_size))


def _point_shapes(y_hat: torch.Tensor, y: torch.Tensor) -> tuple[int, int, int, int]:
    if y_hat.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected tensors of shape (B, N, D), got {y_hat.shape} and {y.shape}")
    if y_hat.shape[0] != y.shape[0]:
        raise ValueError(f"Batch sizes must match, got {y_hat.shape[0]} and {y.shape[0]}")
    if y_hat.shape[2] != y.shape[2]:
        raise ValueError(f"Point dimensions must match, got {y_hat.shape[2]} and {y.shape[2]}")
    return int(y_hat.shape[0]), int(y_hat.shape[1]), int(y.shape[1]), int(y_hat.shape[2])


def _pairwise_euclidean_flops(batch_size: int, n_pred: int, n_target: int, point_dim: int) -> int:
    # Per pair: D subtractions, D multiplications, D - 1 additions, and one sqrt.
    return batch_size * n_pred * n_target * max(3 * point_dim, 1)


def _nearest_reduction_flops(batch_size: int, n_pred: int, n_target: int) -> int:
    pred_to_target = batch_size * n_pred * max(n_target - 1, 0)
    target_to_pred = batch_size * n_target * max(n_pred - 1, 0)
    return pred_to_target + target_to_pred


def _mean_reduction_flops(batch_size: int, n_pred: int, n_target: int) -> int:
    return batch_size * (max(n_pred - 1, 0) + max(n_target - 1, 0) + 1)


def _default_projection_views(point_dim: int) -> int:
    return 3 if point_dim >= 3 else 1


def estimate_loss_flops(
    loss_name: str,
    y_hat: torch.Tensor,
    y: torch.Tensor,
    loss_kwargs: dict | None = None,
) -> FlopEstimate:
    """
    Estimate FLOPs for one loss forward pass.

    Parameters
    ----------
    loss_name:
        Name accepted by ``src.losses.loss_factory.get_loss``.
    y_hat, y:
        Point clouds with shape ``(B, N, D)`` and ``(B, M, D)``.
    loss_kwargs:
        Optional loss parameters. Only parameters that affect operation counts
        are read, such as projection/voxel grid size and Sinkhorn iteration
        estimate.
    """
    loss_kwargs = loss_kwargs or {}
    name = loss_name.lower()
    batch_size, n_pred, n_target, point_dim = _point_shapes(y_hat, y)

    pairwise = _pairwise_euclidean_flops(batch_size, n_pred, n_target, point_dim)
    nearest = _nearest_reduction_flops(batch_size, n_pred, n_target)
    means = _mean_reduction_flops(batch_size, n_pred, n_target)

    if name == "chamfer":
        squared = bool(loss_kwargs.get("squared", True))
        square_cost = batch_size * n_pred * n_target if squared else 0
        flops = pairwise + square_cost + nearest + means
        return FlopEstimate.make(flops, "analytic_chamfer", batch_size)

    if name == "temporal_weighted_chamfer":
        scale_cost = batch_size * (n_pred + n_target) * point_dim
        square_cost = batch_size * n_pred * n_target
        flops = scale_cost + pairwise + square_cost + nearest + means
        return FlopEstimate.make(flops, "analytic_temporal_weighted_chamfer", batch_size)

    if name == "hausdorff":
        squared = bool(loss_kwargs.get("squared", False))
        square_cost = batch_size * n_pred * n_target if squared else 0
        max_cost = batch_size * (max(n_pred - 1, 0) + max(n_target - 1, 0) + 1)
        flops = pairwise + square_cost + nearest + max_cost
        return FlopEstimate.make(flops, "analytic_hausdorff", batch_size)

    if name == "density_aware_chamfer":
        square_cost = batch_size * n_pred * n_target
        exp_cost = 20 * batch_size * (n_pred + n_target)
        weighting_cost = 8 * batch_size * (n_pred + n_target)
        flops = pairwise + square_cost + nearest + exp_cost + weighting_cost + means
        return FlopEstimate.make(flops, "analytic_density_aware_chamfer", batch_size)

    if name == "emd":
        if n_pred != n_target:
            raise ValueError("Hungarian EMD FLOP estimate expects N == M.")
        assignment_cost = batch_size * (n_pred ** 3)
        matched_mean = batch_size * max(n_pred - 1, 0)
        flops = pairwise + assignment_cost + matched_mean
        return FlopEstimate.make(flops, "analytic_emd_with_cubic_assignment", batch_size)

    if name == "sinkhorn":
        iterations = int(loss_kwargs.get("sinkhorn_iterations_estimate", 50))
        cost_matrix = pairwise
        updates = iterations * batch_size * n_pred * n_target * 6
        reductions = iterations * batch_size * (n_pred + n_target)
        flops = cost_matrix + updates + reductions
        return FlopEstimate.make(flops, f"rough_sinkhorn_{iterations}_iterations", batch_size)

    if name == "projection":
        grid_size = int(loss_kwargs.get("grid_size", 32))
        views = loss_kwargs.get("views", None)
        num_views = len(views) if views is not None else _default_projection_views(point_dim)
        pixels = grid_size * grid_size
        # Per point/pixel projection: 2D squared distance, exp, max reduction.
        pred_projection = batch_size * n_pred * pixels * (8 + 20)
        target_projection = batch_size * n_target * pixels * (8 + 20)
        max_reduce = batch_size * num_views * pixels * (max(n_pred - 1, 0) + max(n_target - 1, 0))
        image_loss = batch_size * num_views * pixels * 3
        flops = num_views * (pred_projection + target_projection) + max_reduce + image_loss
        return FlopEstimate.make(flops, "analytic_soft_projection", batch_size)

    if name == "voxel":
        grid_size = int(loss_kwargs.get("grid_size", 32))
        voxels = grid_size ** 3
        pred_voxel = batch_size * n_pred * voxels * (11 + 20)
        target_voxel = batch_size * n_target * voxels * (11 + 20)
        max_reduce = batch_size * voxels * (max(n_pred - 1, 0) + max(n_target - 1, 0))
        voxel_loss_cost = batch_size * voxels * 3
        flops = pred_voxel + target_voxel + max_reduce + voxel_loss_cost
        return FlopEstimate.make(flops, "analytic_soft_voxel", batch_size)

    if name in {"mse", "l1", "smooth_l1"}:
        flops = batch_size * n_pred * point_dim * 3
        return FlopEstimate.make(flops, f"analytic_{name}", batch_size)

    return FlopEstimate.make(0, "unavailable", batch_size)
