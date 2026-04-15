import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances.

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        dist: (B, N, M)
    """
    if src.ndim != 3 or dst.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got {src.shape} and {dst.shape}")
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Batch-aware point indexing.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    if points.ndim != 3:
        raise ValueError(f"Expected points shape (B, N, C), got {points.shape}")

    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS), pure PyTorch version.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint)
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape (B, N, 3), got {xyz.shape}")

    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    KNN grouping by Euclidean distance.

    Args:
        k: number of neighbors
        xyz: (B, N, 3)
        new_xyz: (B, S, 3)

    Returns:
        group_idx: (B, S, k)
    """
    if xyz.ndim != 3 or new_xyz.ndim != 3:
        raise ValueError(f"Expected xyz/new_xyz to be 3D, got {xyz.shape}, {new_xyz.shape}")

    B, N, _ = xyz.shape
    k = min(k, N)

    dist = square_distance(new_xyz, xyz)  # (B, S, N)
    _, group_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return group_idx


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query in pure PyTorch.
    If fewer than nsample points are found within radius, indices are padded
    by repeating the first valid index.

    Args:
        radius: neighborhood radius
        nsample: max number of neighbors
        xyz: (B, N, 3)
        new_xyz: (B, S, 3)

    Returns:
        group_idx: (B, S, nsample)
    """
    if xyz.ndim != 3 or new_xyz.ndim != 3:
        raise ValueError(f"Expected xyz/new_xyz to be 3D, got {xyz.shape}, {new_xyz.shape}")

    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    nsample = min(nsample, N)

    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)

    mask = sqrdists > radius * radius
    group_idx[mask] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    first_group = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    invalid_mask = group_idx == N
    group_idx[invalid_mask] = first_group[invalid_mask]

    return group_idx


class SharedMLP2d(nn.Module):
    """
    Shared MLP implemented as stacked 1x1 Conv2d layers.
    Input is expected in shape (B, C, K, S) or similar 4D format.
    """

    def __init__(self, channels: list[int], bn: bool = True):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least input and output size")

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    """
    Simplified PointNet++ Set Abstraction block.

    Pipeline:
    - sample centroids with FPS
    - group local neighborhoods (KNN or ball query)
    - concatenate relative xyz with point features
    - apply shared MLP
    - max-pool over neighborhood dimension
    """

    def __init__(
        self,
        npoint: int,
        nsample: int,
        in_channel: int,
        mlp_channels: list[int],
        group_type: str = "knn",
        radius: float | None = None,
        use_xyz: bool = True,
        bn: bool = True,
    ):
        super().__init__()

        if group_type not in {"knn", "ball"}:
            raise ValueError("group_type must be 'knn' or 'ball'")
        if group_type == "ball" and radius is None:
            raise ValueError("radius must be provided when group_type='ball'")

        self.npoint = npoint
        self.nsample = nsample
        self.group_type = group_type
        self.radius = radius
        self.use_xyz = use_xyz

        mlp_in = in_channel + 3 if use_xyz else in_channel
        self.mlp = SharedMLP2d([mlp_in] + mlp_channels, bn=bn)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C) or None

        Returns:
            new_xyz: (B, S, 3)
            new_points: (B, S, C_out)
        """
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz shape (B, N, 3), got {xyz.shape}")

        B, N, _ = xyz.shape
        S = min(self.npoint, N)

        fps_idx = farthest_point_sample(xyz, S)         # (B, S)
        new_xyz = index_points(xyz, fps_idx)            # (B, S, 3)

        if self.group_type == "knn":
            group_idx = knn_point(self.nsample, xyz, new_xyz)  # (B, S, K)
        else:
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        grouped_xyz = index_points(xyz, group_idx)      # (B, S, K, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, group_idx)   # (B, S, K, C)
            if self.use_xyz:
                grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_features = grouped_points
        else:
            grouped_features = grouped_xyz_norm

        # (B, S, K, C_total) -> (B, C_total, K, S)
        grouped_features = grouped_features.permute(0, 3, 2, 1).contiguous()

        new_points = self.mlp(grouped_features)         # (B, C_out, K, S)
        new_points = torch.max(new_points, dim=2)[0]    # (B, C_out, S)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B, S, C_out)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Simplified Feature Propagation block.

    Interpolates coarse features onto finer points using nearest neighbors,
    then refines with an MLP.
    """

    def __init__(self, in_channel: int, mlp_channels: list[int], bn: bool = True):
        super().__init__()
        self.mlp = SharedMLP2d([in_channel] + mlp_channels, bn=bn)

    def forward(
        self,
        unknown_xyz: torch.Tensor,
        known_xyz: torch.Tensor,
        unknown_points: torch.Tensor | None,
        known_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            unknown_xyz: (B, N, 3) finer points
            known_xyz: (B, S, 3) coarser points
            unknown_points: (B, N, C1) or None
            known_points: (B, S, C2)

        Returns:
            new_points: (B, N, C_out)
        """
        if known_xyz is None or known_points is None:
            raise ValueError("known_xyz and known_points must not be None")

        dists = square_distance(unknown_xyz, known_xyz)              # (B, N, S)
        k = min(3, known_xyz.shape[1])
        dists, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=False)

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_points = torch.sum(
            index_points(known_points, idx) * weight.unsqueeze(-1),
            dim=2,
        )  # (B, N, C2)

        if unknown_points is not None:
            new_points = torch.cat([interpolated_points, unknown_points], dim=-1)
        else:
            new_points = interpolated_points

        # (B, N, C) -> (B, C, N, 1)
        new_points = new_points.permute(0, 2, 1).unsqueeze(-1).contiguous()
        new_points = self.mlp(new_points)                           # (B, C_out, N, 1)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        return new_points