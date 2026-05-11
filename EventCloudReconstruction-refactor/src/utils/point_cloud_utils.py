import torch
import numpy as np

def fast_soft_project_to_image_batched(points, plane_normal, plane_offset, image_size=(128, 128), sigma=0.1):
    """
    Fast differentiable projection of 3D points onto a 2D plane using vectorized soft rasterization.

    Args:
    - points (torch.Tensor): (N, 3) 3D points.
    - plane_normal (torch.Tensor): (3,) Normal vector of the plane.
    - plane_offset (torch.Tensor): Scalar, offset in the plane equation.
    - image_size (tuple): (H, W), size of the output image.
    - sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    - image (torch.Tensor): (H, W) Differentiable soft occupancy map.
    """
    bs = points.shape[0]
    device = points.device

    # Normalize the normal vector
    plane_normal = plane_normal / plane_normal.norm(p=2)

    # Project points onto the plane
    signed_distance = torch.sum(points * plane_normal, dim=2, keepdim=True) + plane_offset
    points = points - signed_distance * plane_normal  # (N, 3)

    # Define a 2D coordinate system
    if torch.abs(plane_normal[2]) < 0.9:
        u = torch.cross(plane_normal, torch.tensor([0, 0, 1.0], dtype=torch.float32, device=device))
    else:
        u = torch.cross(plane_normal, torch.tensor([1.0, 0, 0], dtype=torch.float32, device=device))
    u /= u.norm(p=2)
    v = torch.cross(plane_normal, u)
    v /= v.norm(p=2)

    # Convert projected 3D points into 2D coordinates
    points = torch.stack([
        torch.sum(points * u, dim=2),
        torch.sum(points * v, dim=2)
    ], dim=2)  # (N, 2)

    # Normalize and scale to image space
    min_xy = points.min(dim=1)[0].unsqueeze(1)
    max_xy = points.max(dim=1)[0].unsqueeze(1)
    points = (points - min_xy) / (max_xy - min_xy + 1e-6)
    points = points * torch.tensor((image_size[0]-1, image_size[1]-1), dtype=torch.float32, device=device)

    # Generate pixel grid (no need to store full grid, compute on-the-fly)
    H, W = image_size
    x_grid = torch.linspace(0, W - 1, W, device=device).half()
    y_grid = torch.linspace(0, H - 1, H, device=device).half() # origin in top left

    chunk_size = -1
    
    if chunk_size == -1:
        chunk_dist_sq = (points[:, :, 0].view(bs, -1, 1, 1).half() - x_grid.view(-1, 1)).pow(2) + \
        (points[:, :, 1].view(bs, -1, 1, 1).half() - y_grid.view(1, -1)).pow(2)
        image = torch.exp(-chunk_dist_sq / (2 * sigma**2)).sum(dim=1)
    else:
        image = torch.zeros(bs, H, W, device=device)
        for i in range(0, points.shape[1], chunk_size):
            chunk = points[:, i:i+chunk_size]        
            chunk_dist_sq = (chunk[:, :, 0].view(bs, -1, 1, 1).half() - x_grid.view(-1, 1)).pow(2) + \
            (chunk[:, :, 1].view(bs, -1, 1, 1).half() - y_grid.view(1, -1)).pow(2)  # (N, H, W)
            image = image + torch.exp(-chunk_dist_sq / (2 * sigma**2)).sum(dim=1)

    return image / image.max()  # Normalize image intensity


def to_frame(cloud, frame_size, rescale=True):
    """
    Converts a point cloud to a 2D frame representation in a NON-differentiable way.
    Args:
    - cloud (np.ndarray): Point cloud of shape (N, 3) where each point is (x, y, t).
    - frame_size (tuple): Size of the output frame (height, width, channels).
    - rescale (bool): If True, rescales the time values to [0, 1] range.
    """
    t_min = cloud[:, 2].min()
    t_max = cloud[:, 2].max()
    cloud = cloud[np.argsort(cloud[:, 2])]
    frame = np.zeros(frame_size)
    for point in cloud:
        if rescale:
            frame[round(point[1]*(frame_size[1]-1)), round(point[0]*(frame_size[0]-1)), 2] = (point[2] - t_min) / (t_max - t_min)
            frame[round(point[1]*(frame_size[1]-1)), round(point[0]*(frame_size[0]-1)), 1] = 1 - ((point[2] - t_min) / (t_max - t_min))
        else:
            frame[int(point[1]), int(point[0]), 2] = (point[2] - t_min) / (t_max - t_min)
        frame[int(point[1]), int(point[0]), 1] = 1 - ((point[2] - t_min) / (t_max - t_min))
    return frame