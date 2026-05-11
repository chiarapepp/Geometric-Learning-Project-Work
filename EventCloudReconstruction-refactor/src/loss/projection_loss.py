from utils.point_cloud_utils import fast_soft_project_to_image_batched
import torch

def projection_loss(y_hat, y, image_size=(128//10, 128//10), loss_type='l2', sigma=0.1):
    """
    Projection loss for 3D point clouds.
    Args:
        y_hat (torch.Tensor): Predicted points, shape (batch_size, num_points, 3).
        y (torch.Tensor): Ground truth points, shape (batch_size, num_points, 3).
        image_size (tuple): Size of the output image (height, width).
        loss_type (str): Type of loss to compute ('l1' or 'l2').
        sigma (float): Standard deviation for Gaussian kernel in projection.
    """
    # Define the projection plane
    #plane_normal = torch.tensor([0, 0, 1.0], dtype=torch.float32, device=y.device)
    # random plane
    plane_normal = torch.randn(3, device=y.device)
    plane_offset = 0.0

    y_concat = torch.cat([y_hat, y], dim=0)

    # Project the point clouds
    # y_hat_image = fast_soft_project_to_image_batched(y_hat, plane_normal, plane_offset, image_size, sigma)
    # y_image = fast_soft_project_to_image_batched(y, plane_normal, plane_offset, image_size, sigma)
    y_concat_image = fast_soft_project_to_image_batched(y_concat, plane_normal, plane_offset, image_size, sigma)
    y_hat_image = y_concat_image[:y_hat.shape[0]]
    y_image = y_concat_image[y_hat.shape[0]:]

    # Compute the L1 loss
    if loss_type == 'l1':
        loss = torch.mean(torch.abs(y_hat_image - y_image))
    elif loss_type == 'l2':
        loss = torch.mean((y_hat_image - y_image) ** 2)
    else:
        raise ValueError("Unsupported loss type. Use 'l1' or 'l2'.")

    return loss