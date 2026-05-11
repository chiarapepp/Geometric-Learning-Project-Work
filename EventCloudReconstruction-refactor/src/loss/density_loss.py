from chamferdist.chamfer import knn_points
import torch

def density_loss(y_hat, y, K=20):
    """
    Density loss for 3D point clouds.
    Args:
        y_hat (torch.Tensor): Predicted points, shape (batch_size, num_points, 3).
        y (torch.Tensor): Ground truth points, shape (batch_size, num_points, 3).
        K (int): Number of nearest neighbors to consider for density estimation.
    """

    # find the K nearest neighbors
    lengths1 = torch.tensor([len(x) for x in y_hat]).to(y.device)
    lengths2 = torch.tensor([len(x) for x in y]).to(y.device)
    y_hat_nearest = knn_points(y_hat, y, lengths1, lengths2, K)
    y_nearest = knn_points(y, y_hat, lengths1, lengths2, K)

    density1 = y_hat_nearest[0].mean(-1)
    density2 = y_nearest[0].mean(-1)

    d_loss = torch.mean((density1 - density2) ** 2)

    return d_loss
