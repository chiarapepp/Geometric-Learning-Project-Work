from chamferdist import ChamferDistance

def chamfer_loss(y_hat, y):
    """
    Chamfer loss function using ChamferDistance from chamferdist library.
    Args:
        y_hat (torch.Tensor): Predicted points, shape (batch_size, num_points, 3).
        y (torch.Tensor): Ground truth points, shape (batch_size, num_points, 3).
    """
    chamfer_dist = ChamferDistance()
    loss = chamfer_dist(y_hat, y, bidirectional=True, point_reduction = "mean")
    return loss