import torch
from losses.chamfer_loss import chamfer_loss
# from losses.emd_loss import emd_loss
# from losses.density_aware_chamfer_loss import density_aware_chamfer_loss
# from losses.projection_loss import projection_loss
# from losses.voxel_loss import voxel_loss
# from losses.sinkhorn_loss import sinkhorn_loss


def mse(y_hat, y):
    return torch.nn.MSELoss()(y_hat, y)


def l1(y_hat, y):
    return torch.nn.L1Loss()(y_hat, y)


def smooth_l1(y_hat, y):
    return torch.nn.SmoothL1Loss()(y_hat, y)


def cross_entropy(y_hat, y):
    return torch.nn.CrossEntropyLoss()(y_hat, y)


def get_loss(loss_name):
    """
    Return a loss function by name.

    Args:
        loss_name (str): Name of the loss.

    Returns:
        callable: Loss function.
    """
    if loss_name == "chamfer":
        return chamfer_loss
    # elif loss_name == "emd":
    #     return emd_loss
    # elif loss_name == "density_aware_chamfer":
    #     return density_aware_chamfer_loss
    # elif loss_name == "projection":
    #     return projection_loss
    # elif loss_name == "voxel":
    #     return voxel_loss
    # elif loss_name == "sinkhorn":
    #     return sinkhorn_loss
    elif loss_name == "mse":
        return mse
    elif loss_name == "l1":
        return l1
    elif loss_name == "smooth_l1":
        return smooth_l1
    elif loss_name == "cross_entropy":
        return cross_entropy
    else:
        raise NotImplementedError(f"Loss '{loss_name}' not implemented.")


def build_composite_loss(loss_name_list, loss_weights):
    """
    Build a weighted composite loss.

    Args:
        loss_name_list (list[str]): List of loss names.
        loss_weights (list[float]): Corresponding weights.

    Returns:
        callable: A function composite_loss(y_hat, y) -> (loss, loss_dict)
    """
    if len(loss_name_list) != len(loss_weights):
        raise ValueError("loss_name_list and loss_weights must have the same length.")

    loss_list = [get_loss(loss_name) for loss_name in loss_name_list]

    def composite_loss(y_hat, y):
        total_loss = 0.0
        loss_dict = {}

        for loss_name, loss_fn, weight in zip(loss_name_list, loss_list, loss_weights):
            cur_loss = loss_fn(y_hat, y)
            total_loss = total_loss + weight * cur_loss
            loss_dict[loss_name] = cur_loss.item()

        return total_loss, loss_dict

    return composite_loss