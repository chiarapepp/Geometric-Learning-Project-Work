import torch

from losses.chamfer_loss import chamfer_loss
from losses.density_aware_chamfer_loss import density_aware_chamfer_loss
from losses.emd_loss import emd_loss
from losses.sinkhorn_loss import sinkhorn_loss
from losses.temporal_weighted_chamfer_loss import temporal_weighted_chamfer_loss
from losses.hausdorff_loss import hausdorff_loss
from losses.projection_loss import projection_loss
from losses.voxel_loss import voxel_loss


def mse(y_hat, y):
    return torch.nn.MSELoss()(y_hat, y)


def l1(y_hat, y):
    return torch.nn.L1Loss()(y_hat, y)


def smooth_l1(y_hat, y):
    return torch.nn.SmoothL1Loss()(y_hat, y)


def cross_entropy(y_hat, y):
    return torch.nn.CrossEntropyLoss()(y_hat, y)


def get_loss(loss_name, **kwargs):
    """
    Return a loss function by name.

    Args:
        loss_name (str): Name of the loss.
        **kwargs: Optional parameters for configurable losses.

    Returns:
        callable: Loss function with signature loss_fn(y_hat, y)
    """
    loss_name = loss_name.lower()

    if loss_name == "chamfer":
        return chamfer_loss

    elif loss_name == "density_aware_chamfer":
        return lambda y_hat, y: density_aware_chamfer_loss(
            y_hat,
            y,
            alpha=kwargs.get("alpha", 1000),
            n_lambda=kwargs.get("n_lambda", 1),
            non_reg=kwargs.get("non_reg", False),
        )

    elif loss_name == "emd":
        return lambda y_hat, y: emd_loss(
            y_hat,
            y,
            reduction=kwargs.get("reduction", "mean"),
        )

    elif loss_name == "sinkhorn":
        return lambda y_hat, y: sinkhorn_loss(
            y_hat,
            y,
            p=kwargs.get("p", 2),
            blur=kwargs.get("blur", 0.05),
            scaling=kwargs.get("scaling", 0.9),
            diameter=kwargs.get("diameter", None),
            reduction=kwargs.get("reduction", "mean"),
        )

    elif loss_name == "temporal_weighted_chamfer":
        return lambda y_hat, y: temporal_weighted_chamfer_loss(
            y_hat,
            y,
            time_weight=kwargs.get("time_weight", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )

    elif loss_name == "hausdorff":
        return lambda y_hat, y: hausdorff_loss(
            y_hat,
            y,
            reduction=kwargs.get("reduction", "mean"),
            squared=kwargs.get("squared", False),
        )

    elif loss_name == "projection":
        return lambda y_hat, y: projection_loss(
            y_hat,
            y,
            grid_size=kwargs.get("grid_size", 32),
            sigma=kwargs.get("sigma", 0.04),
            reduction=kwargs.get("reduction", "mean"),
            normalize=kwargs.get("normalize", True),
            views=kwargs.get("views", None),
            loss_type=kwargs.get("loss_type", "mse"),
        )

    elif loss_name == "voxel":
        return lambda y_hat, y: voxel_loss(
            y_hat,
            y,
            grid_size=kwargs.get("grid_size", 32),
            sigma=kwargs.get("sigma", 0.04),
            reduction=kwargs.get("reduction", "mean"),
            normalize=kwargs.get("normalize", True),
            loss_type=kwargs.get("loss_type", "mse"),
            dims=kwargs.get("dims", (0, 1, 2)),
        )

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


def build_composite_loss(loss_name_list, loss_weights, loss_kwargs_list=None):
    """
    Build a weighted composite loss.

    Args:
        loss_name_list (list[str]): List of loss names.
        loss_weights (list[float]): Corresponding weights.
        loss_kwargs_list (list[dict] | None): Optional kwargs for each loss.

    Returns:
        callable: A function composite_loss(y_hat, y) -> (loss, loss_dict)
    """
    if len(loss_name_list) != len(loss_weights):
        raise ValueError("loss_name_list and loss_weights must have the same length.")

    if loss_kwargs_list is None:
        loss_kwargs_list = [{} for _ in loss_name_list]

    if len(loss_kwargs_list) != len(loss_name_list):
        raise ValueError("loss_kwargs_list must have the same length as loss_name_list.")

    loss_list = [
        get_loss(loss_name, **loss_kwargs)
        for loss_name, loss_kwargs in zip(loss_name_list, loss_kwargs_list)
    ]

    def composite_loss(y_hat, y):
        total_loss = 0.0
        loss_dict = {}

        for loss_name, loss_fn, weight in zip(loss_name_list, loss_list, loss_weights):
            cur_loss = loss_fn(y_hat, y)
            total_loss = total_loss + weight * cur_loss
            loss_dict[loss_name] = float(cur_loss.detach().item())

        return total_loss, loss_dict

    return composite_loss