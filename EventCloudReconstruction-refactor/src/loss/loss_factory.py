import torch
from loss.chamfer_loss import chamfer_loss
from loss.projection_loss import projection_loss
from loss.density_loss import density_loss


def get_loss(loss_name):
    """
    Factory function to get the loss function by name.
    Args:
        loss_name (str): Name of the loss function.
    """
    if loss_name == 'chamfer':
        return chamfer_loss
    elif loss_name == 'projection':
        return projection_loss
    elif loss_name == 'density':
        return density_loss
    elif loss_name == 'mse':
        return mse
    elif loss_name == 'l1':
        return l1
    elif loss_name == 'smooth_l1':
        return smooth_l1
    elif loss_name == 'cross_entropy':
        return cross_entropy
    else:
        raise NotImplementedError(f'Loss {loss_name} not implemented')

def build_composite_loss(loss_name_list, loss_weights):
    """
    Factory function to build a composite loss function from a list of loss names and their corresponding weights.
    Args:
        loss_name_list (list of str): List of loss function names.
        loss_weights (list of float): List of weights corresponding to each loss function.
    """
    def composite_loss(y_hat, y):
        loss = 0
        loss_dict = {}
        for loss_fn, weight in zip(loss_list, loss_weights):
            cur_loss = loss_fn(y_hat, y)
            loss_dict[loss_fn.__name__] = cur_loss.item()
            loss += weight*cur_loss
        return loss, loss_dict
    loss_list = [get_loss(loss_name) for loss_name in loss_name_list]
    return composite_loss


def mse(y_hat, y):
    return torch.nn.MSELoss()(y_hat, y)

def l1(y_hat, y):
    return torch.nn.L1Loss()(y_hat, y)

def smooth_l1(y_hat, y):
    return torch.nn.SmoothL1Loss()(y_hat, y)

def cross_entropy(y_hat, y):
    return torch.nn.CrossEntropyLoss()(y_hat, y)