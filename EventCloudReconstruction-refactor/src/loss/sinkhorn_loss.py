import torch
from geomloss import SamplesLoss

def sinkhorn(x: torch.Tensor, y: torch.Tensor):
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    L = loss(x, y)
    return L.mean()