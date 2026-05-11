import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    """
    Base class for encoders in the model. It defines the basic structure and properties that all encoders should have.
    Encoders should inherit from this class and implement the forward method.
    """
    def __init__(self, input_dim, latent_size):
        super(BaseEncoder, self).__init__()
        self.input_dim = input_dim # input
        self.latent_size = latent_size # encoder output
        
    def forward(self):
        raise NotImplementedError("Forward function not implemented, must be implemented in subclass")
