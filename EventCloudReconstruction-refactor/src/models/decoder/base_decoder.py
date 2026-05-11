import torch

class BaseDecoder(torch.nn.Module):
    """
    Base class for decoders in the model. It defines the basic structure and properties that all decoders should have.
    """
    def __init__(self, latent_size, output_dim):
        super(BaseDecoder, self).__init__()
        self.latent_size = latent_size
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError("Forward function not implemented, must be implemented in subclass")