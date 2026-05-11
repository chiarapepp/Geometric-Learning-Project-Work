import torch
import numpy as np
from models.decoder.base_decoder import BaseDecoder

class LinearDecoder(BaseDecoder):
    """
    Linear Decoder for reconstructing data from a latent representation.
    """
    def __init__(self, latent_size, output_dim):
        super(LinearDecoder, self).__init__(latent_size, output_dim)
        self.output_size = np.prod(output_dim)
        self.linear = torch.nn.Linear(latent_size, self.output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).view(-1, *self.output_dim)