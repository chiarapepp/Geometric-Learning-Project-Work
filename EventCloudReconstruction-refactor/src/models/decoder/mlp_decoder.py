import torch
import numpy as np
from models.decoder.base_decoder import BaseDecoder

class MLPDecoder(BaseDecoder):
    """
    MLP Decoder for reconstructing data from a latent representation.
    """
    def __init__(self, latent_size, output_dim):
        super(MLPDecoder, self).__init__(latent_size, output_dim)
        self.output_size = np.prod(output_dim)
        self.linear1 = torch.nn.Linear(latent_size, latent_size*2)
        self.linear2 = torch.nn.Linear(latent_size*2, self.output_size)

    def forward(self, x):
        # relu
        x = torch.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x)).view(-1, *self.output_dim)
        # return self.linear2(x).view(-1, *self.output_dim)