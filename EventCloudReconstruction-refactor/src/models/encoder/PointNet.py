import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder.base_encoder import BaseEncoder

class PointNetEncoder(BaseEncoder):
    """
    PointNet encoder. Disable 'reduce_max' to avoid the max pooling operation and obtain a per-point latent representation.
    """
    def __init__(self, latent_size, input_dim, reduce_max=True):
        super(PointNetEncoder, self).__init__(input_dim, latent_size)

        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        self.reduce_max = reduce_max

    def forward(self, x):
        x = x.permute(0, 2, 1) # B x channels x seq_len
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.reduce_max:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, self.latent_size)
        return x