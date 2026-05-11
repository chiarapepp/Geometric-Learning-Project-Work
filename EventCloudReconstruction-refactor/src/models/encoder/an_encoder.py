import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder.base_encoder import BaseEncoder
from einops import rearrange

class AnEncoder(BaseEncoder):
    """
    It's just an encoder, no idea what it should do yet. :'(
    """
    def __init__(self, latent_size, input_dim, reduce_max=True):
        super(AnEncoder, self).__init__(input_dim, latent_size)

        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        self.reduce_max = reduce_max
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.latent_size, nhead=8), num_layers=1)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.latent_size, nhead=8), num_layers=1)
        self.transformer_encoder2 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.latent_size, nhead=8), num_layers=1)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, self.latent_size))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, self.latent_size))

    def forward(self, x):
        x = x.permute(0, 2, 1) # B x channels x seq_len
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # num chunks
        num_chunks = 16

        # cut to valid chunks
        valid_chunks = x.size(2) // num_chunks
        x = x[:, :, :valid_chunks * num_chunks]

        # random ids
        random_ids = torch.randperm(x.size(2))

        # shuffle samples
        x = x[:, :, random_ids]
        x = rearrange(x, 'B C (CHUNKS SEQLEN) -> SEQLEN (B CHUNKS) C', CHUNKS=num_chunks)

        x = torch.cat([self.cls_token1.repeat(1, x.size(1), 1), x], dim=0)
        x = self.transformer_encoder(x)

        x = rearrange(x[0], '(B CHUNKS) C -> CHUNKS B C', CHUNKS=num_chunks)

        x = torch.cat([self.cls_token2.repeat(1, x.size(1), 1), x], dim=0)

        x = self.transformer_encoder2(x)
        x = x[0, :, :]

        return x #, torch.max(x_upper, 2, keepdim=True)[0].view(-1, self.latent_size), torch.max(x_lower, 2, keepdim=True)[0].view(-1, self.latent_size)