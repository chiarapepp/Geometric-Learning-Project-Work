from datasets.base_datasets import DatasetNpy
import torch
import os
from glob import glob
import numpy as np
import utils.data_utils as duts
from tqdm import tqdm
import socket
from torchvision import transforms


class MICCGestureDataset(DatasetNpy):
    """
    This dataset is only used for testing purposes.
    """
    def __init__(self, data_path, train, data_dim=(1280,720,2), N=-1, stride=1, use_polarity=True, transform=None):
        assert(train == False)
        super(MICCGestureDataset, self).__init__(data_path, train, data_dim, N, stride, use_polarity, transform)

    def get_file_list(self):
        return glob(os.path.join(self.data_path, '*/*.npy'))
    
    def get_item(self, idx):
        file_idx, start_idx, end_idx = self.video_chunks_ids[idx]
        target = int(self.file_list[file_idx].split('_')[-1].split('.')[0])
        data = duts.read_npy_chunk(self.file_list[file_idx], start_idx, end_idx - start_idx).astype(np.float32) # x, y, p, t
        data = data[:, [0, 1, 3, 2]] # x, y, t, p
        return data, target
    