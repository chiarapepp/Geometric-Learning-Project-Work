from datasets.base_datasets import DatasetNpy
import os
from glob import glob
import numpy as np
import utils.data_utils as duts


class NCARSDataset(DatasetNpy):
    def __init__(self, data_path, train, data_dim=(40, 80, 2), N=-1, stride=-1, use_polarity=True, transform=None,
                 normalizer=None):
        super(NCARSDataset, self).__init__(data_path, train, data_dim, N, stride, use_polarity, transform, normalizer)

        self.classes = ['background', 'cars']

    def get_file_list(self):
        return glob(os.path.join(self.data_path, '*/*.npy'))

    def get_item(self, idx):
        file_idx, start_idx, end_idx = self.video_chunks_ids[idx]
        target = (self.file_list[file_idx].split('/')[-2])
        target = (self.classes.index(target))
        data = duts.read_npy_chunk(self.file_list[file_idx], start_idx, end_idx - start_idx).astype(
            np.float32)  # x, y, p, t
        data = data[:, [0, 1, 3, 2]]  # x, y, t, p
        return data, target