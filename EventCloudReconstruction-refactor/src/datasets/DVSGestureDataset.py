from datasets.base_datasets import DatasetNpy
import os
from glob import glob
import numpy as np
import utils.data_utils as duts


class DVSGestureDataset(DatasetNpy):
    """
    Dataset for DVS Gesture Recognition.
    Can be downloaded from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8
    """
    def __init__(self, data_path, train, data_dim=(128,128,2), N=-1, stride=-1, use_polarity=True, transform=None):
        super(DVSGestureDataset, self).__init__(data_path, train, data_dim, N, stride, use_polarity, transform)

    def get_file_list(self):
        return glob(os.path.join(self.data_path, '*/*.npy'))
    
    def get_item(self, idx):
        file_idx, start_idx, end_idx = self.video_chunks_ids[idx]
        target = int(self.file_list[file_idx].split('/')[-1].split('.')[0])
        data = duts.read_npy_chunk(self.file_list[file_idx], start_idx, end_idx - start_idx).astype(np.float32) # x, y, p, t
        data = data[:, [0, 1, 3, 2]] # x, y, t, p
        return data, target