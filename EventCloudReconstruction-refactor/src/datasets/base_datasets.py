import torch
import os
from glob import glob
import numpy as np
import utils.data_utils as duts
from tqdm import tqdm
import socket
from torchvision import transforms


class DatasetNpy(torch.utils.data.Dataset):
    """
    This is a dataset for event camera data stored in multiple npy files.
    When slicing is enabled, the whole dataset will be composed of different samples, obtained by listing all the segments of length N (number of events) in all the files.
    """

    def __init__(self, data_path, train, data_dim=(128,128,2), N=-1, stride=-1, use_polarity=True, transform=None):
        """
        Args:
            data_path (str): path to the directory containing the npy files
            train (bool): whether to load the training or the test set
            data_dim (tuple): dimensions of the data
            N (int): number of events to load. If -1, the whole video will be loaded
            stride (int): stride to use when slicing the video (active only when N > 0)
            transform (torchvision.transforms): transformation to apply to the data

        This class should be modified only if strictly necessary.
        Extend it to adapt it to different datasets that use npy files.
        """
        self.train = train
        self.data_path = data_path
        self.use_polarity = use_polarity

        self.N = N
        if stride == -1:
            self.stride = N
        else:
            self.stride = stride
        self.data = []
        self.file_list = self.get_file_list()
        self.video_chunks_ids = self.prepare_chunk_ids()
        self.transform = None                                             
        self.dimensions = data_dim
        self.cache = {}
        self.video_targets = [int(f.split('/')[-1].split('.')[-2].split('_')[-1]) for f in self.file_list]
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        
    def get_file_list(self):
        """
        Load the list of files in the dataset and store it in a list of paths.
        """
        pass

    def prepare_chunk_ids(self):
        """
        This function will return a list of tuples, each one containing the file index and the starting index of the chunk.
        """
        chunk_ids = []
        self.num_events_per_video = []
        for i in tqdm(range(len(self.file_list))):
            n = duts.get_len_without_loading(self.file_list[i])
            self.num_events_per_video.append(n)
            if self.N == -1:
                chunk_ids.append((i, 0, n))
            for j in range(0, n-self.N+1, self.stride):
                chunk_ids.append((i, j, j+self.N))
        return chunk_ids
    
    def __len__(self):
        return len(self.video_chunks_ids)
    
    def __getitem__(self, idx):
        """
        This function will load the idx-th chunk of data.
        Data is always x, y, t, p
        """
        if idx in self.cache:
            data, target = self.cache[idx]
        else:
            data, target = self.get_item(idx)
            if not self.use_polarity:
                data = data[:, [0, 1, 2]]
            self.cache[idx] = data, target

        if self.transform:
            data = self.transform(data)
        return idx, data, target

if __name__ == "__main__":
    print('main')
    d = DatasetNpy('/media/becattini/SSD4TB/datasets/DVSGesture/ibmGestureTrain', True, N=5000)

    print(len(d))
    print(d)