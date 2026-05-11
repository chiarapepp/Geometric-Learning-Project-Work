from datasets.DVSGestureDataset import DVSGestureDataset
from datasets.MICCGestureDataset import MICCGestureDataset
from datasets.NCARSDataset import NCARSDataset
from datasets.NCaltech101Dataset import NCaltech101Dataset
import os


def get_dataset(dataset_name, train, N, stride, use_polarity=True, transform=None):
    """
    Factory function to get the dataset based on the dataset name.
    Args:
        dataset_name (str): name of the dataset to load
        train (bool): whether to load the training or the test set
        N (int): number of events to load. If -1, the whole video will be loaded
        stride (int): stride to use when slicing the video (active only when N > 0)
        use_polarity (bool): whether to use polarity information
        transform (callable, optional): transformation to apply to the data
    """
    print(f'Loading dataset {dataset_name}... Split: {"train" if train else "test"} with N={N} and stride={stride}')
    if dataset_name == 'DVSGesture':
        data_path = 'data/DVSGesture/'
        data_dim = (128, 128, 2)
        if train:
            data_path = os.path.join(data_path, 'ibmGestureTrain')
        else:
            data_path = os.path.join(data_path, 'ibmGestureTest')
        return DVSGestureDataset(data_path, train, data_dim, N, stride, use_polarity, transform)
    elif dataset_name == 'MICCGesture':
        data_path = 'data/MICC-event-gesture-dataset'
        data_dim = (1280, 720, 2)
        if train:
            raise ValueError('MICCGestureDataset is only used for testing purposes.')
        return MICCGestureDataset(data_path, train, data_dim, N, stride, transform)
    elif dataset_name == 'NCARS':
        data_path = 'data/NCARS'
        data_path = os.path.join(data_path, 'n-cars_train' if train else 'n-cars_test')
        dimensions = (100, 120, 2)
        return NCARSDataset(data_path, train, dimensions, N, stride, use_polarity, transform)
    elif dataset_name == 'NCaltech101':
        data_path = 'data/NCALTECH101/Caltech101'
        dimensions =  (240, 320, 2)
        return NCaltech101Dataset(data_path, train, dimensions, N, stride, use_polarity, transform)
    else:
        raise ValueError(f'Dataset {dataset_name} not found.')