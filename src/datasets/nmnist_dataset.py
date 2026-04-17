from collections.abc import Callable
import numpy as np

from .base_tonic_dataset import BaseTonicDataset


class NMNISTDataset(BaseTonicDataset):
    sensor_size = (34, 34, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        first_saccade_only: bool = False,
        stabilize: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ):
        super().__init__(
            save_to=save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.train = train
        self.first_saccade_only = first_saccade_only
        self.stabilize = stabilize

        import tonic

        self.dataset = tonic.datasets.NMNIST(
            save_to=save_to,
            train=train,
            first_saccade_only=first_saccade_only,
            stabilize=stabilize,
        )

        self.data = list(range(len(self.dataset)))
        self.targets = None

    def __getitem__(self, index):
        events, target = self.dataset[index]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        return events, target

    def __len__(self):
        return len(self.dataset)

    def _check_exists(self):
        return True
