from collections.abc import Callable
import numpy as np
import tonic

from .base_tonic_dataset import BaseTonicDataset


class DVSGestureDataset(BaseTonicDataset):
    sensor_size = (128, 128, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
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

        self.dataset = tonic.datasets.DVSGesture(save_to=save_to, train=train)

        # Per coerenza con la base class, manteniamo data/targets
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