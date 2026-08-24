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
        split: str | None = None,
        split_seed: int = 13,
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
        split = split or ("train" if train else "test")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.train = split != "test"
        self.first_saccade_only = first_saccade_only
        self.stabilize = stabilize

        import tonic

        self.dataset = tonic.datasets.NMNIST(
            save_to=save_to,
            train=self.train,
            first_saccade_only=first_saccade_only,
            stabilize=stabilize,
        )

        all_indices = np.arange(len(self.dataset))
        if split in {"train", "val"}:
            shuffled = np.random.default_rng(split_seed).permutation(all_indices)
            cut = int(round(0.8 * len(shuffled)))
            self.indices = np.sort(shuffled[:cut] if split == "train" else shuffled[cut:])
        else:
            self.indices = all_indices
        self.data = self.indices.tolist()
        self.targets = None

    def __getitem__(self, index):
        events, target = self.dataset[int(self.indices[index])]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        return events, target

    def __len__(self):
        return len(self.indices)

    def _check_exists(self):
        return True
