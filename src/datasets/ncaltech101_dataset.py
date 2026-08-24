from collections.abc import Callable
import numpy as np

from .base_tonic_dataset import BaseTonicDataset


class NCaltech101Dataset(BaseTonicDataset):
    sensor_size = None
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        split: str | None = None,
        split_ratio: float = 0.8,
        split_seed: int = 13,
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

        import tonic

        self.dataset = tonic.datasets.NCALTECH101(save_to=save_to)
        split = split or ("train" if train else "test")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.train = split
        self.split_ratio = split_ratio
        self.split_seed = split_seed

        all_indices = np.arange(len(self.dataset))
        rng = np.random.default_rng(split_seed)
        shuffled_indices = rng.permutation(all_indices)
        train_end = int(round(split_ratio * len(shuffled_indices)))
        val_end = int(round((split_ratio + (1.0 - split_ratio) / 2) * len(shuffled_indices)))
        ranges = {"train": shuffled_indices[:train_end], "val": shuffled_indices[train_end:val_end], "test": shuffled_indices[val_end:]}
        self.indices = np.sort(ranges[split])

        self.data = self.indices.tolist()
        if getattr(self.dataset, "targets", None) is not None:
            self.targets = [self.dataset.targets[i] for i in self.indices]
        else:
            self.targets = None

    def __getitem__(self, index):
        dataset_index = int(self.indices[index])
        events, target = self.dataset[dataset_index]

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
