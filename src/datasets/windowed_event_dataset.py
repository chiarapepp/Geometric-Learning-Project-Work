import numpy as np
from torch.utils.data import Dataset


class WindowedEventDataset(Dataset):
    """
    Slice raw event-stream samples into event-count windows.

    This mirrors the reference EventCloudReconstruction-style ``N``/``stride``
    loading while keeping the underlying dataset backend, e.g. tonic.
    """

    def __init__(
        self,
        dataset,
        window_size: int,
        stride: int | None = None,
        transform=None,
        target_transform=None,
        transforms=None,
        drop_last: bool = True,
        sort_by_time: bool = True,
        max_windows_per_sample: int | None = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride is None:
            stride = window_size
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if max_windows_per_sample is not None and max_windows_per_sample <= 0:
            raise ValueError("max_windows_per_sample must be > 0 when provided")

        self.dataset = dataset
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.drop_last = drop_last
        self.sort_by_time = sort_by_time
        self.max_windows_per_sample = max_windows_per_sample
        self.windows = self._build_windows()

    def _build_windows(self):
        windows = []
        for sample_idx in range(len(self.dataset)):
            events, _ = self.dataset[sample_idx]
            num_events = len(events)
            sample_windows = self._windows_for_length(num_events)
            if self.max_windows_per_sample is not None:
                sample_windows = sample_windows[: self.max_windows_per_sample]
            windows.extend((sample_idx, start, end) for start, end in sample_windows)
        return windows

    def _windows_for_length(self, num_events):
        if num_events <= 0:
            return []

        if num_events < self.window_size:
            if self.drop_last:
                return []
            return [(0, num_events)]

        windows = []
        for start in range(0, num_events - self.window_size + 1, self.stride):
            windows.append((start, start + self.window_size))

        if not self.drop_last and windows and windows[-1][1] < num_events:
            next_start = windows[-1][0] + self.stride
            if next_start < num_events:
                windows.append((next_start, num_events))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample_idx, start, end = self.windows[index]
        events, target = self.dataset[sample_idx]

        if self.sort_by_time and getattr(events.dtype, "names", None) and "t" in events.dtype.names:
            events = events[np.argsort(events["t"])]

        events = events[start:end].copy()

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)

        return events, target
