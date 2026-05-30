"""
Dataset and transform factory utilities.

This module centralizes:
- Dataset instantiation for the supported event datasets (DVS Gesture, NMNIST, NCALTECH101).
- A standard preprocessing pipeline that converts raw events into fixed-size point clouds
    (XYT plus optional polarity) suitable for point-based models.
- Optional wrapping of a base dataset into a windowed stream via :class:`WindowedEventDataset`
    when stream_mode='windowed'.
"""
from .dvs_gesture_dataset import DVSGestureDataset
from .nmnist_dataset import NMNISTDataset
from .ncaltech101_dataset import NCaltech101Dataset
from .windowed_event_dataset import WindowedEventDataset
from .transforms import (
        Compose,
        EventsToXYTP,
        NormalizeXYT,
        PolarityToMinusOnePlusOne,
        DropPolarity,
        SamplePoints,
        ShufflePoints,
        ToTensor,
)

DATASET_SPECS = {
    "dvsgesture": {"sensor_size": DVSGestureDataset.sensor_size, "has_train_split": True},
    "nmnist": {"sensor_size": NMNISTDataset.sensor_size, "has_train_split": True},
    "ncaltech101": {"sensor_size": NCaltech101Dataset.sensor_size, "has_train_split": True},
}


def get_sensor_size(dataset_name: str):
    name = dataset_name.lower()
    return DATASET_SPECS[name]["sensor_size"]


# Define a pipeline of transformation to convert raw events into fixed-size 
# point clouds (XYT plus optional polarity)

def build_pointcloud_transform(
    dataset_name: str,
    num_points: int = 1024,   # Number of points to sample from the event cloud
    input_dim: int = 4, 
    temporal_weight: float = 1.0,
    sample_mode: str = "random",
    pad_mode: str = "repeat",  # How to pad if there are fewer than num_points 
    shuffle: bool = True,
):
    if input_dim not in (3, 4): # 3 for XYT, 4 for XYTP 
        raise ValueError("input_dim must be 3 or 4")

    # The transformation pipeline includes:
    # 1. Convert raw events to XYT(P) format.
    # 2. Normalize XYT coordinates to [0, 1], with an optional temporal weight to scale the time dimension.
    # 3. Convert polarity to -1/+1 if using polarity, or drop it if input_dim=3.
    # 4. Sample a fixed number of points from the event cloud, with options for sampling mode and padding.
    # 5. Optionally shuffle the points to remove any ordering bias.
    # 6. Convert the final point cloud to a PyTorch tensor.
    
    transforms = [
        EventsToXYTP(),
        NormalizeXYT(get_sensor_size(dataset_name), temporal_weight=temporal_weight),
        PolarityToMinusOnePlusOne(),
    ]
    if input_dim == 3:
        transforms.append(DropPolarity())
    transforms.append(SamplePoints(num_points=num_points, mode=sample_mode, pad_mode=pad_mode))
    if shuffle:
        transforms.append(ShufflePoints())
    transforms.append(ToTensor())
    return Compose(transforms)


def get_dataset(
    dataset_name: str,
    save_to: str,
    train: bool = True,
    transform=None,
    target_transform=None,
    transforms=None,
    stream_mode: str = "sample",
    window_size: int | None = None, # Size of the window in number of events when stream_mode='windowed'. If None, the entire sample is one window.
    window_stride: int | None = None, # Stride of the window in number of events when stream_mode='windowed'. If None, the window is contiguous.
    window_drop_last: bool = True, # if True, drop the last window if it has fewer than window_size events when stream_mode='windowed'.
    window_sort_by_time: bool = True,
    max_windows_per_sample: int | None = None,
    **kwargs,
):
    name = dataset_name.lower()

    # stream_mode can be either 'sample' where each sample is returned as a single point 
    # cloud, or 'windowed' where each sample is returned as a stream of windowed point clouds.
    stream_mode = stream_mode.lower()
    if stream_mode not in {"sample", "windowed"}:
        raise ValueError("stream_mode must be 'sample' or 'windowed'")

    if name == "dvsgesture":
        dataset = DVSGestureDataset(
            save_to=save_to,
            train=train,
        )

    elif name == "nmnist":
        dataset = NMNISTDataset(
            save_to=save_to,
            train=train,
            # Optionally only return the first saccade (first 300ms of events) to reduce the 
            # number of events and speed up training, since the later saccades often contain 
            # fewer events and may not add much information for classification.
            first_saccade_only=kwargs.get("first_saccade_only", False), 
            stabilize=kwargs.get("stabilize", False),
        )

    elif name == "ncaltech101":
        dataset = NCaltech101Dataset(
            save_to=save_to,
            train=train,
            #  the split wasn't provided by the original dataset
            split_ratio=kwargs.get("split_ratio", 0.8),
            split_seed=kwargs.get("split_seed", 13),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if stream_mode == "sample":
        dataset.transform = transform
        dataset.target_transform = target_transform
        dataset.transforms = transforms
        return dataset

    if window_size is None:
        raise ValueError("window_size is required when stream_mode='windowed'")

    return WindowedEventDataset(
        dataset=dataset,
        window_size=window_size,
        stride=window_stride,
        transform=transform,
        target_transform=target_transform,
        transforms=transforms,
        drop_last=window_drop_last,
        sort_by_time=window_sort_by_time,
        max_windows_per_sample=max_windows_per_sample,
    )
