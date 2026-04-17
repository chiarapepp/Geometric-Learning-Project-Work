from .dvs_gesture_dataset import DVSGestureDataset
from .nmnist_dataset import NMNISTDataset
from .ncaltech101_dataset import NCaltech101Dataset
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


def build_pointcloud_transform(
    dataset_name: str,
    num_points: int = 1024,
    input_dim: int = 4,
    temporal_weight: float = 1.0,
    sample_mode: str = "random",
    pad_mode: str = "repeat",
    shuffle: bool = True,
):
    if input_dim not in (3, 4):
        raise ValueError("input_dim must be 3 or 4")

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
    **kwargs,
):
    name = dataset_name.lower()

    if name == "dvsgesture":
        return DVSGestureDataset(
            save_to=save_to,
            train=train,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    if name == "nmnist":
        return NMNISTDataset(
            save_to=save_to,
            train=train,
            first_saccade_only=kwargs.get("first_saccade_only", False),
            stabilize=kwargs.get("stabilize", False),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    if name == "ncaltech101":
        return NCaltech101Dataset(
            save_to=save_to,
            train=train,
            split_ratio=kwargs.get("split_ratio", 0.8),
            split_seed=kwargs.get("split_seed", 13),
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")
