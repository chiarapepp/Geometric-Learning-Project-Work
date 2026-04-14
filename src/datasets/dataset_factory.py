from .dvs_gesture_dataset import DVSGestureDataset
from .nmnist_dataset import NMNISTDataset
from .ncaltech101_dataset import NCaltech101Dataset


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
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")