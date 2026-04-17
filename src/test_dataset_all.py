import torch

from src.datasets.dataset_factory import get_dataset
from src.datasets.transforms import (
    Compose,
    EventsToXYTP,
    NormalizeXYT,
    DropPolarity,
    SamplePoints,
    ToTensor,
)


def build_transform(sensor_size, num_points=1024, temporal_weight=1.0):
    return Compose([
        EventsToXYTP(),
        NormalizeXYT(sensor_size=sensor_size, temporal_weight=temporal_weight),
        DropPolarity(),
        SamplePoints(num_points=num_points, mode="random", pad_mode="repeat"),
        ToTensor(),
    ])


def test_one_dataset(dataset_name, save_to="./data", train=True, num_points=1024, index=0):
    if dataset_name == "DVSGesture":
        sensor_size = (128, 128, 2)
    elif dataset_name == "NMNIST":
        sensor_size = (34, 34, 2)
    elif dataset_name == "NCaltech101":
        sensor_size = (240, 180, 2)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    transform = build_transform(sensor_size=sensor_size, num_points=num_points)

    dataset = get_dataset(
        dataset_name=dataset_name,
        save_to=save_to,
        train=train,
        transform=transform,
    )

    print(f"\n===== TESTING {dataset_name} =====")
    print(f"Length: {len(dataset)}")

    points, label = dataset[index]

    print(f"Label: {label}")
    print(f"Shape: {tuple(points.shape)}")
    print(f"Dtype: {points.dtype}")
    print(f"x range: {points[:, 0].min().item():.4f} / {points[:, 0].max().item():.4f}")
    print(f"y range: {points[:, 1].min().item():.4f} / {points[:, 1].max().item():.4f}")
    print(f"t range: {points[:, 2].min().item():.4f} / {points[:, 2].max().item():.4f}")
    print(f"First 3 points:\n{points[:3]}")

    assert isinstance(points, torch.Tensor)
    assert points.shape == (num_points, 3)
    print("OK")


def main():
    datasets = ["DVSGesture", "NMNIST", "NCaltech101"]

    for dataset_name in datasets:
        try:
            test_one_dataset(dataset_name)
        except Exception as e:
            print(f"ERROR while testing {dataset_name}: {e}")


if __name__ == "__main__":
    main()