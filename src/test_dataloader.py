import argparse
from torch.utils.data import DataLoader

from datasets.dataset_factory import get_dataset
from datasets.transforms import (
    Compose,
    EventsToXYTP,
    NormalizeXYT,
    DropPolarity,
    SamplePoints,
    ToTensor,
)


def infer_sensor_size(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "dvsgesture":
        return (128, 128, 2)
    if dataset_name == "nmnist":
        return (34, 34, 2)
    if dataset_name == "ncaltech101":
        return (240, 180, 2)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_transform(sensor_size, num_points=1024):
    return Compose([
        EventsToXYTP(),
        NormalizeXYT(sensor_size=sensor_size, temporal_weight=1.0),
        DropPolarity(),
        SamplePoints(num_points=num_points, mode="random", pad_mode="repeat"),
        ToTensor(),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["DVSGesture", "NMNIST", "NCaltech101"])
    parser.add_argument("--save_to", type=str, default="./data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    sensor_size = infer_sensor_size(args.dataset)
    transform = build_transform(sensor_size=sensor_size, num_points=args.num_points)

    dataset = get_dataset(
        dataset_name=args.dataset,
        save_to=args.save_to,
        train=args.train,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    batch_points, batch_labels = next(iter(loader))

    print("=== BATCH INFO ===")
    print(f"batch_points shape: {tuple(batch_points.shape)}")
    print(f"batch_points dtype: {batch_points.dtype}")
    print(f"batch_labels shape: {tuple(batch_labels.shape)}")
    print(f"batch_labels dtype: {batch_labels.dtype}")

    print("\n=== VALUE RANGES ===")
    print(f"x min/max: {batch_points[:, :, 0].min().item():.4f} / {batch_points[:, :, 0].max().item():.4f}")
    print(f"y min/max: {batch_points[:, :, 1].min().item():.4f} / {batch_points[:, :, 1].max().item():.4f}")
    print(f"t min/max: {batch_points[:, :, 2].min().item():.4f} / {batch_points[:, :, 2].max().item():.4f}")

    assert batch_points.ndim == 3
    assert batch_points.shape[1] == args.num_points
    assert batch_points.shape[2] == 3

    print("\nOK: dataloader batch looks valid.")


if __name__ == "__main__":
    main()