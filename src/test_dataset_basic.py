import argparse
import torch

from src.datasets.dataset_factory import get_dataset
from src.datasets.transforms import (
    Compose,
    EventsToXYTP,
    NormalizeXYT,
    DropPolarity,
    SamplePoints,
    ShufflePoints,
    ToTensor,
)


def build_transform(
    sensor_size,
    num_points=1024,
    use_polarity=False,
    temporal_weight=1.0,
    sample_mode="random",
    pad_mode="repeat",
    shuffle_points=True,
):
    transforms = [
        EventsToXYTP(),
        NormalizeXYT(sensor_size=sensor_size, temporal_weight=temporal_weight),
    ]

    if not use_polarity:
        transforms.append(DropPolarity())

    transforms.append(SamplePoints(num_points=num_points, mode=sample_mode, pad_mode=pad_mode))
    if shuffle_points:
        transforms.append(ShufflePoints())
    transforms.append(ToTensor())
    return Compose(transforms)


def infer_sensor_size(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "dvsgesture":
        return (128, 128, 2)
    if dataset_name == "nmnist":
        return (34, 34, 2)
    if dataset_name == "ncaltech101":
        # N-Caltech101 has variable size, but after loading events are shifted to start from 0.
        # For normalization we can use a conservative upper bound.
        return (240, 180, 2)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["DVSGesture", "NMNIST", "NCaltech101"])
    parser.add_argument("--save_to", type=str, default="./data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--use_polarity", action="store_true")
    parser.add_argument("--temporal_weight", type=float, default=1.0)
    parser.add_argument("--sample_mode", default="random", choices=["random", "uniform", "first"])
    parser.add_argument("--pad_mode", default="repeat", choices=["repeat", "zeros"])
    parser.add_argument("--no_shuffle_points", action="store_true")
    parser.add_argument("--stream_mode", default="sample", choices=["sample", "windowed"])
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--keep_last_window", action="store_true")
    parser.add_argument("--max_windows_per_sample", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    sensor_size = infer_sensor_size(args.dataset)

    transform = build_transform(
        sensor_size=sensor_size,
        num_points=args.num_points,
        use_polarity=args.use_polarity,
        temporal_weight=args.temporal_weight,
        sample_mode=args.sample_mode,
        pad_mode=args.pad_mode,
        shuffle_points=not args.no_shuffle_points,
    )

    dataset = get_dataset(
        dataset_name=args.dataset,
        save_to=args.save_to,
        train=args.train,
        transform=transform,
        stream_mode=args.stream_mode,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_drop_last=not args.keep_last_window,
        max_windows_per_sample=args.max_windows_per_sample,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Train split: {args.train}")
    print(f"Dataset length: {len(dataset)}")

    points, label = dataset[args.index]

    print("\n=== SAMPLE INFO ===")
    print(f"Sample index: {args.index}")
    print(f"Label: {label}")
    print(f"Type(points): {type(points)}")
    print(f"Shape(points): {tuple(points.shape)}")
    print(f"Dtype(points): {points.dtype}")

    if not isinstance(points, torch.Tensor):
        raise TypeError("Expected torch.Tensor after ToTensor transform.")

    print("\n=== VALUE RANGES ===")
    print(f"x min/max: {points[:, 0].min().item():.4f} / {points[:, 0].max().item():.4f}")
    print(f"y min/max: {points[:, 1].min().item():.4f} / {points[:, 1].max().item():.4f}")
    print(f"t min/max: {points[:, 2].min().item():.4f} / {points[:, 2].max().item():.4f}")

    if points.shape[1] == 4:
        print(f"p min/max: {points[:, 3].min().item():.4f} / {points[:, 3].max().item():.4f}")

    print("\n=== FIRST 5 POINTS ===")
    print(points[:5])

    assert points.shape[0] == args.num_points, "Wrong number of sampled points."
    assert points.ndim == 2, "Points tensor must be 2D."
    assert points.shape[1] in (3, 4), "Points must have 3 or 4 channels."

    print("\nOK: dataset sample looks valid.")


if __name__ == "__main__":
    main()
