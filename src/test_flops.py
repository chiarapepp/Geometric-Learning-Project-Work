import torch

from src.evaluate import ALL_LOSSES
from src.flops import estimate_loss_flops


def main():
    pred = torch.rand(2, 64, 4)
    target = torch.rand(2, 64, 4)
    loss_kwargs = {
        "projection": {"grid_size": 16},
        "voxel": {"grid_size": 8},
        "sinkhorn": {"sinkhorn_iterations_estimate": 50},
    }

    print("Estimated FLOPs for one loss forward pass")
    for loss_name in ALL_LOSSES:
        estimate = estimate_loss_flops(
            loss_name,
            pred,
            target,
            loss_kwargs=loss_kwargs.get(loss_name, {}),
        )
        print(
            f"{loss_name:28s} "
            f"flops={estimate.flops:12d} "
            f"per_sample={estimate.flops_per_sample:12.1f} "
            f"method={estimate.method}"
        )


if __name__ == "__main__":
    main()
