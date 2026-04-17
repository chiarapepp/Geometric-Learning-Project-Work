import torch

from src.losses.chamfer_loss import chamfer_loss
from src.losses.density_aware_chamfer_loss import density_aware_chamfer_loss
from src.losses.emd_loss import emd_loss
from src.losses.sinkhorn_loss import sinkhorn_loss
from src.losses.temporal_weighted_chamfer_loss import temporal_weighted_chamfer_loss
from src.losses.hausdorff_loss import hausdorff_loss
from src.losses.projection_loss import projection_loss
from src.losses.voxel_loss import voxel_loss


def make_dummy_point_clouds(
    batch_size=2,
    num_points=128,
    point_dim=3,
    device="cpu",
):
    """
    Create dummy predicted and target point clouds.
    """
    target = torch.rand(batch_size, num_points, point_dim, device=device)
    pred = target + 0.05 * torch.randn(batch_size, num_points, point_dim, device=device)
    return pred, target


def run_single_test(name, fn, pred, target):
    try:
        loss = fn(pred, target)
        print(f"{name}: {loss.item():.6f}")
    except Exception as e:
        print(f"{name} ERROR: {e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n=== TEST SU POINT CLOUD 3D: (x, y, t) ===")
    pred3, target3 = make_dummy_point_clouds(
        batch_size=2,
        num_points=128,
        point_dim=3,
        device=device,
    )

    print("pred shape:", pred3.shape)
    print("target shape:", target3.shape)

    run_single_test("chamfer_loss", chamfer_loss, pred3, target3)
    run_single_test(
        "density_aware_chamfer_loss",
        lambda a, b: density_aware_chamfer_loss(a, b, alpha=1000, n_lambda=1),
        pred3,
        target3,
    )
    run_single_test("emd_loss", emd_loss, pred3, target3)
    run_single_test(
        "sinkhorn_loss",
        lambda a, b: sinkhorn_loss(a, b, p=2, blur=0.05, scaling=0.9),
        pred3,
        target3,
    )
    run_single_test(
        "temporal_weighted_chamfer_loss",
        lambda a, b: temporal_weighted_chamfer_loss(a, b, time_weight=2.0),
        pred3,
        target3,
    )
    run_single_test("hausdorff_loss", hausdorff_loss, pred3, target3)
    run_single_test(
        "projection_loss",
        lambda a, b: projection_loss(a, b, grid_size=32, sigma=0.04, loss_type="mse"),
        pred3,
        target3,
    )
    run_single_test(
        "voxel_loss",
        lambda a, b: voxel_loss(a, b, grid_size=16, sigma=0.05, loss_type="mse"),
        pred3,
        target3,
    )

    print("\n=== TEST SU POINT CLOUD 4D: (x, y, t, p) ===")
    pred4, target4 = make_dummy_point_clouds(
        batch_size=2,
        num_points=128,
        point_dim=4,
        device=device,
    )

    print("pred shape:", pred4.shape)
    print("target shape:", target4.shape)

    run_single_test("chamfer_loss_4d", chamfer_loss, pred4, target4)
    run_single_test(
        "density_aware_chamfer_loss_4d",
        lambda a, b: density_aware_chamfer_loss(a, b, alpha=1000, n_lambda=1),
        pred4,
        target4,
    )
    run_single_test("emd_loss_4d", emd_loss, pred4, target4)
    run_single_test(
        "sinkhorn_loss_4d",
        lambda a, b: sinkhorn_loss(a, b, p=2, blur=0.05, scaling=0.9),
        pred4,
        target4,
    )
    run_single_test(
        "temporal_weighted_chamfer_loss_4d",
        lambda a, b: temporal_weighted_chamfer_loss(a, b, time_weight=2.0),
        pred4,
        target4,
    )
    run_single_test("hausdorff_loss_4d", hausdorff_loss, pred4, target4)
    run_single_test(
        "projection_loss_4d",
        lambda a, b: projection_loss(
            a,
            b,
            grid_size=32,
            sigma=0.04,
            loss_type="mse",
            views=[(0, 1), (0, 2), (1, 2)],
        ),
        pred4,
        target4,
    )
    run_single_test(
        "voxel_loss_4d",
        lambda a, b: voxel_loss(
            a,
            b,
            grid_size=16,
            sigma=0.05,
            loss_type="mse",
            dims=(0, 1, 2),
        ),
        pred4,
        target4,
    )


if __name__ == "__main__":
    main()
    