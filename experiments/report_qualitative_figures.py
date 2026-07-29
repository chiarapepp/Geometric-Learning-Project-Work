"""Build the four compact qualitative figures used in the report.

The script composes the full visual-evaluation panels produced by
``experiments/visual_reconstruction_eval.py``. It does not rerun inference:
the target, corrupted input, and reconstruction crops all come from the fixed
sample-0 artifacts generated with seed 13.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COMPOSITES = ROOT / "outputs" / "final" / "qualitative" / "report_figures"
REPORT_FIGURES = ROOT / "report" / "figures"

DATASET_TITLES = {
    "dvsgesture": "DVSGesture",
    "nmnist": "N-MNIST",
    "ncaltech101": "N-Caltech101",
}

# The original 2x4 report composites contain clean target and reconstruction
# panels. These boxes retain the plot and axis labels while removing the old
# headings, which are replaced with a consistent layout below.
COMPOSITE_X = {
    "target": (285, 245, 1085, 815),
    "noise_reconstruction": (1225, 245, 2025, 815),
    "drop_reconstruction": (3065, 245, 3865, 815),
}
COMPOSITE_Y_OFFSET = 713

# Full 4x3 visual-evaluation panel crop for the corrupted-input xy projection.
XY_CORRUPTED_INPUT = (810, 875, 1620, 1425)

# Full 4x3 visual-evaluation panel crops for the xt row.
XT_CROPS = {
    "target": (20, 1475, 820, 2070),
    "corrupted": (810, 1475, 1620, 2070),
    "reconstruction": (1610, 1475, 2425, 2070),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual-eval-all",
        type=Path,
        default=None,
        help=(
            "Directory containing the six Chamfer visual-evaluation run "
            "folders. If omitted, it is discovered under ~/Downloads."
        ),
    )
    parser.add_argument(
        "--visual-eval-losses",
        type=Path,
        default=None,
        help=(
            "Directory containing DVSGesture PointNet Chamfer and Hausdorff "
            "visual-evaluation folders. If omitted, it is discovered under "
            "~/Downloads."
        ),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def find_visual_root(
    explicit: Path | None,
    directory_name: str,
    required_run: str,
) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / required_run).is_dir():
            raise FileNotFoundError(f"Missing {required_run} under {root}")
        return root

    downloads = Path.home() / "Downloads"
    candidates = sorted(downloads.rglob(directory_name))
    for candidate in candidates:
        if (candidate / required_run).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {directory_name}/{required_run} under {downloads}. "
        "Pass the corresponding command-line option explicitly."
    )


def shifted_box(box: tuple[int, int, int, int], row: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    offset = row * COMPOSITE_Y_OFFSET
    return left, top + offset, right, bottom + offset


def load_crop(
    path: Path,
    box: tuple[int, int, int, int],
    top_mask: int = 0,
) -> Image.Image:
    with Image.open(path) as image:
        crop = image.convert("RGB").crop(box)
    if top_mask:
        ImageDraw.Draw(crop).rectangle((110, 0, crop.width, top_mask), fill="white")
    return crop


def full_panel_path(
    visual_root: Path,
    dataset: str,
    model: str,
    loss: str,
    corruption: str,
) -> Path:
    run = f"{dataset}_{model}_{loss}"
    suffix = {
        "gaussian_noise": "sample_0000_gaussian_noise_0p1.png",
        "temporal_shuffle": "sample_0000_temporal_shuffle_1p0.png",
        "random_drop": "sample_0000_random_drop_0p5.png",
    }[corruption]
    path = visual_root / run / suffix
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_time_colorbar(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis"),
        ax=axes,
        orientation="horizontal",
        fraction=0.028,
        pad=0.055,
        aspect=45,
    )
    colorbar.set_label("Normalized timestamp $t$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def build_xy_composite(
    dataset: str,
    row_specs: list[tuple[str, str]],
    visual_root: Path,
    dpi: int,
) -> None:
    original_name = {
        "dvsgesture": "dvsgesture_chamfer_composite_xy.png",
        "nmnist": "nmnist_chamfer_vs_dcd_xy.png",
        "ncaltech101": "ncaltech101_chamfer_composite_xy.png",
    }[dataset]
    original_path = ORIGINAL_COMPOSITES / original_name

    fig, grid = plt.subplots(
        len(row_specs),
        5,
        figsize=(13.2, 5.35),
        constrained_layout=True,
    )
    if len(row_specs) == 1:
        grid = grid[None, :]

    column_titles = [
        "Clean target",
        "Noisy input ($\\sigma=0.10$)",
        "Noise reconstruction",
        "Dropped input ($\\delta=0.50$)",
        "Drop reconstruction",
    ]
    all_axes: list[plt.Axes] = []

    for row, (row_label, model) in enumerate(row_specs):
        noise_panel = full_panel_path(
            visual_root, dataset, model, "chamfer", "gaussian_noise"
        )
        drop_panel = full_panel_path(
            visual_root, dataset, model, "chamfer", "random_drop"
        )

        images = [
            load_crop(original_path, shifted_box(COMPOSITE_X["target"], row), 40),
            load_crop(noise_panel, XY_CORRUPTED_INPUT, 42),
            load_crop(
                original_path,
                shifted_box(COMPOSITE_X["noise_reconstruction"], row),
                40,
            ),
            load_crop(drop_panel, XY_CORRUPTED_INPUT, 42),
            load_crop(
                original_path,
                shifted_box(COMPOSITE_X["drop_reconstruction"], row),
                40,
            ),
        ]

        for column, (axis, image) in enumerate(zip(grid[row], images)):
            axis.imshow(image)
            axis.set_axis_off()
            if row == 0:
                axis.set_title(column_titles[column], fontsize=9, pad=4)
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    row_label,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axis.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )
            all_axes.append(axis)

    comparison = (
        "Chamfer versus DCD"
        if dataset == "nmnist"
        else "Chamfer-trained PointNet versus PointNet++"
    )
    fig.suptitle(
        f"{DATASET_TITLES[dataset]}: {comparison} ($xy$ projection)",
        fontsize=12,
    )
    add_time_colorbar(fig, all_axes)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURES / original_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_dvs_loss_comparison(visual_root: Path, dpi: int) -> None:
    chamfer_panel = full_panel_path(
        visual_root,
        "dvsgesture",
        "pointnet_ae",
        "chamfer",
        "temporal_shuffle",
    )
    hausdorff_panel = full_panel_path(
        visual_root,
        "dvsgesture",
        "pointnet_ae",
        "hausdorff",
        "temporal_shuffle",
    )
    images = [
        load_crop(chamfer_panel, XT_CROPS["target"], 56),
        load_crop(chamfer_panel, XT_CROPS["corrupted"], 56),
        load_crop(chamfer_panel, XT_CROPS["reconstruction"], 56),
        load_crop(hausdorff_panel, XT_CROPS["reconstruction"], 56),
    ]
    titles = [
        "Clean target",
        "Shuffled input ($\\rho=1.00$)",
        "Chamfer reconstruction",
        "Hausdorff reconstruction",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.15), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(title, fontsize=10, pad=4)

    fig.suptitle(
        "DVSGesture PointNet: temporal-shuffle response ($xt$ projection)",
        fontsize=12,
    )
    add_time_colorbar(fig, list(axes))
    fig.savefig(
        REPORT_FIGURES / "dvsgesture_chamfer_vs_hausdorff_xt.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    visual_all = find_visual_root(
        args.visual_eval_all,
        "visual_eval_all",
        "ncaltech101_pointnetpp_ae_chamfer",
    )
    visual_losses = find_visual_root(
        args.visual_eval_losses,
        "visual_eval",
        "dvsgesture_pointnet_ae_hausdorff",
    )

    build_xy_composite(
        "dvsgesture",
        [("PointNet", "pointnet_ae"), ("PointNet++", "pointnetpp_ae")],
        visual_all,
        args.dpi,
    )
    build_xy_composite(
        "nmnist",
        [("Chamfer", "pointnet_ae"), ("DCD", "pointnet_ae")],
        visual_all,
        args.dpi,
    )
    build_xy_composite(
        "ncaltech101",
        [("PointNet", "pointnet_ae"), ("PointNet++", "pointnetpp_ae")],
        visual_all,
        args.dpi,
    )
    build_dvs_loss_comparison(visual_losses, args.dpi)

    print(f"Wrote four qualitative figures to {REPORT_FIGURES}")


if __name__ == "__main__":
    main()
