"""Create the standalone conceptual latent-retrieval pipeline figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                           linewidth=1.5, edgecolor=color, facecolor="white")
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=11, color="#202124")


def arrow(axis, x1, y1, x2, y2):
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                  arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#4d5156"})


def main():
    output = Path("Img/latent_retrieval/latent_retrieval_pipeline.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 5.0))
    axis.set_xlim(0, 13); axis.set_ylim(0, 5); axis.axis("off")
    axis.text(0.35, 4.55, "Latent-space reconstruction fidelity and instance retrieval",
              fontsize=16, weight="bold", color="#202124")
    axis.text(0.45, 3.55, "Clean reference", fontsize=11, color="#5f6368")
    box(axis, 0.35, 2.55, 1.55, 0.65, r"$P_i$", "#2e7d32")
    box(axis, 2.35, 2.55, 1.55, 0.65, r"$E(P_i)=z_i^{clean}$", "#2e7d32")
    arrow(axis, 1.9, 2.875, 2.35, 2.875)

    axis.text(0.45, 1.95, "Corrupted query", fontsize=11, color="#5f6368")
    box(axis, 0.35, 0.95, 1.55, 0.65, r"$\widetilde P_i$", "#c62828")
    box(axis, 2.35, 0.95, 1.15, 0.65, r"$E$", "#1565c0")
    box(axis, 3.95, 0.95, 1.15, 0.65, r"$D$", "#1565c0")
    box(axis, 5.55, 0.95, 1.55, 0.65, r"$\widehat P_i$", "#c62828")
    box(axis, 7.55, 0.95, 1.15, 0.65, r"$E$", "#1565c0")
    box(axis, 9.15, 0.95, 1.75, 0.65, r"$z_i^{recon}$", "#1565c0")
    arrow(axis, 1.9, 1.275, 2.35, 1.275); arrow(axis, 3.5, 1.275, 3.95, 1.275)
    arrow(axis, 5.1, 1.275, 5.55, 1.275); arrow(axis, 7.1, 1.275, 7.55, 1.275)
    arrow(axis, 8.7, 1.275, 9.15, 1.275)

    box(axis, 11.25, 2.6, 1.25, 0.55, r"$d_{true}$", "#6a1b9a")
    box(axis, 11.25, 1.75, 1.25, 0.55, "Exact Top-1", "#6a1b9a")
    box(axis, 11.25, 0.9, 1.25, 0.55, "margin", "#6a1b9a")
    arrow(axis, 10.9, 1.275, 11.25, 1.0); arrow(axis, 10.9, 1.275, 11.25, 2.0)
    arrow(axis, 3.9, 2.875, 11.25, 2.875)
    axis.text(6.05, 3.05, r"compare $z_i^{clean}$ with $z_i^{recon}$ and the clean gallery",
              ha="center", fontsize=10, color="#5f6368")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
