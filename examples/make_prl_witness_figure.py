"""Create the deterministic four-panel operational spacetime witness figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def _flow_box(ax, x: float, y: float, label: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        0.34,
        0.20,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="0.25",
        facecolor="0.97",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x + 0.17, y + 0.10, label, ha="center", va="center", fontsize=7.3, transform=ax.transAxes)


def main() -> None:
    output = Path("paper/prl/figures/operational_spacetime_witness.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    _flow_box(ax, 0.03, 0.62, "black-box\nprobabilities")
    _flow_box(ax, 0.62, 0.62, "response\nJacobians")
    _flow_box(ax, 0.03, 0.20, "three-obstruction\naudit")
    _flow_box(ax, 0.62, 0.20, "null-sheet causal\nspectroscopy")
    arrow = dict(arrowstyle="->", linewidth=1.0, color="0.25", shrinkA=3, shrinkB=3)
    ax.annotate("", xy=(0.62, 0.72), xytext=(0.37, 0.72), xycoords="axes fraction", arrowprops=arrow)
    ax.annotate("", xy=(0.20, 0.40), xytext=(0.20, 0.62), xycoords="axes fraction", arrowprops=arrow)
    ax.annotate("", xy=(0.62, 0.30), xytext=(0.37, 0.30), xycoords="axes fraction", arrowprops=arrow)
    ax.text(0.5, 0.04, "blind output: tangent rank + quadratic inertia", ha="center", fontsize=7.7, transform=ax.transAxes)
    ax.set_title("a  Beyond Hilbert-space witnesses", loc="left", fontsize=9.5, fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    fixtures = ["positive", "multiplicity", "kernel\nmismatch", "hidden\ndepth"]
    data = np.asarray([[0.0, 0.0, 0.0], [0.707, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    image = ax.imshow(data, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(3), [r"$\Delta_{\rm uni}$", r"$\Delta_{\rm comp}$", r"$\Delta_{\rm depth}$"], fontsize=7.2)
    ax.set_yticks(range(4), fixtures, fontsize=7.2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2g}", ha="center", va="center", fontsize=7.5)
    ax.set_title("b  One audit localizes failures", loc="left", fontsize=9.5, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)

    ax = fig.add_subplot(grid[1, 0])
    depth = np.arange(1, 5)
    ax.plot(depth, [3, 3, 3, 3], marker="o", label="closed positive")
    ax.plot(depth, [3, 4, 4, 4], marker="s", label="hidden commutator")
    ax.plot(depth, [3, 3, 4, 4], marker="^", label="ramified protocol")
    ax.set_xlabel("registered depth $k$", fontsize=8)
    ax.set_ylabel("certified response rank", fontsize=8)
    ax.set_xticks(depth)
    ax.set_ylim(2.85, 4.12)
    ax.legend(fontsize=6.8, frameon=False, loc="center right")
    ax.set_title("c  Finite-depth growth", loc="left", fontsize=9.5, fontweight="bold")

    ax = fig.add_subplot(grid[1, 1])
    scale = np.linspace(0, 8, 120)
    for gap, style in [(0.4, "-"), (0.8, "--"), (1.2, ":")]:
        ax.semilogy(scale, np.exp(-gap * scale), linestyle=style, label=fr"$\lambda_2={gap}$")
    ax.set_xlabel(r"conversion scale $\ell$", fontsize=8)
    ax.set_ylabel("relative-copy defect", fontsize=8)
    ax.legend(fontsize=6.8, frameon=False)
    ax.set_title("d  Gap-controlled collective descent", loc="left", fontsize=9.5, fontweight="bold")

    fig.savefig(output, format="svg")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, format="pdf")
    print(output)
    print(pdf_output)


if __name__ == "__main__":
    main()
