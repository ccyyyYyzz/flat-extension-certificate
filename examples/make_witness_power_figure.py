"""Render the deterministic rank/signature power study from the JSON artifact."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    data = json.loads(Path("artifacts/prl_witness_power.json").read_text(encoding="utf-8"))
    rank = data["rank_power"]
    signature = data["signature_power"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    axes[0].semilogx(
        [item["shots_per_setting"] for item in rank],
        [item["certification_power"] for item in rank],
        marker="o",
    )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("shots per finite-difference setting")
    axes[0].set_ylabel("rank-3 certification power")
    axes[0].set_title("a  Response-rank power", loc="left", fontweight="bold")
    axes[1].plot(
        [item["noise_standard_deviation"] for item in signature],
        [item["median_metric_error"] for item in signature],
        marker="s",
        label="metric error",
    )
    axes[1].plot(
        [item["noise_standard_deviation"] for item in signature],
        [item["median_null_residual"] for item in signature],
        marker="^",
        label="null residual",
    )
    axes[1].set_xlabel("null-frequency noise standard deviation")
    axes[1].set_ylabel("median relative error")
    axes[1].set_title("b  Causal-form reconstruction", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    output = Path("paper/prl/figures/witness_power.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    print(output)


if __name__ == "__main__":
    main()
