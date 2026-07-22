"""Generate a deterministic power and experimental-design study for the PRL protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cyz_m.experimental_design import greedy_null_sheet_design, ramsey_frequency_budget
from cyz_m.witness_power import estimate_rank_power, estimate_signature_power


def _convert(value):
    if hasattr(value, "__dict__"):
        return {key: _convert(item) for key, item in value.__dict__.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_convert(item) for item in value]
    return value


def main() -> None:
    jacobian = np.asarray([
        [3.0, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, 0.0, 2.2],
        [1.2, -0.4, 0.6],
        [-0.8, 1.0, 0.4],
    ])
    rng = np.random.default_rng(812731)
    q = rng.normal(size=(120, 3))
    design = greedy_null_sheet_design(q, sample_count=10)
    rank_power = estimate_rank_power(jacobian, shots_per_setting=[500, 1000, 2500, 5000, 10000], trials=250)
    signature_power = estimate_signature_power(
        design.selected_tangents,
        [0.12, -0.04, 0.06],
        np.asarray([[1.2, 0.08, 0.0], [0.08, 0.9, 0.04], [0.0, 0.04, 1.1]]),
        noise_standard_deviations=[0.0, 0.01, 0.03, 0.06, 0.1],
        trials=250,
    )
    report = {
        "schema": "cyz_m.spacetime_witness.power.v2",
        "null_sheet_design": _convert(design),
        "ramsey_planning_model": _convert(
            ramsey_frequency_budget(0.02, dephasing_rate=0.5, contrast=0.9)
        ),
        "rank_power": _convert(rank_power),
        "signature_power": _convert(signature_power),
    }
    path = Path("artifacts/prl_witness_power.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
