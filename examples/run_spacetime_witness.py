"""Generate deterministic PRL witness fixtures and a machine-readable report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cyz_m.collective_cad import collective_schur_effective, flat_collective_fixture, holonomy_frustrated_cycle
from cyz_m.spacetime_witness import (
    causal_signature_from_null_sheets,
    matrix_lie_closure,
    obstruction_audit,
    ramified_unitary_qfi,
    stable_cad_certificate,
)


def _as_json(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if hasattr(value, "__dict__"):
        return {key: _as_json(item) for key, item in value.__dict__.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def main() -> None:
    seed = 812731
    rng = np.random.default_rng(seed)
    base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cad = stable_cad_certificate(base, [1.3 * base, -0.7 * base], base)
    positive = obstruction_audit(np.asarray([[1.0, 0.0], [2.0, 0.0]]), base, [base, base], [base, base])
    multiplicity = obstruction_audit(np.eye(2), base, [base, base], [base, base])
    hidden = obstruction_audit(
        np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        base,
        [base, np.vstack([base, [0.0, 0.0, 1.0]])],
        [base[:1], np.vstack([base[:1], [0.0, 1.0, 0.0]])],
    )
    q = rng.normal(size=(256, 3))
    tilt = np.asarray([0.15, -0.05, 0.08])
    metric = np.asarray([[1.2, 0.1, 0.0], [0.1, 0.8, 0.04], [0.0, 0.04, 1.1]])
    radius = np.sqrt(np.einsum("ni,ij,nj->n", q, metric, q))
    signature = causal_signature_from_null_sheets(q, q @ tilt + radius, q @ tilt - radius)
    x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    lie = matrix_lie_closure([1j * x, 1j * y])
    flat = flat_collective_fixture(port_count=4, target_dimension=2)
    schur = collective_schur_effective(flat.audit.laplacian, [0.02 * x, 0.015 * y, 0.01 * (x + y), 0.005 * x], coupling_strength=4.0)
    frustrated = holonomy_frustrated_cycle(0.35)
    report = {
        "schema": "cyz_m.spacetime_witness.v1",
        "seed": seed,
        "positive": _as_json(positive),
        "multiplicity_negative_control": _as_json(multiplicity),
        "depth_negative_control": _as_json(hidden),
        "stable_cad": _as_json(cad),
        "causal_signature": _as_json(signature),
        "ramification": _as_json(ramified_unitary_qfi(0.0, 0.75)),
        "lie_closure": {"dimensions_by_depth": lie.dimensions_by_depth, "closure_depth": lie.closure_depth, "algebra_dimension": lie.algebra_dimension},
        "collective_flat": _as_json(flat.audit),
        "collective_frustrated": _as_json(frustrated.audit),
        "collective_schur": _as_json(schur),
    }
    output = Path("artifacts/prl_witness_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
