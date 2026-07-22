"""Run the CF-RSP v3 bridge-response and common-cone diagnostics."""
from __future__ import annotations

import numpy as np

from cyz_m.bridge_response import (
    audit_principal_intertwiner,
    common_cone_soldering_audit,
    cone_consensus_flow,
    normalize_positive_frequency,
    operational_quotient_audit,
    response_pencil,
)
from cyz_m.dimension_lorentz import minkowski_metric


def main() -> None:
    identity = np.eye(2, dtype=np.complex128)
    x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    frequency = np.asarray([[2.0, 0.35], [0.35, 1.1]], dtype=np.complex128)
    primitive = [
        0.2 * identity + x,
        -0.1 * identity + 0.9 * y,
        0.3 * identity + 1.2 * z,
    ]
    raw = [
        primitive[0],
        primitive[1],
        primitive[2],
        primitive[0] + primitive[1],
        2.0 * primitive[0] - primitive[2],
        primitive[1] + 2.0 * primitive[2],
    ]
    normalized = normalize_positive_frequency(frequency, raw)
    first = response_pencil(normalized.normalized_spatial, minkowski_metric())

    theta = 0.61
    unitary = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    second_coefficients = [
        unitary @ coefficient @ unitary.conj().T
        for coefficient in normalized.normalized_spatial
    ]
    second = response_pencil(second_coefficients, minkowski_metric())

    intertwining = audit_principal_intertwiner(
        normalized.normalized_spatial, second_coefficients, unitary
    )
    common = common_cone_soldering_audit([first, second], [(0, 1)])

    rng = np.random.default_rng(7)
    protocol_jacobian = rng.normal(size=(7, 4)) @ first.solder_matrix
    quotient = operational_quotient_audit(first, protocol_jacobian)

    negative_primitive = [1.5 * x, 0.7 * y, 1.1 * z]
    negative = response_pencil(
        [
            negative_primitive[0],
            negative_primitive[1],
            negative_primitive[2],
            negative_primitive[0] + negative_primitive[1],
            2.0 * negative_primitive[0] - negative_primitive[2],
            negative_primitive[1] + 2.0 * negative_primitive[2],
        ],
        minkowski_metric(),
    )
    negative_common = common_cone_soldering_audit([first, negative], [(0, 1)])

    flow = cone_consensus_flow(
        [
            response_pencil([scale * x, y, z], minkowski_metric()).characteristic_form
            for scale in (0.6, 0.9, 1.2, 1.6)
        ],
        [(0, 1), (1, 2), (2, 3)],
        np.linspace(0.0, 8.0, 161),
    )

    print("frequency defect:", normalized.frequency_defect)
    print("principal intertwiner allowed:", intertwining.allowed)
    print("connected common cone:", common.universal)
    print("target-blind spatial quotient rank:", quotient.spatial_quotient_dimension)
    print("target-blind spacetime quotient rank:", quotient.spacetime_quotient_dimension)
    print("spectator response-kernel dimension:", quotient.response_kernel_dimension)
    print("independent multicone negative control passes:", negative_common.universal)
    print("graph algebraic connectivity:", flow.algebraic_connectivity)
    print("fitted consensus decay rate:", flow.fitted_decay_rate)


if __name__ == "__main__":
    main()
