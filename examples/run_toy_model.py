"""Run a compact CF-RSP finite-dimensional demonstration."""

from __future__ import annotations

import json

import numpy as np

from cyz_m import (
    basis_state,
    best_bipartition,
    consecutive_bell_pairs,
    evolve_state,
    measurement_dilation_z,
    normalised_relation_weights,
    pairwise_mutual_information,
    relation_hamiltonian,
    relation_lengths,
    shortest_path_metric,
    spectral_dimension,
    weighted_laplacian,
)


def rounded(matrix: np.ndarray, digits: int = 4) -> list[list[float]]:
    return np.round(matrix.astype(float), digits).tolist()


def bell_pair_partition_demo() -> dict[str, object]:
    state = consecutive_bell_pairs(2)
    dims = [2, 2, 2, 2]
    mutual_information = pairwise_mutual_information(state, dims)
    weights = normalised_relation_weights(mutual_information, dims)
    metric = shortest_path_metric(relation_lengths(weights, epsilon=1e-8))
    partition = best_bipartition(weights)

    diffusion_times = np.logspace(-2, 2, 31)
    heat, dimension = spectral_dimension(
        weighted_laplacian(weights), diffusion_times
    )
    midpoint = len(diffusion_times) // 2

    return {
        "scenario": "product of two Bell pairs",
        "mutual_information_bits": rounded(mutual_information),
        "normalised_relation_weights": rounded(weights),
        "relation_metric": rounded(metric),
        "best_balanced_partition": {
            "left": partition.left,
            "right": partition.right,
            "score": round(partition.score, 6),
            "within_mean": round(partition.within_mean, 6),
            "across_mean": round(partition.across_mean, 6),
        },
        "spectral_sample": {
            "diffusion_time": float(diffusion_times[midpoint]),
            "heat_trace": float(heat[midpoint]),
            "spectral_dimension": float(dimension[midpoint]),
            "warning": "A four-node graph has no continuum plateau; this is only an API check.",
        },
    }


def measurement_demo() -> dict[str, object]:
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    result = measurement_dilation_z(plus)
    return {
        "scenario": "Z measurement of |+> by CNOT dilation",
        "global_purity": result.global_purity,
        "reduced_system_purity": result.system_purity,
        "reduced_system_l1_coherence": result.system_coherence_l1,
        "system_apparatus_mutual_information_bits": result.mutual_information_bits,
        "interpretation": (
            "The global state stays pure while local coherence disappears and "
            "system-apparatus correlation grows."
        ),
    }


def relation_dynamics_demo() -> dict[str, object]:
    couplings = np.asarray(
        [
            [0.0, 1.0, 0.15, 0.0],
            [1.0, 0.0, 0.35, 0.0],
            [0.15, 0.35, 0.0, 0.8],
            [0.0, 0.0, 0.8, 0.0],
        ]
    )
    hamiltonian = relation_hamiltonian(couplings, local_fields=[0.2, -0.1, 0.05, 0.15])
    initial = basis_state("0000")
    snapshots: list[dict[str, object]] = []
    for time in (0.0, 0.2, 0.5, 0.9):
        state = evolve_state(initial, hamiltonian, time)
        mutual_information = pairwise_mutual_information(state, [2, 2, 2, 2])
        weights = normalised_relation_weights(mutual_information, [2, 2, 2, 2])
        snapshots.append(
            {
                "time_parameter": time,
                "mutual_information_bits": rounded(mutual_information),
                "best_partition": best_bipartition(weights).__dict__,
            }
        )
    return {
        "scenario": "abstract relation-coupling dynamics",
        "caveat": (
            "The time parameter and qubit tensor factors are gauge-fixed toy inputs, "
            "not fundamental CF-RSP structures."
        ),
        "snapshots": snapshots,
    }


def main() -> None:
    report = {
        "cf_rsp_toy_model": {
            "partition_emergence": bell_pair_partition_demo(),
            "measurement_reconfiguration": measurement_demo(),
            "relation_dynamics": relation_dynamics_demo(),
        }
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=list))


if __name__ == "__main__":
    main()
