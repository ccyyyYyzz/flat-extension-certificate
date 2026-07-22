"""Run the CF-RSP v1 finite core without a supplied subsystem decomposition."""
from __future__ import annotations

import numpy as np

from cyz_m.factorization import (
    hidden_tensor_events,
    process_algebra_no_go,
    search_intrinsic_factorizations,
)
from cyz_m.physics import verify_gibbs_modular_alignment
from cyz_m.spectral_audit import PlateauRule, best_spectral_plateau


def main() -> None:
    events, _, _ = hidden_tensor_events(2, 2, seed=7, include_bridge=True)
    recovered = search_intrinsic_factorizations(events, max_bridge_count=1)
    assert recovered.best is not None
    print("Intrinsic factor search")
    print("  left events:", recovered.best.left_events)
    print("  right events:", recovered.best.right_events)
    print("  bridge events:", recovered.best.bridge_events)
    print("  factor sizes:", recovered.best.left_factor_size, recovered.best.right_factor_size)
    print("  exact:", recovered.best.exact)
    print("  score gap:", f"{recovered.score_gap:.6g}")

    rng = np.random.default_rng(11)
    generic_events = []
    for _ in range(3):
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        generic_events.append((matrix + matrix.conj().T) / 2)
    no_go = process_algebra_no_go(generic_events)
    print("\nNo-go diagnostic")
    print("  event algebra dimension:", no_go.event_algebra_dimension)
    print("  commutant dimension:", no_go.commutant_dimension)
    print("  noiseless factor forbidden:", no_go.nontrivial_exact_noiseless_factor_forbidden)

    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    alignment = verify_gibbs_modular_alignment(x, z, beta=1.5, modular_parameter=0.3)
    print("\nModular/Hamiltonian compatibility")
    print("  calibrated physical time:", alignment.physical_time)
    print("  relative error:", f"{alignment.error:.3e}")

    times = np.logspace(-3, 3, 121)
    log_times = np.log10(times)
    synthetic_dimension = 3.0 + 0.02 * np.sin(5 * log_times)
    synthetic_dimension[np.abs(log_times) > 1.2] += (
        np.abs(log_times[np.abs(log_times) > 1.2]) - 1.2
    ) ** 2
    rule = PlateauRule(
        min_log_width_decades=1.0,
        min_points=15,
        max_standard_deviation=0.08,
        max_absolute_slope=0.08,
        target_dimension=3.0,
        target_tolerance=0.1,
    )
    plateau = best_spectral_plateau(times, synthetic_dimension, rule=rule)
    print("\nTarget-blind plateau selection (synthetic protocol check)")
    print("  interval:", None if plateau is None else (plateau.start_time, plateau.end_time))
    print("  selected mean:", None if plateau is None else plateau.mean_dimension)
    print("  target was not used to choose the interval")


if __name__ == "__main__":
    main()
