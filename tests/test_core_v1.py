from __future__ import annotations

import unittest

import numpy as np

from cyz_m.factorization import (
    generated_star_algebra,
    hidden_tensor_events,
    perturb_events,
    process_algebra_no_go,
    search_intrinsic_factorizations,
)
from cyz_m.physics import (
    born_probabilities,
    gibbs_state,
    modular_invariance_defect,
    spohn_entropy_production,
    verify_gibbs_modular_alignment,
)
from cyz_m.spectral_audit import (
    PlateauRule,
    best_spectral_plateau,
    finite_graph_endpoint_data,
    spectral_dimension_curve,
)


class IntrinsicFactorizationTests(unittest.TestCase):
    def test_recovers_hidden_factors_without_tensor_input(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=17)
        result = search_intrinsic_factorizations(events)

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertTrue(result.best.exact)
        self.assertEqual(result.best.left_events, (0, 1))
        self.assertEqual(result.best.right_events, (2, 3))
        self.assertEqual(result.best.left_factor_size, 2)
        self.assertEqual(result.best.right_factor_size, 2)
        self.assertGreater(result.score_gap, 1.0)
        self.assertEqual(result.exact_candidate_count, 1)

    def test_factorization_survives_predeclared_small_perturbation(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=18)
        noisy = perturb_events(events, 1e-6, seed=5)
        result = search_intrinsic_factorizations(
            noisy, tolerance=1e-3, exact_tolerance=1e-2
        )

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertEqual(result.best.left_events, (0, 1))
        self.assertEqual(result.best.right_events, (2, 3))
        self.assertLess(result.best.commutator_defect, 1e-5)

    def test_identifies_an_interaction_event_as_a_bridge(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=19, include_bridge=True)
        result = search_intrinsic_factorizations(events, max_bridge_count=1)

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertTrue(result.best.exact)
        self.assertEqual(result.best.bridge_events, (4,))
        self.assertEqual(result.best.left_events, (0, 1))
        self.assertEqual(result.best.right_events, (2, 3))

    def test_full_random_event_algebra_gives_no_go_certificate(self) -> None:
        rng = np.random.default_rng(23)
        events = []
        for _ in range(3):
            matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            events.append((matrix + matrix.conj().T) / 2)
        certificate = process_algebra_no_go(events)

        self.assertEqual(certificate.event_algebra_dimension, 16)
        self.assertEqual(certificate.commutant_dimension, 1)
        self.assertTrue(certificate.nontrivial_exact_noiseless_factor_forbidden)

    def test_discovered_factor_is_modularly_invariant_for_product_gibbs_state(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=29)
        result = search_intrinsic_factorizations(events)
        assert result.best is not None
        left_basis = generated_star_algebra(
            [events[index] for index in result.best.left_events]
        )
        state = np.eye(4, dtype=np.complex128) / 4
        defect = modular_invariance_defect(left_basis, state, [-1.0, 0.0, 1.0])
        self.assertLess(defect, 1e-10)


class SpectralAuditTests(unittest.TestCase):
    def test_plateau_selection_is_target_blind(self) -> None:
        times = np.logspace(-3, 3, 121)
        x = np.log10(times)
        dimensions = 3.0 + 0.02 * np.sin(5 * x)
        dimensions[x < -1.2] = 3.0 + (x[x < -1.2] + 1.2) ** 2
        dimensions[x > 1.2] = 3.0 + (x[x > 1.2] - 1.2) ** 2

        common = dict(
            min_log_width_decades=1.0,
            min_points=15,
            max_standard_deviation=0.08,
            max_absolute_slope=0.08,
            target_tolerance=0.1,
        )
        target_three = PlateauRule(**common, target_dimension=3.0)
        target_seven = PlateauRule(**common, target_dimension=7.0)
        plateau_three = best_spectral_plateau(times, dimensions, rule=target_three)
        plateau_seven = best_spectral_plateau(times, dimensions, rule=target_seven)

        self.assertEqual(plateau_three, plateau_seven)
        self.assertIsNotNone(plateau_three)
        assert plateau_three is not None
        self.assertAlmostEqual(plateau_three.mean_dimension, 3.0, delta=0.08)

    def test_finite_graph_has_zero_endpoint_spectral_dimensions(self) -> None:
        weights = np.zeros((5, 5), dtype=float)
        for index in range(5):
            weights[index, (index + 1) % 5] = 1.0
            weights[(index + 1) % 5, index] = 1.0
        laplacian = np.diag(weights.sum(axis=1)) - weights
        data = finite_graph_endpoint_data(laplacian)
        self.assertEqual(data["zero_mode_count"], 1)
        self.assertEqual(data["small_time_spectral_dimension_limit"], 0.0)
        self.assertEqual(data["large_time_spectral_dimension_limit"], 0.0)

        _, dimensions = spectral_dimension_curve(laplacian, np.logspace(-6, 6, 101))
        self.assertLess(abs(dimensions[0]), 1e-4)
        self.assertLess(abs(dimensions[-1]), 1e-4)


class PhysicsCompatibilityTests(unittest.TestCase):
    def test_gibbs_modular_flow_equals_hamiltonian_flow_after_calibration(self) -> None:
        x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
        z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
        check = verify_gibbs_modular_alignment(x, z, beta=1.7, modular_parameter=0.4)
        self.assertAlmostEqual(check.physical_time, -0.68)
        self.assertLess(check.error, 1e-12)

    def test_born_rule_and_positive_spohn_production(self) -> None:
        state = np.diag([0.8, 0.2]).astype(np.complex128)
        effects = [
            np.diag([1.0, 0.0]).astype(np.complex128),
            np.diag([0.0, 1.0]).astype(np.complex128),
        ]
        np.testing.assert_allclose(born_probabilities(state, effects), [0.8, 0.2])

        stationary = np.eye(2, dtype=np.complex128) / 2
        generator_action = stationary - state
        production = spohn_entropy_production(state, stationary, generator_action)
        self.assertGreater(production, 0.0)

        gibbs = gibbs_state(np.diag([0.0, 1.0]), beta=2.0)
        self.assertAlmostEqual(float(np.trace(gibbs).real), 1.0, places=12)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(gibbs))), 0.0)


if __name__ == "__main__":
    unittest.main()
