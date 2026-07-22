from __future__ import annotations

import unittest
import numpy as np

from cyz_m.spacetime_witness import (
    causal_signature_from_null_sheets,
    finite_lie_depth_bound,
    kernel_completeness_gap,
    matrix_lie_closure,
    obstruction_audit,
    ramified_unitary_qfi,
    response_rank_witness,
    stable_cad_certificate,
    universality_gap,
)


class OperationalWitnessTests(unittest.TestCase):
    def test_rank_witness_is_target_blind(self) -> None:
        response = np.diag([3.0, 1.0, 0.04, 0.001])
        witness = response_rank_witness(response, noise_bound=0.02)
        self.assertEqual(witness.certified_lower_bound, 3)
        self.assertEqual(witness.numerical_rank, 3)

    def test_obstruction_audit_separates_multiplicity_and_kernel_mismatch(self) -> None:
        coefficients = np.eye(2)
        atomic = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        full1 = atomic.copy()
        full2 = np.vstack([atomic, [0.0, 0.0, 1.0]])
        orbit1 = np.asarray([[1.0, 0.0, 0.0]])
        orbit2 = np.vstack([orbit1, [0.0, 1.0, 0.0]])
        audit = obstruction_audit(coefficients, atomic, [full1, full2], [orbit1, orbit2])
        self.assertIsNotNone(audit.universality_gap)
        assert audit.universality_gap is not None
        self.assertGreater(audit.universality_gap, 0.6)
        self.assertAlmostEqual(audit.completeness_gap, 1.0)
        self.assertEqual(audit.derivative_growth, (0, 1))
        self.assertEqual(audit.orbit_growth, (0, 1))

    def test_one_line_universality_is_zero(self) -> None:
        coefficients = np.asarray([[1.0, 0.0], [2.0, 0.0], [-0.5, 0.0]])
        gap, rank, _ = universality_gap(coefficients)
        self.assertAlmostEqual(gap or 0.0, 0.0, places=12)
        self.assertEqual(rank, 1)

    def test_null_sheet_spectroscopy_recovers_signature_without_target(self) -> None:
        rng = np.random.default_rng(5)
        q = rng.normal(size=(200, 3))
        tilt = np.asarray([0.2, -0.1, 0.05])
        metric = np.asarray([[1.4, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 1.2]])
        radius = np.sqrt(np.einsum("ni,ij,nj->n", q, metric, q))
        midpoint = q @ tilt
        witness = causal_signature_from_null_sheets(q, midpoint + radius, midpoint - radius)
        np.testing.assert_allclose(witness.tilt, tilt, atol=1e-10)
        np.testing.assert_allclose(witness.spatial_metric, metric, atol=1e-10)
        self.assertEqual((witness.positive, witness.negative, witness.zero), (1, 3, 0))
        self.assertEqual(witness.spatial_rank, 3)
        self.assertLess(witness.null_residual, 1e-10)

    def test_stable_cad_certificate_has_explicit_lower_bound(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(base, [2.0 * base, -0.5 * base], base.copy())
        self.assertTrue(certificate.certified_isomorphism)
        self.assertEqual(certificate.certificate_status, "certified")
        self.assertFalse(certificate.error_bound_vacuous)
        self.assertEqual(certificate.quotient_dimension, 2)
        self.assertGreater(certificate.descended_lower_singular_bound, 0.99)
        self.assertLess(certificate.universality_gap, 1e-12)

    def test_stable_cad_certificate_valid_finite_error_bound_still_certifies(self) -> None:
        # A finite-error bound with noise < gap is a genuine bounded-angle
        # near-isomorphism certificate: eta stays below 1 and every gate passes.
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            noise_norm=0.1, spectral_gap=1.0,
        )
        self.assertTrue(certificate.certified_isomorphism)
        self.assertEqual(certificate.certificate_status, "certified")
        self.assertFalse(certificate.error_bound_vacuous)
        self.assertLess(certificate.kernel_projector_gap, 1.0)
        self.assertGreater(certificate.descended_lower_singular_bound, 0.0)

    def test_stable_cad_certificate_eps_ge_gamma_is_inconclusive(self) -> None:
        # eps >= gamma is a vacuous Wedin bound; even on otherwise-green inputs it
        # must never be clipped into a certified branch (issue #3).
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            noise_norm=1.0, spectral_gap=0.5,
        )
        self.assertFalse(certificate.certified_isomorphism)
        self.assertTrue(certificate.error_bound_vacuous)
        self.assertEqual(certificate.certificate_status, "inconclusive")
        self.assertEqual(certificate.descended_lower_singular_bound, 0.0)

    def test_stable_cad_certificate_eps_eq_gamma_boundary_is_inconclusive(self) -> None:
        # Boundary noise == gap saturates the bound at 1 and is still vacuous.
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            noise_norm=0.5, spectral_gap=0.5,
        )
        self.assertFalse(certificate.certified_isomorphism)
        self.assertEqual(certificate.certificate_status, "inconclusive")

    def test_stable_cad_certificate_requires_both_finite_error_inputs(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            stable_cad_certificate(base, [2.0 * base], base.copy(), noise_norm=0.1)

    def test_two_matrix_sum_not_max_degrades_bound(self) -> None:
        # R3 Section E: the certified mismatch is the two-matrix triangle bound
        # eta_obs + b_A + b_F, not max(eta_obs, b_A, b_F). Here measured_eta = 0
        # and b_A = b_F = 0.6, each individually below 1 so a max rule would keep
        # eta = 0.6 < 1 and certify; the correct SUM is 1.2 >= 1 and must degrade
        # the verdict to inconclusive.
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            atomic_noise_norm=0.375, atomic_spectral_gap=1.0,
            full_noise_norm=0.375, full_spectral_gap=1.0,
        )
        # b_A = b_F = 0.375 / (1.0 - 0.375) = 0.6, each < 1 (max would pass).
        self.assertFalse(certificate.certified_isomorphism)
        self.assertTrue(certificate.error_bound_vacuous)
        self.assertEqual(certificate.certificate_status, "inconclusive")
        self.assertAlmostEqual(certificate.kernel_projector_gap, 1.2, places=9)
        self.assertEqual(certificate.descended_lower_singular_bound, 0.0)

    def test_two_matrix_valid_triangle_bound_certifies(self) -> None:
        # Small radii: b_A = b_F ~= 0.0526, so eta = 0 + b_A + b_F ~= 0.105 < 1
        # and every gate passes.
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            atomic_noise_norm=0.05, atomic_spectral_gap=1.0,
            full_noise_norm=0.05, full_spectral_gap=1.0,
        )
        self.assertTrue(certificate.certified_isomorphism)
        self.assertEqual(certificate.certificate_status, "certified")
        self.assertFalse(certificate.error_bound_vacuous)
        self.assertAlmostEqual(certificate.kernel_projector_gap, 2.0 * (0.05 / 0.95), places=9)
        self.assertGreater(certificate.descended_lower_singular_bound, 0.9)

    def test_two_matrix_nonpositive_gamma_safe_is_inconclusive(self) -> None:
        # gamma_safe = observed_gap - eps <= 0 for one matrix: no rank-stable
        # certified lower gap, so no triangle bound is formable.
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        certificate = stable_cad_certificate(
            base, [2.0 * base, -0.5 * base], base.copy(),
            atomic_noise_norm=1.0, atomic_spectral_gap=0.5,
            full_noise_norm=0.05, full_spectral_gap=1.0,
        )
        self.assertFalse(certificate.certified_isomorphism)
        self.assertTrue(certificate.error_bound_vacuous)
        self.assertEqual(certificate.certificate_status, "inconclusive")
        self.assertEqual(certificate.descended_lower_singular_bound, 0.0)

    def test_two_matrix_wiring_is_all_or_none(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            stable_cad_certificate(
                base, [2.0 * base], base.copy(),
                atomic_noise_norm=0.1, atomic_spectral_gap=1.0,
                full_noise_norm=0.1,  # full_spectral_gap missing
            )

    def test_two_matrix_and_single_pair_cannot_be_mixed(self) -> None:
        base = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            stable_cad_certificate(
                base, [2.0 * base], base.copy(),
                noise_norm=0.1, spectral_gap=1.0,
                atomic_noise_norm=0.1, atomic_spectral_gap=1.0,
                full_noise_norm=0.1, full_spectral_gap=1.0,
            )

    def test_ramified_parameter_has_zero_qfi_at_origin_but_regular_coordinate_does_not(self) -> None:
        witness = ramified_unitary_qfi(0.0, generator_variance=0.75)
        self.assertEqual(witness.qfi_raw, 0.0)
        self.assertAlmostEqual(witness.qfi_regular, 3.0)
        self.assertFalse(witness.first_derivative_visible)
        self.assertTrue(witness.second_order_visible)

    def test_matrix_lie_closure_detects_hidden_commutator_direction(self) -> None:
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        report = matrix_lie_closure([1j * x, 1j * y])
        self.assertEqual(report.dimensions_by_depth[0], 2)
        self.assertEqual(report.algebra_dimension, 3)
        self.assertEqual(report.closure_depth, 3)
        self.assertEqual(finite_lie_depth_bound(3, 2), 2)

    def test_kernel_gap_zero_for_equal_kernels(self) -> None:
        first = np.asarray([[1.0, 0.0, 0.0]])
        second = np.asarray([[2.0, 0.0, 0.0]])
        self.assertAlmostEqual(kernel_completeness_gap(first, second), 0.0)


if __name__ == "__main__":
    unittest.main()
