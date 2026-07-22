from __future__ import annotations

import unittest

import numpy as np

from cyz_m.dimension_lorentz import (
    audit_cone_universality,
    audit_sl2c_lorentz,
    causal_atom_report,
    co_marginal_canonical_drift,
    co_marginal_sum_rule_residual,
    derive_dimension_selection,
    determinant_norm,
    determinant_quadratic_form,
    flat_spin_holonomy_defect,
    hermitian_from_coordinates,
    minkowski_metric,
    principal_symbol_determinant,
    quadratic_form_signature,
    select_isolated_point_node_dimension,
    select_minimal_quadratic_causal_atom,
    sl2c_boost,
    weyl_cone_metric,
)


class TargetBlindDimensionSelectionTests(unittest.TestCase):
    def test_minimal_quadratic_atom_infers_one_plus_three_signature(self) -> None:
        atom = select_minimal_quadratic_causal_atom(range(1, 8))
        self.assertEqual(atom.matrix_dimension, 2)
        self.assertEqual(atom.determinant_degree, 2)
        self.assertEqual(atom.self_adjoint_dimension, 4)
        self.assertEqual(
            (atom.signature_positive, atom.signature_negative, atom.signature_zero),
            (1, 3, 0),
        )
        self.assertEqual(atom.spatial_dimension, 3)
        self.assertEqual(atom.spacetime_dimension, 4)
        self.assertFalse(causal_atom_report(3).quadratic)

    def test_determinant_form_is_recovered_without_inserting_eta(self) -> None:
        inferred = determinant_quadratic_form()
        np.testing.assert_allclose(inferred, minkowski_metric(), atol=1e-12)
        self.assertEqual(quadratic_form_signature(inferred), (1, 3, 0))
        coordinates = np.asarray([2.0, 0.5, -0.25, 1.0])
        matrix = hermitian_from_coordinates(coordinates)
        self.assertAlmostEqual(float(np.linalg.det(matrix).real), determinant_norm(coordinates))

    def test_generic_isolated_two_level_node_selects_three_parameters(self) -> None:
        selected, reports = select_isolated_point_node_dimension(range(1, 9))
        self.assertEqual(selected, 3)
        report_by_dimension = {report.parameter_dimension: report for report in reports}
        self.assertFalse(report_by_dimension[2].generic_degeneracy_exists)
        self.assertTrue(report_by_dimension[3].isolated)
        self.assertEqual(report_by_dimension[4].generic_degeneracy_dimension, 1)
        self.assertTrue(report_by_dimension[4].extended)

    def test_two_independent_selectors_lock_three_plus_one(self) -> None:
        report = derive_dimension_selection(spectral_estimate=3.08, spectral_tolerance=0.1)
        self.assertTrue(report.unique)
        self.assertEqual(report.selected_spatial_dimension, 3)
        self.assertEqual(report.selected_spacetime_dimension, 4)
        self.assertTrue(report.spectral_consistent)

        mismatch = derive_dimension_selection(spectral_estimate=2.6, spectral_tolerance=0.1)
        self.assertFalse(mismatch.spectral_consistent)


class LorentzReconstructionTests(unittest.TestCase):
    def test_sl2c_action_preserves_minkowski_metric(self) -> None:
        rapidity = 0.7
        spin = sl2c_boost(rapidity, axis=2)
        audit = audit_sl2c_lorentz(spin)
        self.assertTrue(audit.allowed)
        self.assertAlmostEqual(audit.determinant, 1.0, places=10)
        self.assertAlmostEqual(audit.time_component, np.cosh(rapidity), places=10)
        np.testing.assert_allclose(
            audit.lorentz_matrix.T @ minkowski_metric() @ audit.lorentz_matrix,
            minkowski_metric(),
            atol=1e-10,
        )

    def test_transverse_weyl_node_has_lorentzian_principal_symbol(self) -> None:
        velocity = np.asarray(
            [[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.1]], dtype=float
        )
        cone = weyl_cone_metric(velocity, clock_scale=1.7)
        self.assertTrue(cone.lorentzian)
        self.assertEqual(
            (cone.signature_positive, cone.signature_negative, cone.signature_zero),
            (1, 3, 0),
        )
        frequency = 0.4
        momentum = np.asarray([0.2, -0.1, 0.3])
        direct = principal_symbol_determinant(
            frequency, momentum, velocity, clock_scale=1.7
        )
        four_vector = np.concatenate([[frequency], momentum])
        quadratic = float(four_vector @ cone.contravariant_metric @ four_vector)
        self.assertAlmostEqual(direct, quadratic, places=10)

    def test_shared_cone_is_conformal_not_coordinate_tuned(self) -> None:
        eta = minkowski_metric()
        shared = audit_cone_universality([eta, 3.5 * eta])
        self.assertTrue(shared.universal)
        split = eta.copy()
        split[3, 3] = -0.8
        rejected = audit_cone_universality([eta, split], tolerance=1e-3)
        self.assertFalse(rejected.universal)
        self.assertGreater(rejected.maximum_defect, 0.01)

    def test_flat_spin_gluing_has_central_holonomy(self) -> None:
        first = sl2c_boost(0.3, axis=0)
        second = sl2c_boost(-0.3, axis=0)
        self.assertLess(flat_spin_holonomy_defect([first, second]), 1e-10)
        curved = flat_spin_holonomy_defect(
            [sl2c_boost(0.3, axis=0), sl2c_boost(0.2, axis=1)]
        )
        self.assertGreater(curved, 1e-3)


class NewPredictionTests(unittest.TestCase):
    def test_dimension_drift_obeys_co_marginal_sum_rule(self) -> None:
        prediction = co_marginal_canonical_drift(
            3.6,
            gauge_coupling=0.7,
            yukawa_coupling=0.4,
            quartic_coupling=0.2,
        )
        self.assertAlmostEqual(prediction.common_dimension_drift, -0.4)
        self.assertAlmostEqual(prediction.normalized_gauge_drift, -0.4)
        self.assertAlmostEqual(prediction.normalized_yukawa_drift, -0.4)
        self.assertAlmostEqual(prediction.normalized_quartic_drift, -0.4)
        residual = co_marginal_sum_rule_residual(
            gauge_beta_residual=prediction.gauge_beta_canonical,
            yukawa_beta_residual=prediction.yukawa_beta_canonical,
            quartic_beta_residual=prediction.quartic_beta_canonical,
            gauge_coupling=0.7,
            yukawa_coupling=0.4,
            quartic_coupling=0.2,
        )
        self.assertLess(residual, 1e-12)


if __name__ == "__main__":
    unittest.main()
