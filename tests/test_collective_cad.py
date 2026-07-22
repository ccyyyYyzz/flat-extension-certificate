from __future__ import annotations

import unittest
import numpy as np

from cyz_m.collective_cad import (
    attempt_rate_certificate,
    audit_connection_laplacian,
    audit_cq_detailed_balance,
    collective_schur_effective,
    flat_collective_fixture,
    holonomy_frustrated_cycle,
    holonomy_partial_fixed_cycle,
    identity_transport_table,
)


class CollectiveCADTests(unittest.TestCase):
    def test_flat_connected_graph_activates_exactly_one_target_copy(self) -> None:
        fixture = flat_collective_fixture(port_count=5, target_dimension=3)
        self.assertTrue(fixture.audit.collective_activated)
        self.assertTrue(fixture.audit.one_copy_activated)
        self.assertEqual(fixture.audit.zero_mode_count, 3)
        self.assertGreater(fixture.audit.algebraic_gap, 0.0)

    def test_holonomy_frustration_removes_collective_zero_copy(self) -> None:
        fixture = holonomy_frustrated_cycle(0.4)
        self.assertFalse(fixture.audit.collective_activated)
        self.assertEqual(fixture.audit.zero_mode_count, 0)

    def test_so3_partial_fixed_cycle_reports_intermediate_fixed_subspace(self) -> None:
        fixture = holonomy_partial_fixed_cycle(0.4)
        self.assertEqual(fixture.audit.target_dimension, 3)
        self.assertEqual(fixture.audit.zero_mode_count, 1)
        self.assertTrue(fixture.audit.collective_activated)
        self.assertFalse(fixture.audit.one_copy_activated)
        self.assertTrue(fixture.audit.partial_fixed_subspace)

    def test_disconnected_graph_activates_one_copy_per_component(self) -> None:
        weights = np.zeros((4, 4))
        weights[0, 1] = weights[1, 0] = 1.0
        weights[2, 3] = weights[3, 2] = 1.0
        audit = audit_connection_laplacian(weights, identity_transport_table(4, 2))
        self.assertEqual(audit.zero_mode_count, 4)
        self.assertFalse(audit.one_copy_activated)

    def test_collective_schur_bound_is_respected(self) -> None:
        fixture = flat_collective_fixture(port_count=3, target_dimension=2)
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        report = collective_schur_effective(
            fixture.audit.laplacian,
            [0.03 * x, 0.02 * z, 0.01 * (x + z)],
            coupling_strength=3.0,
        )
        self.assertLessEqual(report.correction_norm, report.correction_bound + 1e-12)
        self.assertGreater(report.relative_gap, report.block_norm)
        self.assertEqual(report.collective_dimension, 2)

    def test_cq_markov_model_is_trace_preserving_stationary_and_detailed_balanced(self) -> None:
        rates = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        identity = np.eye(2, dtype=np.complex128)
        audit = audit_cq_detailed_balance(rates, [0.5, 0.5], ((identity, identity), (identity, identity)))
        self.assertLess(audit.trace_preservation_defect, 1e-10)
        self.assertLess(audit.stationary_defect, 1e-10)
        self.assertLess(audit.detailed_balance_defect, 1e-10)
        self.assertGreater(audit.spectral_gap, 0.0)

    def test_attempt_rate_operator_is_the_exact_uniform_hazard_certificate(self) -> None:
        first = np.diag([1.0, 0.0]).astype(np.complex128)
        second = np.diag([0.0, 1.0]).astype(np.complex128)
        certificate = attempt_rate_certificate([first, second], [0.3, 0.7])
        self.assertTrue(certificate.uniform_positive_hazard)
        self.assertAlmostEqual(certificate.minimum_rate, 0.3)
        dark = attempt_rate_certificate([first], [0.3])
        self.assertFalse(dark.uniform_positive_hazard)
        self.assertAlmostEqual(dark.minimum_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
