from __future__ import annotations

import unittest
import numpy as np

from cyz_m.bridge_response import (
    audit_principal_intertwiner,
    bridge_current_action,
    bridge_current_bilinear,
    common_cone_soldering_audit,
    cone_consensus_flow,
    haar_averaged_inner_product,
    kms_dirichlet_audit,
    modular_bridge_response_generator,
    normalize_positive_frequency,
    operational_quotient_audit,
    passive_response_resolvent,
    projective_null_transfer_defect,
    response_pencil,
)
from cyz_m.dimension_lorentz import minkowski_metric


def pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity = np.eye(2, dtype=np.complex128)
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    return identity, x, y, z


def su2_z(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]).astype(np.complex128)


class ResidueAndResponsePencilTests(unittest.TestCase):
    def test_positive_residue_normalization_is_canonical_and_preserves_zeros(self) -> None:
        identity, x, y, z = pauli()
        frequency = np.asarray([[2.0, 0.4], [0.4, 1.2]], dtype=np.complex128)
        coefficients = [
            0.3 * identity + x,
            -0.2 * identity + 0.7 * y,
            0.1 * identity + 1.2 * z,
        ]
        normalized = normalize_positive_frequency(frequency, coefficients)
        np.testing.assert_allclose(normalized.normalized_frequency, identity, atol=1e-11)
        self.assertLess(normalized.frequency_defect, 1e-11)

        omega = 0.7
        q = np.asarray([0.2, -0.3, 0.4])
        original = omega * frequency - sum(q[i] * coefficients[i] for i in range(3))
        transformed = omega * identity - sum(
            q[i] * normalized.normalized_spatial[i] for i in range(3)
        )
        expected = np.linalg.det(normalized.whitening) ** 2 * np.linalg.det(original)
        self.assertAlmostEqual(abs(np.linalg.det(transformed) - expected), 0.0, places=10)

    def test_response_quotient_discovers_visible_and_spectator_directions(self) -> None:
        identity, x, y, z = pauli()
        primitive = [
            0.15 * identity + x,
            -0.20 * identity + y,
            0.05 * identity + z,
        ]
        coefficients = [
            primitive[0],
            primitive[1],
            primitive[2],
            primitive[0] + primitive[1],
            2 * primitive[0] - primitive[2],
            primitive[1] + 2 * primitive[2],
        ]
        pencil = response_pencil(coefficients, minkowski_metric())
        self.assertEqual(pencil.spatial_rank, 3)
        self.assertEqual(pencil.extended_rank, 4)
        self.assertEqual(pencil.spatial_kernel.shape[1], 3)
        self.assertEqual(pencil.response_kernel.shape[1], 3)

        rng = np.random.default_rng(4)
        readout = rng.normal(size=(7, 4)) @ pencil.solder_matrix
        audit = operational_quotient_audit(pencil, readout)
        self.assertTrue(audit.minimal_faithful)
        self.assertEqual(audit.spatial_quotient_dimension, 3)
        self.assertEqual(audit.spacetime_quotient_dimension, 4)

        leaking = np.vstack([readout, pencil.response_kernel[:, 0]])
        rejected = operational_quotient_audit(pencil, leaking)
        self.assertFalse(rejected.minimal_faithful)
        assert rejected.protocol_through_solder_defect is not None
        self.assertGreater(rejected.protocol_through_solder_defect, 1e-3)


class CommonConeSolderingTests(unittest.TestCase):
    def test_connected_unitarily_related_species_have_one_cone(self) -> None:
        identity, x, y, z = pauli()
        base = [
            0.2 * identity + 1.1 * x,
            -0.1 * identity + 0.9 * y,
            0.3 * identity + 1.2 * z,
        ]
        pencils = [response_pencil(base, minkowski_metric())]
        for theta in (0.4, -0.7):
            u = su2_z(theta)
            transformed = [u @ coefficient @ u.conj().T for coefficient in base]
            pencils.append(response_pencil(transformed, minkowski_metric()))
        audit = common_cone_soldering_audit(pencils, [(0, 1), (1, 2)], tolerance=1e-9)
        self.assertTrue(audit.connected)
        self.assertTrue(audit.universal)
        self.assertLess(audit.maximum_edge_defect, 1e-10)

        samples = np.asarray(
            [[1.0, 0.2, -0.3], [-0.4, 0.7, 0.1], [0.3, -0.6, 0.9]]
        )
        self.assertLess(
            projective_null_transfer_defect(pencils[0], pencils[2], samples), 1e-10
        )

    def test_independent_species_are_a_required_negative_control(self) -> None:
        _, x, y, z = pauli()
        first = response_pencil([x, y, z], minkowski_metric())
        second = response_pencil([1.5 * x, 0.7 * y, 1.1 * z], minkowski_metric())
        audit = common_cone_soldering_audit([first, second], [(0, 1)], tolerance=1e-3)
        self.assertFalse(audit.universal)
        self.assertGreater(audit.maximum_edge_defect, 0.05)
        samples = np.asarray([[1.0, 0.2, -0.3], [-0.4, 0.7, 0.1]])
        self.assertGreater(projective_null_transfer_defect(first, second, samples), 1e-2)

    def test_disconnected_species_graph_does_not_claim_global_soldering(self) -> None:
        _, x, y, z = pauli()
        cone_a = response_pencil([x, y, z], minkowski_metric())
        cone_b = response_pencil([1.3 * x, 0.8 * y, z], minkowski_metric())
        audit = common_cone_soldering_audit(
            [cone_a, cone_a, cone_b, cone_b], [(0, 1), (2, 3)], tolerance=1e-10
        )
        self.assertFalse(audit.connected)
        self.assertFalse(audit.universal)
        self.assertEqual(audit.components, ((0, 1), (2, 3)))


class BridgeDynamicsAndOrbitTests(unittest.TestCase):
    def test_modular_bridge_generator_and_current_are_self_adjoint(self) -> None:
        _, x, _, z = pauli()
        state = np.diag([0.7, 0.3]).astype(np.complex128)
        generator = modular_bridge_response_generator(state, [x, z], [0.2, -0.1])
        np.testing.assert_allclose(generator, generator.conj().T, atol=1e-12)
        self.assertAlmostEqual(float(np.trace(generator).real), 0.0, places=12)
        current = bridge_current_action(x, z)
        np.testing.assert_allclose(current, current.conj().T, atol=1e-12)
        self.assertGreater(np.linalg.norm(current), 0.1)

    def test_response_resolvent_and_current_are_presentation_covariant(self) -> None:
        _, x, y, z = pauli()
        state = np.diag([0.65, 0.35]).astype(np.complex128)
        generator = 0.4 * x + 0.7 * z
        value = passive_response_resolvent(state, generator, x, y, 0.3 + 0.8j)
        current = bridge_current_bilinear(state, generator, x, y)
        u = su2_z(0.63)
        transformed = passive_response_resolvent(
            u @ state @ u.conj().T,
            u @ generator @ u.conj().T,
            u @ x @ u.conj().T,
            u @ y @ u.conj().T,
            0.3 + 0.8j,
        )
        transformed_current = bridge_current_bilinear(
            u @ state @ u.conj().T,
            u @ generator @ u.conj().T,
            u @ x @ u.conj().T,
            u @ y @ u.conj().T,
        )
        self.assertAlmostEqual(abs(value - transformed), 0.0, places=10)
        self.assertAlmostEqual(abs(current - transformed_current), 0.0, places=10)

    def test_reversible_mode_conversion_intertwines_principal_pencils(self) -> None:
        identity, x, y, z = pauli()
        source = [
            0.2 * identity + x,
            -0.1 * identity + y,
            0.3 * identity + z,
        ]
        u = su2_z(0.5)
        target = [u @ coefficient @ u.conj().T for coefficient in source]
        audit = audit_principal_intertwiner(source, target, u)
        self.assertTrue(audit.allowed)
        bad_target = [1.3 * target[0], target[1], target[2]]
        rejected = audit_principal_intertwiner(source, bad_target, u, tolerance=1e-5)
        self.assertFalse(rejected.allowed)
        self.assertGreater(rejected.intertwining_defect, 1e-2)

    def test_tracial_depolarizing_generator_has_positive_dirichlet_form(self) -> None:
        identity, x, y, z = pauli()
        state = identity / 2

        def generator(observable: np.ndarray) -> np.ndarray:
            return np.trace(observable) / 2 * identity - observable

        audit = kms_dirichlet_audit(state, [x, y, z], generator)
        self.assertTrue(audit.positive_semidefinite)
        self.assertLess(audit.symmetry_defect, 1e-12)
        self.assertGreater(audit.minimum_eigenvalue, 0.49)

    def test_haar_quadrature_constructs_invariant_response_metric(self) -> None:
        rotations = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            rotations.append(
                np.asarray(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                    dtype=float,
                )
            )
        base = np.asarray([[3.0, 0.4], [0.4, 1.0]])
        metric = haar_averaged_inner_product(rotations, base_metric=base)
        self.assertAlmostEqual(metric[0, 0], metric[1, 1], places=12)
        self.assertAlmostEqual(metric[0, 1], 0.0, places=12)
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ metric @ rotation, metric, atol=1e-11)

    def test_consensus_flow_rate_is_controlled_by_species_graph_gap(self) -> None:
        _, x, y, z = pauli()
        forms = [
            response_pencil([scale * x, y, z], minkowski_metric()).characteristic_form
            for scale in (0.6, 0.9, 1.2, 1.6)
        ]
        flow = cone_consensus_flow(
            forms, [(0, 1), (1, 2), (2, 3)], np.linspace(0.0, 8.0, 161)
        )
        self.assertGreater(flow.algebraic_connectivity, 0.0)
        self.assertLess(flow.defects[-1], 0.02 * flow.defects[0])
        self.assertGreater(flow.fitted_decay_rate, 0.5 * flow.algebraic_connectivity)
        self.assertLess(flow.fitted_decay_rate, 2.5 * flow.algebraic_connectivity)


if __name__ == "__main__":
    unittest.main()
