from __future__ import annotations

import unittest

import numpy as np

from cyz_m.event_relativity import (
    audit_event_basis_change,
    audit_factor_normalizer,
    mix_event_frame,
    search_resolved_factorizations,
    similarity_star_defect,
    simultaneous_unitary_presentation_change,
    state_weighted_event_gram,
    transpose_choi_min_eigenvalue,
)
from cyz_m.factorization import (
    generated_star_algebra,
    hidden_tensor_events,
    random_unitary,
)


class EventIndividuationRelativityTests(unittest.TestCase):
    def test_resolved_search_is_invariant_under_block_gl_event_frames(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=37)
        sectors = ((0, 1), (2, 3))
        base = search_resolved_factorizations(events, sectors)

        transform = np.zeros((4, 4), dtype=np.complex128)
        transform[:2, :2] = np.asarray([[1.0, 2.0j], [0.5, 1.0]])
        transform[2:, 2:] = np.asarray([[1.0, 0.3], [0.2j, 1.0]])
        audit = audit_event_basis_change(transform, sectors)
        mixed = mix_event_frame(events, transform)
        changed = search_resolved_factorizations(mixed, sectors)

        self.assertTrue(audit.allowed)
        self.assertIsNotNone(base.best)
        self.assertIsNotNone(changed.best)
        assert base.best is not None and changed.best is not None
        self.assertTrue(base.best.exact)
        self.assertTrue(changed.best.exact)
        self.assertEqual(base.best.left_events, changed.best.left_events)
        self.assertEqual(base.best.right_events, changed.best.right_events)
        self.assertAlmostEqual(base.best.score, changed.best.score, places=10)

    def test_cross_sector_mixing_is_not_a_passive_event_basis_change(self) -> None:
        transform = np.eye(4, dtype=np.complex128)
        transform[0, 2] = 0.2
        audit = audit_event_basis_change(transform, ((0, 1), (2, 3)))
        self.assertFalse(audit.allowed)
        self.assertGreater(audit.sector_leakage, 0.0)

    def test_calibrated_event_metric_reduces_block_gl_to_block_unitary(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=38)
        state = np.eye(4, dtype=np.complex128) / 4.0
        metric = state_weighted_event_gram(events, state)
        sectors = ((0, 1), (2, 3))

        unitary_frame = np.eye(4, dtype=np.complex128)
        unitary_frame[:2, :2] = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        allowed = audit_event_basis_change(unitary_frame, sectors, metric=metric)

        nonisometric = np.eye(4, dtype=np.complex128)
        nonisometric[0, 0] = 2.0
        rejected = audit_event_basis_change(nonisometric, sectors, metric=metric)

        self.assertTrue(allowed.allowed)
        self.assertFalse(rejected.allowed)
        assert rejected.metric_defect is not None
        self.assertGreater(rejected.metric_defect, 0.1)

    def test_simultaneous_ambient_unitary_preserves_event_two_point_data(self) -> None:
        events, _, _ = hidden_tensor_events(2, 2, seed=39)
        state = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
        unitary = random_unitary(4, seed=40)
        transformed_events, transformed_state = simultaneous_unitary_presentation_change(
            events, state, unitary
        )
        np.testing.assert_allclose(
            state_weighted_event_gram(events, state),
            state_weighted_event_gram(transformed_events, transformed_state),
            atol=1e-10,
        )

    def test_nonunitary_similarity_and_transpose_are_outside_core_gauge(self) -> None:
        unitary = random_unitary(3, seed=41)
        nonunitary = np.diag([2.0, 1.0, 0.5]).astype(np.complex128)
        self.assertLess(similarity_star_defect(unitary), 1e-10)
        self.assertGreater(similarity_star_defect(nonunitary), 0.1)
        self.assertLess(transpose_choi_min_eigenvalue(2), -0.9)

    def test_factor_normalizer_is_local_unitary_plus_optional_swap(self) -> None:
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        identity = np.eye(2, dtype=np.complex128)
        left = generated_star_algebra([np.kron(x, identity), np.kron(z, identity)])
        right = generated_star_algebra([np.kron(identity, x), np.kron(identity, z)])

        local = np.kron(random_unitary(2, seed=42), random_unitary(2, seed=43))
        self.assertTrue(audit_factor_normalizer(local, left, right).allowed)

        swap = np.zeros((4, 4), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                swap[2 * j + i, 2 * i + j] = 1.0
        self.assertFalse(audit_factor_normalizer(swap, left, right).allowed)
        self.assertTrue(
            audit_factor_normalizer(swap, left, right, allow_swap=True).allowed
        )

        generic = random_unitary(4, seed=44)
        self.assertFalse(
            audit_factor_normalizer(generic, left, right, allow_swap=True).allowed
        )


if __name__ == "__main__":
    unittest.main()
