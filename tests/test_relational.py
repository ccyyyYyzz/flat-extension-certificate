from __future__ import annotations

import unittest

import numpy as np

from cyz_m import (
    basis_state,
    bell_pair_state,
    best_bipartition,
    consecutive_bell_pairs,
    evolve_state,
    measurement_dilation_z,
    normalised_relation_weights,
    pairwise_mutual_information,
    partial_trace,
    relation_hamiltonian,
    relation_lengths,
    shortest_path_metric,
    spectral_dimension,
    von_neumann_entropy,
    weighted_laplacian,
)


class RelationalGeometryTests(unittest.TestCase):
    def test_bell_pair_has_maximal_pairwise_mutual_information(self) -> None:
        state = bell_pair_state()
        reduced = partial_trace(state, [0], [2, 2])
        np.testing.assert_allclose(reduced, np.eye(2) / 2.0, atol=1e-10)
        self.assertAlmostEqual(von_neumann_entropy(reduced), 1.0, places=10)

        matrix = pairwise_mutual_information(state, [2, 2])
        self.assertAlmostEqual(matrix[0, 1], 2.0, places=10)

        weights = normalised_relation_weights(matrix, [2, 2])
        self.assertAlmostEqual(weights[0, 1], 1.0, places=10)

    def test_product_state_has_zero_pairwise_mutual_information(self) -> None:
        state = basis_state("000")
        matrix = pairwise_mutual_information(state, [2, 2, 2])
        np.testing.assert_allclose(matrix, np.zeros((3, 3)), atol=1e-10)

    def test_shortest_path_closure_satisfies_triangle_inequality(self) -> None:
        weights = np.asarray(
            [
                [0.0, 0.8, 0.1],
                [0.8, 0.0, 0.8],
                [0.1, 0.8, 0.0],
            ],
            dtype=float,
        )
        edge_lengths = relation_lengths(weights, epsilon=1e-12)
        metric = shortest_path_metric(edge_lengths)

        self.assertLessEqual(metric[0, 2], edge_lengths[0, 2] + 1e-12)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.assertLessEqual(
                        metric[i, k], metric[i, j] + metric[j, k] + 1e-10
                    )

    def test_bell_pair_product_selects_the_pair_partition(self) -> None:
        state = consecutive_bell_pairs(2)
        matrix = pairwise_mutual_information(state, [2, 2, 2, 2])
        weights = normalised_relation_weights(matrix, [2, 2, 2, 2])
        partition = best_bipartition(weights, balanced=True)

        self.assertEqual(set(partition.left), {0, 1})
        self.assertEqual(set(partition.right), {2, 3})
        self.assertGreater(partition.within_mean, 0.9)
        self.assertLess(partition.across_mean, 1e-10)

    def test_measurement_dilation_moves_local_coherence_into_correlations(self) -> None:
        plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        result = measurement_dilation_z(plus)

        self.assertAlmostEqual(result.global_purity, 1.0, places=10)
        self.assertAlmostEqual(result.system_purity, 0.5, places=10)
        self.assertAlmostEqual(result.system_coherence_l1, 0.0, places=10)
        self.assertAlmostEqual(result.mutual_information_bits, 2.0, places=10)

    def test_relation_hamiltonian_generates_entanglement(self) -> None:
        couplings = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        hamiltonian = relation_hamiltonian(couplings)
        evolved = evolve_state(basis_state("00"), hamiltonian, np.pi / 4.0)
        matrix = pairwise_mutual_information(evolved, [2, 2])
        self.assertAlmostEqual(matrix[0, 1], 2.0, places=9)

    def test_heat_trace_and_spectral_dimension_are_finite(self) -> None:
        weights = np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        )
        laplacian = weighted_laplacian(weights)
        times = np.logspace(-2, 2, 41)
        trace, dimension = spectral_dimension(laplacian, times)

        self.assertTrue(np.all(np.isfinite(trace)))
        self.assertTrue(np.all(np.isfinite(dimension)))
        self.assertTrue(np.all(np.diff(trace) <= 1e-12))
        self.assertTrue(np.all(trace > 0.0))


if __name__ == "__main__":
    unittest.main()
