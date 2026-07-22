from __future__ import annotations

import dataclasses
import unittest
from itertools import product

import numpy as np

from cyz_m.jet_hankel import (
    AugmentedJetHankel,
    FlatExtensionAudit,
    build_augmented_jet_hankel,
    build_jet_letter,
    flat_extension_audit,
)


def _words_up_to(length: int, letter: str = "T") -> list[tuple[str, ...]]:
    """Powers of a single autonomous letter, from the identity word up to `length`."""
    return [tuple([letter] * k) for k in range(length + 1)]


# ---------------------------------------------------------------------------
# Fixture (a): R3's visible-plus-delayed-hidden-chain family.
# A 1-dimensional visible subsystem (constant statistic) direct-summed with a
# hidden classical chain e_0 -> ... -> e_N whose final theta-dependent branch is
# invisible until step N. At small horizons the augmented Hankel is a flat rank-1
# plateau; only a horizon crossing N reveals the mode.
# ---------------------------------------------------------------------------
def _delayed_chain_letter(chain_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    n_states = chain_length + 3  # visible(0), e_0..e_N (1..N+1), sink (N+2)
    visible = 0

    def state(index: int) -> int:
        return 1 + index

    sink = chain_length + 2

    def transition(theta: float) -> np.ndarray:
        matrix = np.zeros((n_states, n_states))
        matrix[visible, visible] = 1.0
        for i in range(chain_length - 1):
            matrix[state(i + 1), state(i)] = 1.0
        matrix[sink, state(chain_length - 1)] = 0.5 - theta
        matrix[state(chain_length), state(chain_length - 1)] = 0.5 + theta
        matrix[state(chain_length), state(chain_length)] = 1.0
        matrix[sink, sink] = 1.0
        return matrix

    t0 = transition(0.0)
    d_transition = np.zeros((n_states, n_states))
    d_transition[sink, state(chain_length - 1)] = -1.0
    d_transition[state(chain_length), state(chain_length - 1)] = 1.0

    v0 = np.zeros(n_states)
    v0[visible] = 1.0
    v0[state(0)] = 0.5
    dv = np.zeros(n_states)

    detect_visible = np.zeros(n_states)
    detect_visible[visible] = 1.0
    detect_final = np.zeros(n_states)
    detect_final[state(chain_length)] = 1.0
    readouts = [(detect_visible, np.zeros(n_states)), (detect_final, np.zeros(n_states))]
    return t0, d_transition, v0, dv, readouts


def _delayed_chain_hankel(chain_length: int, prefix_len: int, suffix_len: int) -> AugmentedJetHankel:
    t0, d_transition, v0, dv, readouts = _delayed_chain_letter(chain_length)
    letter = build_jet_letter(t0, d_transition)
    return build_augmented_jet_hankel(
        (v0, dv),
        {"T": letter},
        readouts,
        _words_up_to(prefix_len),
        _words_up_to(suffix_len),
    )


# ---------------------------------------------------------------------------
# Fixture (b): small known-order positive control -- the classical 4-state chain
# whose augmented (zero+first) jet reachable module has order exactly 5
# (dim J_k = 1,2,3,4,5,5 in run_c7_jet_checks.py CHECK 2).
# ---------------------------------------------------------------------------
def _four_state_hankel(prefix_len: int, suffix_len: int) -> AugmentedJetHankel:
    def transition(theta: float) -> np.ndarray:
        matrix = np.zeros((4, 4))
        matrix[1, 0] = 1.0
        matrix[2, 1] = 1.0
        matrix[2, 2] = 0.5 - theta
        matrix[3, 2] = 0.5 + theta
        matrix[3, 3] = 1.0
        return matrix

    t0 = transition(0.0)
    d_transition = np.zeros((4, 4))
    d_transition[2, 2] = -1.0
    d_transition[3, 2] = 1.0
    letter = build_jet_letter(t0, d_transition)
    v0 = np.array([1.0, 0.0, 0.0, 0.0])
    effect = np.array([0.0, 0.0, 0.0, 1.0])
    return build_augmented_jet_hankel(
        (v0, np.zeros(4)),
        {"T": letter},
        [(effect, np.zeros(4))],
        _words_up_to(prefix_len),
        _words_up_to(suffix_len),
    )


# ---------------------------------------------------------------------------
# Fixture (c): the CZ - R_x(delta) - CZ comb jets. The memory qubit E, left in
# |0> by the S1-write branch (<+|_S1 CZ_{S1,E} |+>_S1 |+>_E ~ |0>), evolves under
# the R_x(delta) link and is read by the S2 slot. Its augmented jet (w.r.t.
# delta) traces a genuine orbit, so the atomic (depth-1) Hankel does not
# flat-extend to depth 2 -- growth must be detected (no false pass).
# ---------------------------------------------------------------------------
def _comb_hankel(delta0: float, depth: int) -> AugmentedJetHankel:
    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    basis = [identity, pauli_x, pauli_y, pauli_z]

    def coords(rho: np.ndarray) -> np.ndarray:
        return np.array([np.real(np.trace(op @ rho)) for op in basis])

    def r_x(angle: float) -> np.ndarray:
        return np.cos(angle / 2) * identity - 1j * np.sin(angle / 2) * pauli_x

    def d_r_x(angle: float) -> np.ndarray:
        return -0.5 * np.sin(angle / 2) * identity - 0.5j * np.cos(angle / 2) * pauli_x

    def channel_jet(unitary: np.ndarray, d_unitary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zeroth = np.zeros((4, 4))
        first = np.zeros((4, 4))
        for col, pin in enumerate(basis):
            rho = 0.5 * pin
            zeroth[:, col] = coords(unitary @ rho @ unitary.conj().T)
            first[:, col] = coords(
                d_unitary @ rho @ unitary.conj().T + unitary @ rho @ d_unitary.conj().T
            )
        return zeroth, first

    t0, d_link = channel_jet(r_x(delta0), d_r_x(delta0))
    letter = build_jet_letter(t0, d_link)
    seed = coords(0.5 * (identity + pauli_z))  # |0>, the S1-write branch memory
    read_y = np.array([0.0, 0.0, 1.0, 0.0])
    read_z = np.array([0.0, 0.0, 0.0, 1.0])
    return build_augmented_jet_hankel(
        (seed, np.zeros(4)),
        {"L": letter},
        [(read_y, np.zeros(4)), (read_z, np.zeros(4))],
        _words_up_to(depth, letter="L"),
        _words_up_to(depth, letter="L"),
    )


# ---------------------------------------------------------------------------
# Fixture (d): a two-letter, purely zeroth-order realization of minimal order 3.
# Both first-order jet blocks are zero, so the augmented span is the zeroth-order
# span. The seed reaches all of R^3 via the length-<=1 words {(), (A,), (B,)} and
# the readout observes all of R^3 via the same suffixes, so the Hankel of that
# max-length-1 block already SATURATES at rank 3 -- while covering neither every
# word through length n_J-1=2 nor reaching horizon 3. It is the canonical
# rank-saturation-without-horizon-exhaustion witness (Lemma 2.4).
# ---------------------------------------------------------------------------
_SAT_A0 = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, -1.0, -2.0]])
_SAT_B0 = np.array([[0.0, -1.0, 2.0], [0.0, -2.0, 0.0], [1.0, -2.0, 1.0]])
_SAT_SEED = np.array([1.0, 0.0, 0.0])
_SAT_READOUT = np.array([-2.0, 0.0, 2.0])


def _saturating_two_letter_hankel(words) -> AugmentedJetHankel:
    letter_a = build_jet_letter(_SAT_A0, np.zeros((3, 3)))
    letter_b = build_jet_letter(_SAT_B0, np.zeros((3, 3)))
    word_list = [tuple(w) for w in words]
    return build_augmented_jet_hankel(
        (_SAT_SEED, np.zeros(3)),
        {"A": letter_a, "B": letter_b},
        [(_SAT_READOUT, np.zeros(3))],
        word_list,
        word_list,
    )


def _all_words_up_to(max_len: int, alphabet=("A", "B")) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for length in range(max_len + 1):
        out.extend(tuple(c) for c in product(alphabet, repeat=length))
    return out


class BuildAugmentedJetHankelTests(unittest.TestCase):
    def test_factorization_matches_hankel(self) -> None:
        data = _four_state_hankel(3, 3)
        np.testing.assert_allclose(
            data.hankel, data.observability @ data.reachability, atol=1e-12
        )
        # rows = readouts (1) * suffixes (4) * order blocks (2); cols = prefixes (4).
        self.assertEqual(data.hankel.shape, (1 * 4 * 2, 4))
        self.assertFalse(data.target_dimension_argument_used)

    def test_build_jet_letter_is_block_triangular(self) -> None:
        t0 = np.array([[0.0, 0.0], [1.0, 0.0]])
        dt = np.array([[0.0, 1.0], [0.0, 0.0]])
        lift = build_jet_letter(t0, dt)
        np.testing.assert_array_equal(lift[:2, :2], t0)
        np.testing.assert_array_equal(lift[2:, 2:], t0)
        np.testing.assert_array_equal(lift[2:, :2], dt)
        np.testing.assert_array_equal(lift[:2, 2:], np.zeros((2, 2)))

    def test_rejects_dimension_mismatched_letter(self) -> None:
        with self.assertRaises(ValueError):
            build_augmented_jet_hankel(
                (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
                {"T": np.eye(3)},  # wrong size: seed jet dim is 4
                [(np.array([1.0, 0.0]), np.array([0.0, 0.0]))],
                [()],
                [()],
            )


class FlatExtensionAuditTests(unittest.TestCase):
    def test_delayed_chain_small_horizon_no_bound_is_inconclusive(self) -> None:
        # Fixture (a): a flat rank-1 plateau at horizon h=2. Without an order
        # bound the audit must never certify -- a plateau certifies nothing.
        data = _delayed_chain_hankel(chain_length=8, prefix_len=2, suffix_len=2)
        audit = flat_extension_audit(data, declared_order_bound=None)
        self.assertEqual(audit.numerical_rank, 1)
        self.assertLess(audit.max_residual, 1e-9)
        self.assertEqual(audit.status, "inconclusive_no_order_bound")

    def test_delayed_chain_underhorizon_bound_must_not_certify(self) -> None:
        # Fixture (a): declaring an order bound n_J = 5 < true order (chain 8) at
        # horizon h=2 must NOT certify -- the horizons do not exhaust the bound.
        data = _delayed_chain_hankel(chain_length=8, prefix_len=2, suffix_len=2)
        audit = flat_extension_audit(data, declared_order_bound=5)
        self.assertFalse(audit.horizons_exhaust_bound)
        self.assertNotEqual(audit.status, "certified_for_declared_class")
        self.assertEqual(audit.status, "inconclusive_no_order_bound")

    def test_delayed_chain_horizon_crossing_activation_detects_growth(self) -> None:
        # Fixture (a): horizons whose one-letter extension crosses depth N=8
        # expose the hidden mode. The base block (prefix words up to length 7)
        # is still flat, but its one-letter prefix extension reaches depth 8.
        data = _delayed_chain_hankel(chain_length=8, prefix_len=7, suffix_len=0)
        audit = flat_extension_audit(data, declared_order_bound=None)
        self.assertEqual(audit.status, "growth_detected")
        self.assertGreater(audit.max_residual, 1e-3)
        # falsification stands even when an (untrue) order bound is declared.
        with_bound = flat_extension_audit(data, declared_order_bound=5)
        self.assertEqual(with_bound.status, "growth_detected")
        prefix_letter, prefix_resid = audit.per_letter_prefix_residual[0]
        self.assertEqual(prefix_letter, "T")
        self.assertGreater(prefix_resid, 1e-3)

    def test_four_state_positive_control_certifies_for_declared_class(self) -> None:
        # Fixture (b): known order 5, horizons 0..5 exhaust the bound and the
        # augmented reachable module is closed -> certified_for_declared_class.
        data = _four_state_hankel(prefix_len=5, suffix_len=5)
        audit = flat_extension_audit(data, declared_order_bound=5)
        self.assertEqual(audit.numerical_rank, 5)
        self.assertTrue(audit.horizons_exhaust_bound)
        self.assertLess(audit.max_residual, 1e-9)
        self.assertLess(audit.largest_discarded_singular_value, 1e-9)
        self.assertEqual(audit.status, "certified_for_declared_class")

    def test_rank_exceeding_declared_bound_is_inconsistent_not_certified(self) -> None:
        # Hardening guard: the four-state module has Hankel rank 5, so a caller
        # declaring order bound 4 is asserting a class that cannot produce the
        # data. The audit must expose the contradiction, never certify into an
        # empty class.
        data = _four_state_hankel(prefix_len=5, suffix_len=5)
        audit = flat_extension_audit(data, declared_order_bound=4)
        self.assertEqual(audit.numerical_rank, 5)
        self.assertEqual(audit.status, "inconsistent_with_declared_bound")

    def test_four_state_underhorizon_detects_ongoing_growth(self) -> None:
        # The same order-5 module is still opening up at horizon 2 (dim J_k has
        # only reached ~3), so the one-letter extension correctly reports growth
        # rather than a premature certificate: no false pass below the order.
        data = _four_state_hankel(prefix_len=2, suffix_len=2)
        audit = flat_extension_audit(data, declared_order_bound=5)
        self.assertFalse(audit.horizons_exhaust_bound)
        self.assertEqual(audit.status, "growth_detected")

    def test_flat_but_underhorizon_declared_bound_is_inconclusive(self) -> None:
        # The flat-plateau-with-unexhausted-bound path: the delayed chain is a
        # genuine flat rank-1 plateau at horizon 2, yet a declared bound n_J=5
        # whose horizons are not exhausted must still stay inconclusive.
        data = _delayed_chain_hankel(chain_length=8, prefix_len=2, suffix_len=2)
        audit = flat_extension_audit(data, declared_order_bound=5)
        self.assertTrue(audit.flatness_holds)
        self.assertFalse(audit.horizons_exhaust_bound)
        self.assertEqual(audit.status, "inconclusive_no_order_bound")

    def test_comb_jets_detect_growth_atomic_to_depth_two(self) -> None:
        # Fixture (c): the CZ-R_x-CZ comb memory jets do not plateau at the
        # atomic (depth-1) horizon -- growth must be detected, no false pass.
        atomic = flat_extension_audit(_comb_hankel(0.7, depth=1), declared_order_bound=None)
        self.assertEqual(atomic.status, "growth_detected")
        self.assertGreater(atomic.max_residual, 1e-3)
        # A declared bound does not launder the growth into a certificate.
        bounded = flat_extension_audit(_comb_hankel(0.7, depth=1), declared_order_bound=2)
        self.assertEqual(bounded.status, "growth_detected")

    def test_audit_reports_are_frozen_and_target_blind(self) -> None:
        audit = flat_extension_audit(_four_state_hankel(3, 3), declared_order_bound=None)
        self.assertIsInstance(audit, FlatExtensionAudit)
        self.assertFalse(audit.target_dimension_argument_used)
        with self.assertRaises(Exception):
            audit.status = "tampered"  # type: ignore[misc]


class RankSaturationRouteTests(unittest.TestCase):
    """Theorem T2''' change #2 / Lemma 2.4: rank saturation exhausts the declared
    order, so certification no longer requires horizon exhaustion."""

    def test_rank_saturation_certifies_without_horizon_exhaustion(self) -> None:
        # Fixture (a): minimal order n=3, fed the NON-exhaustive length-<=1 block
        # {(), (A,), (B,)} whose Hankel still saturates at rank 3. The horizons do
        # not reach the bound and the block covers neither every word through
        # length n_J-1=2, yet the rank-saturation route certifies at order 3.
        block = _all_words_up_to(1)  # {(), (A,), (B,)}
        data = _saturating_two_letter_hankel(block)
        audit = flat_extension_audit(data, declared_order_bound=3)
        self.assertEqual(audit.numerical_rank, 3)
        self.assertLess(audit.max_residual, 1e-9)
        self.assertFalse(audit.horizons_exhaust_bound)  # max word length 1 < 3
        self.assertFalse(audit.all_words_covered)  # AA, AB, BA, BB absent
        self.assertEqual(audit.status, "certified_for_declared_class")
        self.assertEqual(audit.certified_order, 3)

    def test_exhaustive_order_corollary_certifies_at_observed_rank(self) -> None:
        # The expensive alternative (Lemma 2.3): with a declared bound n_J=4 but
        # only rank 3, certification is allowed ONLY because prefixes and suffixes
        # cover EVERY word through length n_J-1=3. It certifies at order r=3.
        block = _all_words_up_to(3)
        data = _saturating_two_letter_hankel(block)
        audit = flat_extension_audit(data, declared_order_bound=4)
        self.assertEqual(audit.numerical_rank, 3)
        self.assertTrue(audit.all_words_covered)
        self.assertFalse(audit.horizons_exhaust_bound)
        self.assertEqual(audit.status, "certified_for_declared_class")
        self.assertEqual(audit.certified_order, 3)

    def test_sparse_plateau_below_saturation_stays_inconclusive(self) -> None:
        # Fixture (b): rank 3 < declared bound 5 WITHOUT full all-words coverage
        # (a sparse sub-saturation plateau) certifies nothing (Lemma 2.10).
        block = _all_words_up_to(1)
        data = _saturating_two_letter_hankel(block)
        audit = flat_extension_audit(data, declared_order_bound=5)
        self.assertEqual(audit.numerical_rank, 3)
        self.assertLess(audit.max_residual, 1e-9)
        self.assertFalse(audit.all_words_covered)
        self.assertIsNone(audit.certified_order)
        self.assertEqual(audit.status, "inconclusive_no_order_bound")


class BoundaryFlatnessTests(unittest.TestCase):
    """SF3 boundary flatness: empty-prefix columns c_i and empty-suffix rows
    r_lambda must lie in the core column/row spaces."""

    def test_boundary_residuals_present_and_clean_for_valid_data(self) -> None:
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        audit = flat_extension_audit(data, declared_order_bound=3)
        # The report exposes SF3 residuals for the boundary column and rows.
        self.assertTrue(audit.empty_prefix_column_residual)
        self.assertTrue(audit.empty_suffix_row_residual)
        self.assertTrue(audit.boundary_flatness_holds)
        self.assertLess(audit.max_boundary_residual, 1e-9)

    def test_boundary_flatness_violation_is_detected(self) -> None:
        # Fixture (c): corrupt an empty-prefix column so it leaves the core column
        # space. SF2 shift flatness still holds, but SF3 must catch the boundary
        # escape and the verdict must be growth_detected (never a false pass).
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        u_full, singular, _ = np.linalg.svd(data.hankel, full_matrices=True)
        rank = int(np.sum(singular > 1e-9))
        corrupted = data.empty_prefix_columns.copy()
        corrupted[:, 0] = corrupted[:, 0] + u_full[:, rank]  # add an out-of-span mode
        bad = dataclasses.replace(data, empty_prefix_columns=corrupted)
        audit = flat_extension_audit(bad, declared_order_bound=3)
        self.assertFalse(audit.boundary_flatness_holds)
        self.assertGreater(audit.max_boundary_residual, 1e-3)
        self.assertEqual(audit.status, "growth_detected")


class NoisyModeTheoremPTests(unittest.TestCase):
    """Theorem P (P1-P2) certificate-state-table statuses."""

    def test_clean_case_is_certified_with_bounds(self) -> None:
        # Fixture (d) clean: saturating rank-3 block with tiny per-letter errors.
        # Delta_H > 0 and every certified flatness upper bound sits within the
        # model tolerance -> CERTIFIED_WITH_BOUNDS.
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        audit = flat_extension_audit(
            data, declared_order_bound=3, eps_H=1e-6, eps_a=1e-6, model_tolerance=1e-3
        )
        self.assertTrue(audit.noisy_mode)
        self.assertGreater(audit.rank_gap_margin, 0.0)
        self.assertIsNotNone(audit.subspace_bound)
        self.assertLessEqual(audit.certified_flatness_upper_bound, audit.model_tolerance)
        self.assertEqual(audit.status, "CERTIFIED_WITH_BOUNDS")
        self.assertEqual(audit.certified_order, 3)

    def test_nonpositive_rank_gap_is_inconclusive_rank_gap(self) -> None:
        # Fixture (d) gap: an error floor eps_H larger than the certified singular
        # gap drives Delta_H <= 0 -> INCONCLUSIVE_RANK_GAP (no clipped pass).
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        audit = flat_extension_audit(
            data, declared_order_bound=3, eps_H=100.0, eps_a=1e-6
        )
        self.assertTrue(audit.noisy_mode)
        self.assertLessEqual(audit.rank_gap_margin, 0.0)
        self.assertEqual(audit.status, "INCONCLUSIVE_RANK_GAP")

    def test_no_order_bound_is_inconclusive_no_order_exhaustion(self) -> None:
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        audit = flat_extension_audit(
            data, declared_order_bound=None, eps_H=1e-6, eps_a=1e-6
        )
        self.assertEqual(audit.status, "INCONCLUSIVE_NO_ORDER_EXHAUSTION")

    def test_boundary_escape_is_failed_model_class(self) -> None:
        # A decisively out-of-span boundary column (certified residual lower bound
        # above the model tolerance) -> FAILED_MODEL_CLASS.
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        u_full, singular, _ = np.linalg.svd(data.hankel, full_matrices=True)
        rank = int(np.sum(singular > 1e-9))
        corrupted = data.empty_prefix_columns.copy()
        corrupted[:, 0] = corrupted[:, 0] + u_full[:, rank]
        bad = dataclasses.replace(data, empty_prefix_columns=corrupted)
        audit = flat_extension_audit(
            bad, declared_order_bound=3, eps_H=1e-6, eps_a=1e-6, model_tolerance=1e-3
        )
        self.assertEqual(audit.status, "FAILED_MODEL_CLASS")
        self.assertGreater(audit.certified_flatness_lower_bound, audit.model_tolerance)

    def test_noisy_mode_requires_both_error_inputs(self) -> None:
        data = _saturating_two_letter_hankel(_all_words_up_to(1))
        with self.assertRaises(ValueError):
            flat_extension_audit(data, declared_order_bound=3, eps_H=1e-6)


if __name__ == "__main__":
    unittest.main()
