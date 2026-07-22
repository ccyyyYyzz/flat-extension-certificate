"""Augmented (zeroth + first order) jet-Hankel flat-extension audit.

This module implements the operational certificate of the saturated first-jet
flat-extension theorem (Theorem T2''' of the external-brain package
``EB-R4-CYZM-8M1X5T``, GitHub issue #5), together with its robust perturbation
statement (Theorem P) and the resource accounting of Theorem R.

Two certification routes (Theorem T2''', changes #2 and Lemma 2.4 / Lemma 2.3):

* **Rank-saturation route (cheap, ``O(|Sigma| n_J^2)``).**  For ``y in C(n_J)``,
  every finite Hankel block obeys ``rank H <= n(y) <= n_J`` (Lemma 2.2/2.4).  If
  the observed ``rank H == n_J`` with a certified rank gap, the declared order is
  *exhausted*: a finite block of rank ``n_J`` pins the minimal order, and no
  additional reachable/observable mode is compatible with the class promise.
  Certification then holds **even when the tested prefix/suffix horizons are not
  exhausted** -- horizon length is no longer required.

* **Exhaustive-order corollary (expensive alternative, exponential in ``n_J``).**
  If ``rank H = r < n_J`` but the prefixes contain *every* word through length
  ``n_J - 1`` (applied to every seed) and the suffixes contain *every* word
  through length ``n_J - 1`` (applied to every readout), then Lemma 2.3 gives
  ``r = n(y)`` already, so certification holds with ``r`` in place of ``n_J``.
  This requires genuine all-words coverage, not merely a maximum word length.

A single sparse plateau with ``rank < n_J`` and no complete-basis certificate
certifies nothing (Lemma 2.10, the delayed-chain obstruction): the verdict is
inconclusive.

Flatness is two-sided and includes boundaries (Definition 1.5):

* **(SF1) rank saturation** -- ``rank H = n_J`` (or the exhaustive corollary).
* **(SF2) two-sided shift flatness** -- ``(I - O O^+) H_a = 0`` and
  ``H_a (I - R^+ R) = 0`` for every letter ``a``.
* **(SF3) boundary flatness** -- empty-prefix columns ``c_i`` and empty-suffix
  rows ``r_lambda`` lie in the core column/row spaces: ``(I - O O^+) c_i = 0`` and
  ``r_lambda (I - R^+ R) = 0``.

Status names.  The audit runs in two modes.

* **Exact mode** (default; no error inputs) keeps the historical lower-case
  verdicts so existing callers are unaffected:

  ==================================  =========================================
  exact-mode status                    certificate-state-table analogue
  ==================================  =========================================
  ``'certified_for_declared_class'``   ``CERTIFIED_WITH_BOUNDS``
  ``'growth_detected'``                ``FAILED_MODEL_CLASS``
  ``'inconclusive_no_order_bound'``    ``INCONCLUSIVE_NO_ORDER_EXHAUSTION``
  ``'inconsistent_with_declared_bound'`` (rank exceeds the declared class)
  ==================================  =========================================

* **Noisy mode** (engaged when ``eps_H``/``eps_a`` are supplied) adopts the
  mandatory certificate-state table of Theorem P and returns the upper-case
  statuses ``INCONCLUSIVE_NO_ORDER_EXHAUSTION``, ``INCONCLUSIVE_RANK_GAP``,
  ``FAILED_MODEL_CLASS``, ``INCONCLUSIVE_FLATNESS``, and ``CERTIFIED_WITH_BOUNDS``
  (Theorem P, P1--P2 with Wedin/Weyl constants).

The audit never consumes a target Hilbert-space dimension:
``target_dimension_argument_used`` is fixed ``False``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# A word over the lifted alphabet is a tuple of letter names; the empty tuple is
# the identity word. Convention: for ``word = (a_1, ..., a_k)`` the letter
# ``a_1`` acts on the seed first, so ``A_word = A_{a_k} @ ... @ A_{a_1}``.
Word = Sequence[str]
Jet = tuple[FloatArray, FloatArray]


@dataclass(frozen=True)
class AugmentedJetHankel:
    """Assembled augmented jet-Hankel block and its factorization.

    ``hankel`` equals ``observability @ reachability``; the factors are retained
    so the audit can form the one-letter shifted blocks
    ``observability @ A_letter @ reachability`` without rebuilding words.

    ``empty_prefix_columns`` are the boundary columns ``c_i`` (readout jets paired
    through every suffix with the bare seed, ``A_epsilon @ seed``) and
    ``empty_suffix_rows`` are the boundary rows ``r_lambda`` (bare readout jets
    paired with every reachable prefix). They feed the SF3 boundary-flatness
    residuals of the audit.
    """

    hankel: FloatArray
    observability: FloatArray
    reachability: FloatArray
    letter_names: tuple[str, ...]
    letter_maps: tuple[FloatArray, ...]
    prefixes: tuple[tuple[str, ...], ...]
    suffixes: tuple[tuple[str, ...], ...]
    readout_count: int
    jet_dimension: int
    empty_prefix_columns: FloatArray
    empty_suffix_rows: FloatArray
    order_blocks: int = 2
    target_dimension_argument_used: bool = False


@dataclass(frozen=True)
class FlatExtensionAudit:
    """Verdict of the augmented jet-Hankel flat-extension audit."""

    status: str
    numerical_rank: int
    singular_values: tuple[float, ...]
    singular_gap: float
    largest_discarded_singular_value: float
    per_letter_prefix_residual: tuple[tuple[str, float], ...]
    per_letter_suffix_residual: tuple[tuple[str, float], ...]
    per_letter_shift_residual: tuple[tuple[str, float], ...]
    empty_prefix_column_residual: tuple[tuple[str, float], ...]
    empty_suffix_row_residual: tuple[tuple[str, float], ...]
    max_residual: float
    max_shift_residual: float
    max_boundary_residual: float
    flatness_holds: bool
    boundary_flatness_holds: bool
    declared_order_bound: int | None
    max_prefix_length: int
    max_suffix_length: int
    horizons_exhaust_bound: bool
    all_words_covered: bool
    certified_order: int | None
    tolerance: float
    # Noisy-mode (Theorem P) report fields; ``None`` in exact mode.
    noisy_mode: bool = False
    rank_gap_margin: float | None = None
    subspace_bound: float | None = None
    certified_flatness_upper_bound: float | None = None
    certified_flatness_lower_bound: float | None = None
    model_tolerance: float | None = None
    target_dimension_argument_used: bool = False


def build_jet_letter(zeroth: FloatArray, first: FloatArray) -> FloatArray:
    """Return the block-triangular first-jet lift ``[[T0, 0], [dT, T0]]``.

    This is the lift of ``examples/run_c7_jet_checks.py`` CHECK 2: it acts on an
    augmented jet ``(v0, dv)`` by ``(v0, dv) -> (T0 v0, T0 dv + dT v0)``.
    """
    t0 = np.asarray(zeroth, dtype=np.float64)
    dt = np.asarray(first, dtype=np.float64)
    if t0.ndim != 2 or t0.shape[0] != t0.shape[1]:
        raise ValueError("Zeroth-order block T0 must be a square matrix.")
    if dt.shape != t0.shape:
        raise ValueError("First-order block dT must match T0's shape.")
    if not np.all(np.isfinite(t0)) or not np.all(np.isfinite(dt)):
        raise ValueError("Jet-letter blocks must be finite.")
    n = t0.shape[0]
    lift = np.zeros((2 * n, 2 * n), dtype=np.float64)
    lift[:n, :n] = t0
    lift[n:, n:] = t0
    lift[n:, :n] = dt
    return lift


def _as_jet_vector(seed_jet: Jet, name: str) -> tuple[FloatArray, int]:
    if not (isinstance(seed_jet, (tuple, list)) and len(seed_jet) == 2):
        raise ValueError(f"{name} must be a (v0, dv) pair of equal-length vectors.")
    v0 = np.asarray(seed_jet[0], dtype=np.float64).reshape(-1)
    dv = np.asarray(seed_jet[1], dtype=np.float64).reshape(-1)
    if v0.shape != dv.shape or v0.size == 0:
        raise ValueError(f"{name} components v0 and dv must be nonempty and equal length.")
    if not np.all(np.isfinite(v0)) or not np.all(np.isfinite(dv)):
        raise ValueError(f"{name} must be finite.")
    return np.concatenate([v0, dv]), int(v0.size)


def _word_map(
    word: Word, letters: Mapping[str, FloatArray], dimension: int
) -> FloatArray:
    matrix = np.eye(dimension, dtype=np.float64)
    for letter in word:
        if letter not in letters:
            raise ValueError(f"Word uses unknown letter {letter!r}.")
        matrix = letters[letter] @ matrix
    return matrix


def build_augmented_jet_hankel(
    seed_jet: Jet,
    lifted_letters: Mapping[str, FloatArray],
    readout_jets: Sequence[Jet],
    prefixes: Sequence[Word],
    suffixes: Sequence[Word],
) -> AugmentedJetHankel:
    """Assemble the stacked zeroth+first-order jet-Hankel block.

    Rows are indexed by ``(readout, suffix, order in {0, 1})``: the covector for
    a readout jet ``(e0, de)`` at order 0 is ``[e0, 0]`` and at order 1 is
    ``[de, e0]`` (the canonical first-jet pairing ``dp = de . v0 + e0 . dv``),
    propagated through the suffix word. Columns are indexed by prefix words; the
    column for prefix ``p`` is the reachable augmented jet ``A_p @ seed``. Each
    Hankel entry is therefore the readout-jet-paired value of
    ``A_suffix @ A_prefix`` applied to the seed, with both order blocks stacked.

    The boundary columns ``c_i`` (empty prefix) and boundary rows ``r_lambda``
    (empty suffix) are computed alongside for the SF3 boundary-flatness residuals.
    """
    if not lifted_letters:
        raise ValueError("Provide at least one lifted alphabet letter.")
    if not readout_jets:
        raise ValueError("Provide at least one readout jet.")
    if not prefixes or not suffixes:
        raise ValueError("Provide at least one prefix word and one suffix word.")

    seed_vec, n = _as_jet_vector(seed_jet, "seed_jet")
    jet_dim = 2 * n

    letters: dict[str, FloatArray] = {}
    for name, raw in lifted_letters.items():
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (jet_dim, jet_dim):
            raise ValueError(
                f"Lifted letter {name!r} must be a {jet_dim}x{jet_dim} block-"
                "triangular jet map matching the seed jet dimension."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Lifted letter {name!r} must be finite.")
        letters[name] = matrix
    letter_names = tuple(letters.keys())

    readout_covectors: list[FloatArray] = []
    for index, jet in enumerate(readout_jets):
        e0 = np.asarray(jet[0], dtype=np.float64).reshape(-1)
        de = np.asarray(jet[1], dtype=np.float64).reshape(-1)
        if e0.shape != (n,) or de.shape != (n,):
            raise ValueError(f"Readout jet #{index} must have (e0, de) of length {n}.")
        order0 = np.concatenate([e0, np.zeros(n)])
        order1 = np.concatenate([de, e0])
        readout_covectors.append(order0)
        readout_covectors.append(order1)

    prefix_words = tuple(tuple(word) for word in prefixes)
    suffix_words = tuple(tuple(word) for word in suffixes)

    reachability = np.column_stack(
        [_word_map(word, letters, jet_dim) @ seed_vec for word in prefix_words]
    )

    rows: list[FloatArray] = []
    for suffix in suffix_words:
        suffix_map = _word_map(suffix, letters, jet_dim)
        for covector_index in range(0, len(readout_covectors), 2):
            for order in range(2):
                covector = readout_covectors[covector_index + order]
                rows.append(covector @ suffix_map)
    observability = np.vstack(rows)

    hankel = observability @ reachability

    # SF3 boundary data (Definition 1.4 / 1.5):
    #   empty-prefix column c_i = observability @ (A_epsilon @ seed) = O @ seed;
    #   empty-suffix row  r_lambda = (bare readout covector) @ reachability.
    empty_prefix_columns = (observability @ seed_vec).reshape(-1, 1)
    empty_suffix_rows = np.vstack(
        [covector @ reachability for covector in readout_covectors]
    )

    return AugmentedJetHankel(
        hankel=np.asarray(hankel, dtype=np.float64),
        observability=np.asarray(observability, dtype=np.float64),
        reachability=np.asarray(reachability, dtype=np.float64),
        letter_names=letter_names,
        letter_maps=tuple(letters[name] for name in letter_names),
        prefixes=prefix_words,
        suffixes=suffix_words,
        readout_count=len(readout_jets),
        jet_dimension=n,
        empty_prefix_columns=np.asarray(empty_prefix_columns, dtype=np.float64),
        empty_suffix_rows=np.asarray(empty_suffix_rows, dtype=np.float64),
    )


def _relative_operator_residual(matrix: FloatArray, scale: float) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2) / max(scale, 1.0))


def _all_words_covered(
    words: Sequence[tuple[str, ...]], alphabet: Sequence[str], max_len: int
) -> bool:
    """Return ``True`` iff ``words`` contains *every* word over ``alphabet`` of
    length ``0 .. max_len`` (the genuine all-words-coverage flag required by the
    exhaustive-order corollary, not merely a maximum word length)."""
    if max_len < 0:
        return True
    present = {tuple(word) for word in words}
    for length in range(max_len + 1):
        for combo in product(alphabet, repeat=length):
            if combo not in present:
                return False
    return True


def flat_extension_audit(
    hankel_data: AugmentedJetHankel,
    *,
    declared_order_bound: int | None = None,
    tolerance: float = 1e-9,
    eps_H: float | None = None,
    eps_a: float | Mapping[str, float] | None = None,
    model_tolerance: float | None = None,
) -> FlatExtensionAudit:
    """Audit one-letter flat extension of an augmented jet-Hankel block.

    Exact-mode verdict rules (Theorem T2''', Lemma 2.4 / 2.3):

    * Any SF2 or SF3 residual above ``tolerance`` -> ``'growth_detected'``.
      Falsification of closure is always valid, with or without a declared bound.
    * ``declared_order_bound is None`` -> ``'inconclusive_no_order_bound'``: a
      finite plateau certifies nothing without an order promise.
    * ``rank H > declared_order_bound`` -> ``'inconsistent_with_declared_bound'``:
      the observed rank exceeds any realization of the declared class.
    * ``rank H == declared_order_bound`` with a clean rank gap ->
      ``'certified_for_declared_class'`` by the **rank-saturation route** (Lemma
      2.4), regardless of whether the tested horizons are exhausted.
    * ``rank H = r < declared_order_bound`` -> certified at order ``r`` **only**
      when the prefixes and suffixes cover every word through length
      ``n_J - 1`` (exhaustive-order corollary, Lemma 2.3); otherwise
      ``'inconclusive_no_order_bound'`` (a sparse sub-saturation plateau).

    Noisy mode (Theorem P) is engaged by supplying ``eps_H`` and ``eps_a``
    (a per-letter scalar or ``{letter: eps}`` mapping); optional ``model_tolerance``
    sets the admissible model-mismatch flatness residual (default ``100`` times the
    combined error floor). It returns the certificate-state-table statuses:
    ``INCONCLUSIVE_NO_ORDER_EXHAUSTION``, ``INCONCLUSIVE_RANK_GAP``,
    ``FAILED_MODEL_CLASS``, ``INCONCLUSIVE_FLATNESS``, ``CERTIFIED_WITH_BOUNDS``.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if declared_order_bound is not None and declared_order_bound < 1:
        raise ValueError("declared_order_bound must be a positive integer or None.")

    noisy_mode = eps_H is not None or eps_a is not None
    if noisy_mode and (eps_H is None or eps_a is None):
        raise ValueError("Noisy mode requires both eps_H and eps_a.")

    hankel = hankel_data.hankel
    observability = hankel_data.observability
    reachability = hankel_data.reachability
    empty_prefix_columns = hankel_data.empty_prefix_columns
    empty_suffix_rows = hankel_data.empty_suffix_rows

    u_matrix, singular, vt_matrix = np.linalg.svd(hankel, full_matrices=False)
    top = float(singular[0]) if singular.size else 0.0
    threshold = tolerance * max(top, 1.0)
    rank = int(np.sum(singular > threshold))
    largest_discarded = float(singular[rank]) if rank < singular.size else 0.0
    if rank == 0:
        singular_gap = float("inf")
    elif rank < singular.size and singular[rank] > 0:
        singular_gap = float(singular[rank - 1] / singular[rank])
    else:
        singular_gap = float("inf")

    left = u_matrix[:, :rank]
    right = vt_matrix[:rank, :].T
    column_reject = np.eye(hankel.shape[0]) - left @ left.T
    row_reject = np.eye(hankel.shape[1]) - right @ right.T
    hankel_pinv = np.linalg.pinv(hankel)

    # --- SF2: two-sided shift flatness (relative residuals) -----------------
    prefix_residuals: list[tuple[str, float]] = []
    suffix_residuals: list[tuple[str, float]] = []
    shift_residuals: list[tuple[str, float]] = []
    for name, letter in zip(hankel_data.letter_names, hankel_data.letter_maps):
        shifted = observability @ letter @ reachability
        prefix_residuals.append(
            (name, _relative_operator_residual(column_reject @ shifted, top))
        )
        suffix_residuals.append(
            (name, _relative_operator_residual(shifted @ row_reject, top))
        )
        induced = hankel @ (hankel_pinv @ shifted)
        shift_residuals.append(
            (name, _relative_operator_residual(shifted - induced, top))
        )

    # --- SF3: boundary flatness (relative residuals) ------------------------
    boundary_column_residuals: list[tuple[str, float]] = []
    for index in range(empty_prefix_columns.shape[1]):
        column = empty_prefix_columns[:, index]
        boundary_column_residuals.append(
            (f"c_{index}", _relative_operator_residual(column_reject @ column, top))
        )
    boundary_row_residuals: list[tuple[str, float]] = []
    for index in range(empty_suffix_rows.shape[0]):
        row = empty_suffix_rows[index, :]
        boundary_row_residuals.append(
            (f"r_{index}", _relative_operator_residual(row @ row_reject, top))
        )

    max_shift_residual = 0.0
    for residuals in (prefix_residuals, suffix_residuals, shift_residuals):
        for _, value in residuals:
            max_shift_residual = max(max_shift_residual, value)
    max_boundary_residual = 0.0
    for residuals in (boundary_column_residuals, boundary_row_residuals):
        for _, value in residuals:
            max_boundary_residual = max(max_boundary_residual, value)
    max_residual = max(max_shift_residual, max_boundary_residual)
    flatness_holds = bool(max_residual <= tolerance)
    boundary_flatness_holds = bool(max_boundary_residual <= tolerance)

    max_prefix_length = max(len(word) for word in hankel_data.prefixes)
    max_suffix_length = max(len(word) for word in hankel_data.suffixes)
    horizons_exhaust = bool(
        declared_order_bound is not None
        and max_prefix_length >= declared_order_bound
        and max_suffix_length >= declared_order_bound
    )
    clean_gap = bool(rank >= 1 and largest_discarded <= threshold)

    # All-words coverage through length n_J - 1 for the exhaustive-order corollary.
    alphabet = hankel_data.letter_names
    all_words_covered = False
    if declared_order_bound is not None:
        cover_len = declared_order_bound - 1
        all_words_covered = bool(
            _all_words_covered(hankel_data.prefixes, alphabet, cover_len)
            and _all_words_covered(hankel_data.suffixes, alphabet, cover_len)
        )

    common = dict(
        numerical_rank=rank,
        singular_values=tuple(float(value) for value in singular),
        singular_gap=singular_gap,
        largest_discarded_singular_value=largest_discarded,
        per_letter_prefix_residual=tuple(prefix_residuals),
        per_letter_suffix_residual=tuple(suffix_residuals),
        per_letter_shift_residual=tuple(shift_residuals),
        empty_prefix_column_residual=tuple(boundary_column_residuals),
        empty_suffix_row_residual=tuple(boundary_row_residuals),
        max_residual=max_residual,
        max_shift_residual=max_shift_residual,
        max_boundary_residual=max_boundary_residual,
        flatness_holds=flatness_holds,
        boundary_flatness_holds=boundary_flatness_holds,
        declared_order_bound=declared_order_bound,
        max_prefix_length=max_prefix_length,
        max_suffix_length=max_suffix_length,
        horizons_exhaust_bound=horizons_exhaust,
        all_words_covered=all_words_covered,
        tolerance=tolerance,
    )

    if noisy_mode:
        return _noisy_verdict(
            hankel_data,
            u_matrix,
            singular,
            vt_matrix,
            eps_H=float(eps_H),
            eps_a=eps_a,
            model_tolerance=model_tolerance,
            common=common,
        )

    # ---------------------- exact-mode verdict ------------------------------
    certified_order: int | None = None
    if not flatness_holds:
        status = "growth_detected"
    elif declared_order_bound is None:
        status = "inconclusive_no_order_bound"
    elif rank > declared_order_bound:
        # The observed rank already exceeds any realization of the declared
        # order: the declared class is contradicted, so certifying "for the
        # declared class" would be vacuous.
        status = "inconsistent_with_declared_bound"
    elif rank == declared_order_bound and clean_gap:
        # Rank-saturation route (Lemma 2.4): a finite block of rank n_J pins the
        # minimal order. No horizon-exhaustion requirement.
        status = "certified_for_declared_class"
        certified_order = declared_order_bound
    elif rank < declared_order_bound and clean_gap and all_words_covered:
        # Exhaustive-order corollary (Lemma 2.3): full all-words coverage through
        # length n_J - 1 makes the observed rank equal the minimal order.
        status = "certified_for_declared_class"
        certified_order = rank
    else:
        # Sparse sub-saturation plateau: certifies nothing (Lemma 2.10).
        status = "inconclusive_no_order_bound"

    return FlatExtensionAudit(
        status=status,
        certified_order=certified_order,
        noisy_mode=False,
        **common,
    )


def _noisy_verdict(
    hankel_data: AugmentedJetHankel,
    u_matrix: FloatArray,
    singular: FloatArray,
    vt_matrix: FloatArray,
    *,
    eps_H: float,
    eps_a: float | Mapping[str, float],
    model_tolerance: float | None,
    common: dict,
) -> FlatExtensionAudit:
    """Theorem P (robust saturated flat extension) verdict, P1--P2.

    Certifies the declared rank ``n`` via the Weyl/Wedin gap ``Delta_H`` and
    bounds the true (SF2/SF3) flatness residuals from the observed residuals plus
    the subspace-perturbation term ``b_H (||block|| + eps_a)``.
    """
    observability = hankel_data.observability
    reachability = hankel_data.reachability
    empty_prefix_columns = hankel_data.empty_prefix_columns
    empty_suffix_rows = hankel_data.empty_suffix_rows
    declared_order_bound = common["declared_order_bound"]

    if isinstance(eps_a, Mapping):
        eps_a_of = lambda name: float(eps_a.get(name, 0.0))
        eps_a_max = max((float(v) for v in eps_a.values()), default=0.0)
    else:
        scalar = float(eps_a)
        eps_a_of = lambda name: scalar
        eps_a_max = scalar

    if model_tolerance is None:
        model_tolerance = 100.0 * (eps_H + eps_a_max)

    def finish(status, **extra):
        base = dict(
            status=status,
            certified_order=None,
            noisy_mode=True,
            model_tolerance=float(model_tolerance),
            rank_gap_margin=None,
            subspace_bound=None,
            certified_flatness_upper_bound=None,
            certified_flatness_lower_bound=None,
        )
        base.update(extra)
        merged = {**common, **base}
        return FlatExtensionAudit(**merged)

    # No order promise -> cannot exhaust the order (table row 1).
    if declared_order_bound is None:
        return finish("INCONCLUSIVE_NO_ORDER_EXHAUSTION")

    n = declared_order_bound
    sigma_n = float(singular[n - 1]) if n - 1 < singular.size else 0.0
    sigma_np1 = float(singular[n]) if n < singular.size else 0.0
    delta_H = sigma_n - sigma_np1 - eps_H

    # P1: certified rank gap (Weyl). Non-positive -> cannot certify the rank.
    if delta_H <= 0:
        return finish("INCONCLUSIVE_RANK_GAP", rank_gap_margin=delta_H)

    b_H = min(1.0, eps_H / delta_H)

    # Leading-n singular subspaces (the Wedin-certified core spaces).
    left_n = u_matrix[:, :n]
    right_n = vt_matrix[:n, :].T
    proj_col = np.eye(observability.shape[0]) - left_n @ left_n.T
    proj_row = np.eye(reachability.shape[1]) - right_n @ right_n.T

    # SF2/SF3 certified true-residual brackets (Theorem P, Step 2):
    #   rho <= rho_obs + eps_a + b_H (||block|| + eps_a)   (upper)
    #   rho >= rho_obs - eps_a - b_H (||block|| + eps_a)    (lower, clamped >= 0)
    upper_bounds: list[float] = []
    lower_bounds: list[float] = []

    def bracket(block_norm: float, obs_residual: float, eps_block: float) -> None:
        slack = eps_block + b_H * (block_norm + eps_block)
        upper_bounds.append(obs_residual + slack)
        lower_bounds.append(max(0.0, obs_residual - slack))

    for name, letter in zip(hankel_data.letter_names, hankel_data.letter_maps):
        shifted = observability @ letter @ reachability
        block_norm = float(np.linalg.norm(shifted, ord=2))
        obs_left = float(np.linalg.norm(proj_col @ shifted, ord=2))
        obs_right = float(np.linalg.norm(shifted @ proj_row, ord=2))
        bracket(block_norm, obs_left, eps_a_of(name))
        bracket(block_norm, obs_right, eps_a_of(name))

    for index in range(empty_prefix_columns.shape[1]):
        column = empty_prefix_columns[:, index]
        block_norm = float(np.linalg.norm(column))
        obs_residual = float(np.linalg.norm(proj_col @ column))
        bracket(block_norm, obs_residual, eps_a_max)
    for index in range(empty_suffix_rows.shape[0]):
        row = empty_suffix_rows[index, :]
        block_norm = float(np.linalg.norm(row))
        obs_residual = float(np.linalg.norm(row @ proj_row))
        bracket(block_norm, obs_residual, eps_a_max)

    rho_upper = max(upper_bounds) if upper_bounds else 0.0
    rho_lower = max(lower_bounds) if lower_bounds else 0.0

    gap_extra = dict(
        rank_gap_margin=delta_H,
        subspace_bound=b_H,
        certified_flatness_upper_bound=rho_upper,
        certified_flatness_lower_bound=rho_lower,
    )

    # Model tolerance not separated from zero -> cannot adjudicate flatness.
    if model_tolerance <= 0:
        return finish("INCONCLUSIVE_FLATNESS", **gap_extra)

    if rho_upper <= model_tolerance:
        # Every certified residual upper bound sits within tolerance.
        return finish(
            "CERTIFIED_WITH_BOUNDS", certified_order=n, **gap_extra
        )
    if rho_lower > model_tolerance:
        # Some residual is certified (lower bound) to exceed the tolerance.
        return finish("FAILED_MODEL_CLASS", **gap_extra)
    # Tolerance falls inside [rho_lower, rho_upper]: not separable.
    return finish("INCONCLUSIVE_FLATNESS", **gap_extra)
