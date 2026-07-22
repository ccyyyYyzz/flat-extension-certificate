"""Event-individuation relativity for the finite CF-RSP core.

A matrix list is a presentation of a resolved event module, not an ontology of
absolute event individuals. This module separates ambient basis covariance,
event-frame covariance inside unresolved sectors, and residual factor gauge.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .factorization import (
    FactorizationCandidate,
    FactorizationSearchResult,
    algebra_commutator_defect,
    center_basis,
    generated_star_algebra,
)

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EventBasisAudit:
    """Numerical certificate for one event-frame transformation."""

    dimension: int
    invertible: bool
    smallest_singular_value: float
    sector_leakage: float
    metric_defect: float | None
    reality_defect: float | None
    allowed: bool


@dataclass(frozen=True)
class FactorNormalizerAudit:
    """Whether an ambient unitary preserves an ordered/unordered factor pair."""

    unitary_defect: float
    ordered_defect: float
    swapped_defect: float
    swap_allowed: bool
    allowed: bool


def _validate_events(events: Sequence[ComplexArray]) -> tuple[list[ComplexArray], int]:
    if not events:
        raise ValueError("At least one event is required.")
    arrays = [np.asarray(event, dtype=np.complex128) for event in events]
    dimension = arrays[0].shape[0]
    if dimension < 1 or any(event.shape != (dimension, dimension) for event in arrays):
        raise ValueError("Events must be equally sized square matrices.")
    if any(not np.all(np.isfinite(event)) for event in arrays):
        raise ValueError("Events must contain only finite values.")
    return arrays, dimension


def normalise_event_sectors(
    event_count: int, sectors: Sequence[Sequence[int]] | None
) -> tuple[tuple[int, ...], ...]:
    """Return a partition of event indices into nonempty resolved sectors."""
    if event_count < 1:
        raise ValueError("event_count must be positive.")
    if sectors is None:
        return tuple((index,) for index in range(event_count))
    normalised = tuple(tuple(sorted(int(index) for index in sector)) for sector in sectors)
    if not normalised or any(not sector for sector in normalised):
        raise ValueError("Event sectors must be nonempty.")
    flattened = [index for sector in normalised for index in sector]
    if sorted(flattened) != list(range(event_count)) or len(set(flattened)) != event_count:
        raise ValueError("Event sectors must partition every event index exactly once.")
    return normalised


def state_weighted_event_gram(
    events: Sequence[ComplexArray], state: ComplexArray
) -> ComplexArray:
    """Return G_ab = Tr(rho F_a^* F_b), the faithful-state event metric."""
    arrays, dimension = _validate_events(events)
    rho = np.asarray(state, dtype=np.complex128)
    if rho.shape != (dimension, dimension):
        raise ValueError("State/event dimensions do not match.")
    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise ValueError("State must be Hermitian.")
    if not np.isclose(np.trace(rho), 1.0, atol=1e-10):
        raise ValueError("State must have unit trace.")
    if float(np.min(np.linalg.eigvalsh(rho)).real) <= 0.0:
        raise ValueError("State must be faithful (positive definite).")
    gram = np.empty((len(arrays), len(arrays)), dtype=np.complex128)
    for left, first in enumerate(arrays):
        for right, second in enumerate(arrays):
            gram[left, right] = np.trace(rho @ first.conj().T @ second)
    return np.asarray((gram + gram.conj().T) / 2.0, dtype=np.complex128)


def mix_event_frame(
    events: Sequence[ComplexArray], transform: ComplexArray
) -> tuple[ComplexArray, ...]:
    """Change event frame by F'_a = sum_b F_b V_ba."""
    arrays, _ = _validate_events(events)
    matrix = np.asarray(transform, dtype=np.complex128)
    if matrix.shape != (len(arrays), len(arrays)):
        raise ValueError("Event-frame transform has the wrong shape.")
    if float(np.min(np.linalg.svd(matrix, compute_uv=False))) <= 0.0:
        raise ValueError("Event-frame transform must be invertible.")
    zero = np.zeros_like(arrays[0])
    return tuple(
        np.asarray(
            sum(
                (arrays[old] * matrix[old, new] for old in range(len(arrays))),
                zero.copy(),
            ),
            dtype=np.complex128,
        )
        for new in range(len(arrays))
    )


def simultaneous_unitary_presentation_change(
    events: Sequence[ComplexArray], state: ComplexArray, unitary: ComplexArray
) -> tuple[tuple[ComplexArray, ...], ComplexArray]:
    """Apply one passive ambient basis change to state and every event."""
    arrays, dimension = _validate_events(events)
    rho = np.asarray(state, dtype=np.complex128)
    u = np.asarray(unitary, dtype=np.complex128)
    if rho.shape != (dimension, dimension) or u.shape != (dimension, dimension):
        raise ValueError("Ambient dimensions do not match.")
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(u.conj().T @ u, identity, atol=1e-10):
        raise ValueError("Ambient basis changes must be unitary.")
    return (
        tuple(u @ event @ u.conj().T for event in arrays),
        np.asarray(u @ rho @ u.conj().T, dtype=np.complex128),
    )


def audit_event_basis_change(
    transform: ComplexArray,
    sectors: Sequence[Sequence[int]],
    *,
    metric: ComplexArray | None = None,
    require_real: bool = False,
    tolerance: float = 1e-9,
) -> EventBasisAudit:
    """Audit the stabiliser boundary for a fixed sector resolution.

    Without ``metric`` the allowed group is block-GL. Supplying a positive
    metric reduces it to the corresponding block-unitary group. ``require_real``
    further restricts it to the declared real form.
    """
    matrix = np.asarray(transform, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transform must be square.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    resolved = normalise_event_sectors(matrix.shape[0], sectors)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    smallest = float(singular_values[-1])
    invertible = smallest > tolerance

    allowed_mask = np.zeros(matrix.shape, dtype=bool)
    for sector in resolved:
        allowed_mask[np.ix_(sector, sector)] = True
    denominator = max(float(np.linalg.norm(matrix, ord="fro")), np.finfo(float).eps)
    sector_leakage = float(np.linalg.norm(matrix[~allowed_mask]) / denominator)

    metric_defect: float | None = None
    if metric is not None:
        gram = np.asarray(metric, dtype=np.complex128)
        if gram.shape != matrix.shape or not np.allclose(gram, gram.conj().T, atol=1e-10):
            raise ValueError("Metric must be a Hermitian matrix of matching size.")
        if float(np.min(np.linalg.eigvalsh(gram)).real) <= 0.0:
            raise ValueError("Metric must be positive definite.")
        metric_defect = float(
            np.linalg.norm(matrix.conj().T @ gram @ matrix - gram, ord="fro")
            / max(float(np.linalg.norm(gram, ord="fro")), np.finfo(float).eps)
        )

    reality_defect: float | None = None
    if require_real:
        reality_defect = float(np.linalg.norm(matrix.imag, ord="fro") / denominator)

    allowed = bool(
        invertible
        and sector_leakage <= tolerance
        and (metric_defect is None or metric_defect <= tolerance)
        and (reality_defect is None or reality_defect <= tolerance)
    )
    return EventBasisAudit(
        dimension=matrix.shape[0],
        invertible=invertible,
        smallest_singular_value=smallest,
        sector_leakage=sector_leakage,
        metric_defect=metric_defect,
        reality_defect=reality_defect,
        allowed=allowed,
    )


def similarity_star_defect(similarity: ComplexArray) -> float:
    """Zero exactly when Ad_S preserves the fixed adjoint on M_d."""
    matrix = np.asarray(similarity, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity must be square.")
    if float(np.min(np.linalg.svd(matrix, compute_uv=False))) <= 0.0:
        raise ValueError("Similarity must be invertible.")
    positive = matrix.conj().T @ matrix
    scalar = np.trace(positive) / matrix.shape[0]
    return float(
        np.linalg.norm(positive - scalar * np.eye(matrix.shape[0]), ord="fro")
        / max(float(np.linalg.norm(positive, ord="fro")), np.finfo(float).eps)
    )


def transpose_choi_min_eigenvalue(dimension: int) -> float:
    """Minimum Choi eigenvalue of transposition; negative for d >= 2."""
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    choi = np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128)
    for row in range(dimension):
        for column in range(dimension):
            matrix_unit = np.zeros((dimension, dimension), dtype=np.complex128)
            matrix_unit[row, column] = 1.0
            choi += np.kron(matrix_unit, matrix_unit.T)
    return float(np.min(np.linalg.eigvalsh(choi)).real)


def _operator_subspace_distance(
    left_basis: Sequence[ComplexArray], right_basis: Sequence[ComplexArray]
) -> float:
    if not left_basis or not right_basis:
        raise ValueError("Both operator subspaces must be nonempty.")
    left = np.column_stack([np.asarray(operator).reshape(-1) for operator in left_basis])
    right = np.column_stack([np.asarray(operator).reshape(-1) for operator in right_basis])
    q_left, _ = np.linalg.qr(left)
    q_right, _ = np.linalg.qr(right)
    denominator = np.sqrt(2.0 * max(q_left.shape[1], q_right.shape[1]))
    return float(
        np.linalg.norm(q_left @ q_left.conj().T - q_right @ q_right.conj().T, ord="fro")
        / denominator
    )


def audit_factor_normalizer(
    unitary: ComplexArray,
    left_basis: Sequence[ComplexArray],
    right_basis: Sequence[ComplexArray],
    *,
    allow_swap: bool = False,
    tolerance: float = 1e-9,
) -> FactorNormalizerAudit:
    """Test the residual basis freedom after a factor pair has been fixed."""
    u = np.asarray(unitary, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("Unitary must be square.")
    identity = np.eye(u.shape[0], dtype=np.complex128)
    unitary_defect = float(np.linalg.norm(u.conj().T @ u - identity, ord="fro"))
    if any(np.asarray(operator).shape != u.shape for operator in (*left_basis, *right_basis)):
        raise ValueError("Factor bases and unitary dimensions do not match.")
    transformed_left = tuple(u @ operator @ u.conj().T for operator in left_basis)
    transformed_right = tuple(u @ operator @ u.conj().T for operator in right_basis)
    ordered = max(
        _operator_subspace_distance(transformed_left, left_basis),
        _operator_subspace_distance(transformed_right, right_basis),
    )
    swapped = max(
        _operator_subspace_distance(transformed_left, right_basis),
        _operator_subspace_distance(transformed_right, left_basis),
    )
    defect = min(ordered, swapped) if allow_swap else ordered
    return FactorNormalizerAudit(
        unitary_defect=unitary_defect,
        ordered_defect=ordered,
        swapped_defect=swapped,
        swap_allowed=allow_swap,
        allowed=bool(unitary_defect <= tolerance and defect <= tolerance),
    )


def _factor_size(algebra_dimension: int) -> int | None:
    size = isqrt(algebra_dimension)
    return size if size * size == algebra_dimension else None


def _resolved_splits(
    sectors: tuple[tuple[int, ...], ...], max_bridge_count: int
):
    """Assign whole sectors; the first sector fixes left/right exchange gauge."""
    for tail in product((0, 1, 2), repeat=len(sectors) - 1):
        assignment = (0, *tail)
        left_sectors = [sectors[i] for i, label in enumerate(assignment) if label == 0]
        right_sectors = [sectors[i] for i, label in enumerate(assignment) if label == 1]
        bridge_sectors = [sectors[i] for i, label in enumerate(assignment) if label == 2]
        if not right_sectors:
            continue
        left = tuple(sorted(index for sector in left_sectors for index in sector))
        right = tuple(sorted(index for sector in right_sectors for index in sector))
        bridges = tuple(sorted(index for sector in bridge_sectors for index in sector))
        if len(bridges) <= max_bridge_count:
            yield left, right, bridges


def search_resolved_factorizations(
    events: Sequence[ComplexArray],
    event_sectors: Sequence[Sequence[int]],
    *,
    max_bridge_count: int = 0,
    tolerance: float = 1e-8,
    exact_tolerance: float = 1e-7,
    bridge_penalty: float = 0.05,
) -> FactorizationSearchResult:
    """Search cuts invariant under block-GL changes inside event sectors."""
    arrays, dimension = _validate_events(events)
    sectors = normalise_event_sectors(len(arrays), event_sectors)
    if len(sectors) < 2:
        raise ValueError("At least two event sectors are needed for a bipartition search.")
    if not 0 <= max_bridge_count <= len(arrays) - 2:
        raise ValueError("Invalid max_bridge_count.")
    if tolerance <= 0 or exact_tolerance <= 0 or bridge_penalty < 0:
        raise ValueError("Invalid search parameters.")

    full_dimension = dimension * dimension
    cache: dict[tuple[int, ...], tuple[ComplexArray, ...]] = {}

    def algebra(indices: tuple[int, ...]) -> tuple[ComplexArray, ...]:
        if indices not in cache:
            cache[indices] = generated_star_algebra(
                [arrays[index] for index in indices], tolerance=tolerance
            )
        return cache[indices]

    candidates: list[FactorizationCandidate] = []
    for left_events, right_events, bridge_events in _resolved_splits(
        sectors, max_bridge_count
    ):
        left, right = algebra(left_events), algebra(right_events)
        joint = algebra(tuple(sorted((*left_events, *right_events))))
        left_center, right_center = center_basis(left), center_basis(right)
        left_size, right_size = _factor_size(len(left)), _factor_size(len(right))
        commutation = algebra_commutator_defect(left, right)
        generation = 1.0 - len(joint) / full_dimension
        dimension_defect = abs(len(left) * len(right) - full_dimension) / full_dimension
        center_defect = float(
            max(0, len(left_center) - 1) + max(0, len(right_center) - 1)
        )
        bridge_fraction = len(bridge_events) / len(arrays)
        score = (
            commutation
            + generation
            + dimension_defect
            + center_defect
            + bridge_penalty * bridge_fraction
        )
        exact = bool(
            commutation <= exact_tolerance
            and generation <= exact_tolerance
            and dimension_defect <= exact_tolerance
            and center_defect == 0.0
            and left_size is not None
            and right_size is not None
            and left_size * right_size == dimension
        )
        candidates.append(
            FactorizationCandidate(
                left_events,
                right_events,
                bridge_events,
                len(left),
                len(right),
                len(joint),
                len(left_center),
                len(right_center),
                left_size,
                right_size,
                commutation,
                generation,
                dimension_defect,
                center_defect,
                bridge_fraction,
                float(score),
                exact,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            len(candidate.bridge_events),
            candidate.left_events,
            candidate.right_events,
        )
    )
    best = candidates[0] if candidates else None
    gap = candidates[1].score - best.score if len(candidates) > 1 else float("inf")
    return FactorizationSearchResult(
        dimension,
        best,
        tuple(candidates),
        float(gap if best is not None else 0.0),
        sum(candidate.exact for candidate in candidates),
    )
