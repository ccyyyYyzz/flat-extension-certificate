"""Intrinsic subsystem search on one unlabelled matrix algebra.

The search receives process-event operators in ``M_d`` but no tensor-product
coordinates.  A candidate cut is a pair of commuting factor *-algebras
obtained from disjoint event subsets; unused events are interactions (bridges).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isqrt
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FactorizationCandidate:
    left_events: tuple[int, ...]
    right_events: tuple[int, ...]
    bridge_events: tuple[int, ...]
    left_algebra_dimension: int
    right_algebra_dimension: int
    joint_algebra_dimension: int
    left_center_dimension: int
    right_center_dimension: int
    left_factor_size: int | None
    right_factor_size: int | None
    commutator_defect: float
    generation_defect: float
    dimension_defect: float
    center_defect: float
    bridge_fraction: float
    score: float
    exact: bool


@dataclass(frozen=True)
class FactorizationSearchResult:
    ambient_dimension: int
    best: FactorizationCandidate | None
    candidates: tuple[FactorizationCandidate, ...]
    score_gap: float
    exact_candidate_count: int

    @property
    def identified(self) -> bool:
        return self.best is not None and self.best.exact and self.score_gap > 0.0


@dataclass(frozen=True)
class NoGoCertificate:
    ambient_dimension: int
    event_algebra_dimension: int
    commutant_dimension: int
    full_event_algebra: bool
    nontrivial_exact_noiseless_factor_forbidden: bool


def _validate(operators: Sequence[ComplexArray]) -> tuple[list[ComplexArray], int]:
    if not operators:
        raise ValueError("At least one operator is required.")
    arrays = [np.asarray(operator, dtype=np.complex128) for operator in operators]
    dimension = arrays[0].shape[0]
    if dimension < 1 or any(array.shape != (dimension, dimension) for array in arrays):
        raise ValueError("Operators must be equally sized square matrices.")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Operators must be finite.")
    return arrays, dimension


def hs_inner(left: ComplexArray, right: ComplexArray) -> complex:
    return complex(np.vdot(np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)))


def hs_norm(operator: ComplexArray) -> float:
    return float(np.linalg.norm(np.asarray(operator, dtype=np.complex128), ord="fro"))


def orthonormal_operator_basis(
    operators: Iterable[ComplexArray], *, tolerance: float = 1e-10
) -> tuple[ComplexArray, ...]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    basis: list[ComplexArray] = []
    shape: tuple[int, int] | None = None
    for raw in operators:
        residual = np.asarray(raw, dtype=np.complex128).copy()
        if residual.ndim != 2 or residual.shape[0] != residual.shape[1]:
            raise ValueError("Operators must be square.")
        if shape is None:
            shape = residual.shape
        elif residual.shape != shape:
            raise ValueError("Operator shapes do not match.")
        for _ in range(2):
            for element in basis:
                residual -= hs_inner(element, residual) * element
        norm = hs_norm(residual)
        if norm > tolerance:
            basis.append(residual / norm)
    return tuple(basis)


def generated_star_algebra(
    generators: Sequence[ComplexArray], *, tolerance: float = 1e-9
) -> tuple[ComplexArray, ...]:
    arrays, dimension = _validate(generators)
    seeds = arrays + [array.conj().T for array in arrays]
    basis = orthonormal_operator_basis(
        [np.eye(dimension, dtype=np.complex128), *seeds], tolerance=tolerance
    )
    limit = dimension * dimension
    for _ in range(limit):
        candidates = [*basis]
        for element in basis:
            for seed in seeds:
                candidates.extend((element @ seed, seed @ element))
        enlarged = orthonormal_operator_basis(candidates, tolerance=tolerance)
        if len(enlarged) == len(basis) or len(enlarged) >= limit:
            return enlarged[:limit]
        basis = enlarged
    return basis[:limit]


def project_onto_algebra(operator: ComplexArray, basis: Sequence[ComplexArray]) -> ComplexArray:
    if not basis:
        raise ValueError("Algebra basis cannot be empty.")
    op = np.asarray(operator, dtype=np.complex128)
    return np.asarray(
        sum((hs_inner(element, op) * element for element in basis), np.zeros_like(op)),
        dtype=np.complex128,
    )


def algebra_residual(operator: ComplexArray, basis: Sequence[ComplexArray]) -> float:
    op = np.asarray(operator, dtype=np.complex128)
    return hs_norm(op - project_onto_algebra(op, basis)) / max(
        hs_norm(op), np.finfo(float).eps
    )


def _nullspace(matrix: ComplexArray, tolerance: float) -> ComplexArray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = float(singular_values[0]) if singular_values.size else 1.0
    rank = int(np.sum(singular_values > tolerance * max(1.0, scale)))
    return vh.conj().T[:, rank:]


def commutant_basis(
    generators: Sequence[ComplexArray], *, tolerance: float = 1e-9
) -> tuple[ComplexArray, ...]:
    arrays, dimension = _validate(generators)
    identity = np.eye(dimension, dtype=np.complex128)
    equations = np.concatenate(
        [np.kron(array.T, identity) - np.kron(identity, array) for array in arrays],
        axis=0,
    )
    null = _nullspace(equations, tolerance)
    return orthonormal_operator_basis(
        [
            null[:, index].reshape((dimension, dimension), order="F")
            for index in range(null.shape[1])
        ],
        tolerance=tolerance,
    )


def center_basis(
    algebra_basis: Sequence[ComplexArray], *, tolerance: float = 1e-9
) -> tuple[ComplexArray, ...]:
    arrays, dimension = _validate(algebra_basis)
    equations = np.column_stack(
        [
            np.concatenate(
                [
                    (candidate @ element - element @ candidate).reshape(-1, order="F")
                    for element in arrays
                ]
            )
            for candidate in arrays
        ]
    )
    null = _nullspace(equations, tolerance)
    centres = [
        sum(
            (null[j, index] * arrays[j] for j in range(len(arrays))),
            np.zeros((dimension, dimension), dtype=np.complex128),
        )
        for index in range(null.shape[1])
    ]
    return orthonormal_operator_basis(centres, tolerance=tolerance)


def algebra_commutator_defect(
    left_basis: Sequence[ComplexArray], right_basis: Sequence[ComplexArray]
) -> float:
    values = [
        hs_norm(left @ right - right @ left)
        / (2.0 * max(hs_norm(left) * hs_norm(right), np.finfo(float).eps))
        for left in left_basis
        for right in right_basis
    ]
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _factor_size(algebra_dimension: int) -> int | None:
    size = isqrt(algebra_dimension)
    return size if size * size == algebra_dimension else None


def _splits(event_count: int, max_bridges: int):
    for tail in product((0, 1, 2), repeat=event_count - 1):
        assignment = (0, *tail)
        left = tuple(i for i, label in enumerate(assignment) if label == 0)
        right = tuple(i for i, label in enumerate(assignment) if label == 1)
        bridges = tuple(i for i, label in enumerate(assignment) if label == 2)
        if right and len(bridges) <= max_bridges:
            yield left, right, bridges


def search_intrinsic_factorizations(
    events: Sequence[ComplexArray],
    *,
    max_bridge_count: int = 0,
    tolerance: float = 1e-8,
    exact_tolerance: float = 1e-7,
    bridge_penalty: float = 0.05,
) -> FactorizationSearchResult:
    """Rank intrinsic two-factor cuts; no tensor coordinates are an input."""
    arrays, dimension = _validate(events)
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
    for left_events, right_events, bridge_events in _splits(
        len(arrays), max_bridge_count
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


def process_algebra_no_go(
    events: Sequence[ComplexArray], *, tolerance: float = 1e-9
) -> NoGoCertificate:
    """Certify: full event algebra + scalar commutant forbids an invariant factor."""
    arrays, dimension = _validate(events)
    algebra = generated_star_algebra(arrays, tolerance=tolerance)
    commutant = commutant_basis(arrays, tolerance=tolerance)
    full = len(algebra) == dimension * dimension
    return NoGoCertificate(
        dimension,
        len(algebra),
        len(commutant),
        full,
        full and len(commutant) == 1,
    )


def subspace_distance(
    left_basis: Sequence[ComplexArray], right_basis: Sequence[ComplexArray]
) -> float:
    if not left_basis or not right_basis:
        raise ValueError("Both bases must be non-empty.")
    left = np.column_stack(
        [np.asarray(operator).reshape(-1) for operator in left_basis]
    )
    right = np.column_stack(
        [np.asarray(operator).reshape(-1) for operator in right_basis]
    )
    q_left, _ = np.linalg.qr(left)
    q_right, _ = np.linalg.qr(right)
    denominator = np.sqrt(2.0 * max(q_left.shape[1], q_right.shape[1]))
    return float(
        np.linalg.norm(
            q_left @ q_left.conj().T - q_right @ q_right.conj().T, ord="fro"
        )
        / denominator
    )


def random_unitary(dimension: int, *, seed: int | None = None) -> ComplexArray:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0, diagonal / np.abs(diagonal), 1.0)
    return np.asarray(q @ np.diag(phases.conj()), dtype=np.complex128)


def matrix_factor_generators(dimension: int) -> tuple[ComplexArray, ComplexArray]:
    if dimension < 2:
        raise ValueError("Factor dimension must be at least two.")
    shift = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension):
        shift[(index + 1) % dimension, index] = 1.0
    clock = np.diag(np.exp(2j * np.pi * np.arange(dimension) / dimension))
    return shift, np.asarray(clock, dtype=np.complex128)


def hidden_tensor_events(
    left_dimension: int,
    right_dimension: int,
    *,
    seed: int | None = None,
    include_bridge: bool = False,
) -> tuple[
    tuple[ComplexArray, ...],
    tuple[ComplexArray, ...],
    tuple[ComplexArray, ...],
]:
    """Benchmark only: hide a tensor factorization by global unitary conjugacy."""
    if left_dimension < 2 or right_dimension < 2:
        raise ValueError("Both hidden factors must be nontrivial.")
    ls, lc = matrix_factor_generators(left_dimension)
    rs, rc = matrix_factor_generators(right_dimension)
    il = np.eye(left_dimension, dtype=np.complex128)
    ir = np.eye(right_dimension, dtype=np.complex128)
    unitary = random_unitary(left_dimension * right_dimension, seed=seed)

    def hide(operator: ComplexArray) -> ComplexArray:
        return unitary @ operator @ unitary.conj().T

    left = tuple(
        hide(operator) for operator in (np.kron(ls, ir), np.kron(lc, ir))
    )
    right = tuple(
        hide(operator) for operator in (np.kron(il, rs), np.kron(il, rc))
    )
    events: tuple[ComplexArray, ...] = (*left, *right)
    if include_bridge:
        events = (*events, hide(np.kron(ls, rs)))
    return events, left, right


def perturb_events(
    events: Sequence[ComplexArray], amplitude: float, *, seed: int | None = None
) -> tuple[ComplexArray, ...]:
    arrays, dimension = _validate(events)
    if amplitude < 0:
        raise ValueError("amplitude must be nonnegative.")
    rng = np.random.default_rng(seed)
    perturbed = []
    for event in arrays:
        noise = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
            size=(dimension, dimension)
        )
        noise /= max(hs_norm(noise), np.finfo(float).eps)
        perturbed.append(event + amplitude * hs_norm(event) * noise)
    return tuple(perturbed)
