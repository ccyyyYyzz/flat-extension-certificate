"""Microscopic collective-descent mechanisms for connected atomic response ports.

The central finite model is a classical--quantum Markov process on a direct sum
of identical matrix blocks. Reversible port conversion transports each internal
block by a unitary. Detailed balance turns the response-vector reduction into a
connection Laplacian whose kernel is the holonomy-fixed space of the network.
The collective copy count is therefore controlled by the holonomy, not by
irreducibility of the target under the ambient group: trivial (flat) holonomy
leaves one full collective copy of the target and gaps every relative-copy mode;
fixed-point-free holonomy leaves none; intermediate holonomy---for instance a
single ``SO(3)`` rotation, which fixes its axis---leaves a partial fixed
subspace that is neither zero nor a full copy. The audit therefore reports the
fixed-space dimension and only claims the clean dichotomy when it is zero or the
full target dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConnectionLaplacianAudit:
    laplacian: FloatArray
    eigenvalues: FloatArray
    zero_mode_count: int
    target_dimension: int
    algebraic_gap: float
    collective_activated: bool
    one_copy_activated: bool
    partial_fixed_subspace: bool = False


@dataclass(frozen=True)
class CollectiveSchurReport:
    effective_block: ComplexArray
    weighted_average: ComplexArray
    correction: ComplexArray
    correction_norm: float
    correction_bound: float
    relative_gap: float
    block_norm: float
    collective_dimension: int


@dataclass(frozen=True)
class CQMarkovAudit:
    superoperator: ComplexArray
    trace_preservation_defect: float
    stationary_defect: float
    detailed_balance_defect: float
    spectral_gap: float


@dataclass(frozen=True)
class AttemptRateCertificate:
    rate_operator: ComplexArray
    minimum_rate: float
    uniform_positive_hazard: bool


@dataclass(frozen=True)
class CollectiveCADFixture:
    weights: FloatArray
    transports: tuple[tuple[FloatArray, ...], ...]
    audit: ConnectionLaplacianAudit


def _real_square(matrix: FloatArray, name: str) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite square real matrix.")
    return value


def _complex_square(matrix: ComplexArray, name: str) -> ComplexArray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite square complex matrix.")
    return value


def _validate_transports(
    transports: Sequence[Sequence[FloatArray]], port_count: int, tolerance: float
) -> tuple[tuple[FloatArray, ...], ...]:
    if len(transports) != port_count or any(len(row) != port_count for row in transports):
        raise ValueError("Transport array must have one square block table over ports.")
    target_dimension: int | None = None
    output: list[tuple[FloatArray, ...]] = []
    for row in transports:
        converted = []
        for raw in row:
            value = _real_square(raw, "transport")
            if target_dimension is None:
                target_dimension = value.shape[0]
            elif value.shape != (target_dimension, target_dimension):
                raise ValueError("All transports must have one common target dimension.")
            identity = np.eye(value.shape[0])
            if np.linalg.norm(value.T @ value - identity, ord="fro") > tolerance:
                raise ValueError("Connection transports must be orthogonal.")
            converted.append(value)
        output.append(tuple(converted))
    assert target_dimension is not None
    for i in range(port_count):
        if np.linalg.norm(output[i][i] - np.eye(target_dimension), ord="fro") > tolerance:
            raise ValueError("Diagonal transports must be identities.")
        for j in range(port_count):
            if np.linalg.norm(output[j][i] - output[i][j].T, ord="fro") > tolerance:
                raise ValueError("Reverse transports must be inverses/transposes.")
    return tuple(output)


def connection_laplacian(
    weights: FloatArray,
    transports: Sequence[Sequence[FloatArray]],
    *,
    tolerance: float = 1e-10,
) -> FloatArray:
    """Return the block connection Laplacian defined by edge mismatch energy."""
    w = _real_square(weights, "weights").copy()
    if tolerance <= 0 or np.any(w < -tolerance) or not np.allclose(w, w.T, atol=tolerance):
        raise ValueError("Weights must be symmetric and nonnegative.")
    np.fill_diagonal(w, 0.0)
    transport = _validate_transports(transports, w.shape[0], tolerance)
    target_dimension = transport[0][0].shape[0]
    result = np.zeros((w.shape[0] * target_dimension,) * 2, dtype=np.float64)
    identity = np.eye(target_dimension)
    for i in range(w.shape[0]):
        for j in range(i + 1, w.shape[0]):
            weight = float(w[i, j])
            if weight <= tolerance:
                continue
            block_i = slice(i * target_dimension, (i + 1) * target_dimension)
            block_j = slice(j * target_dimension, (j + 1) * target_dimension)
            u_ij = transport[i][j]
            result[block_i, block_i] += weight * identity
            result[block_j, block_j] += weight * identity
            result[block_j, block_i] -= weight * u_ij
            result[block_i, block_j] -= weight * u_ij.T
    return np.asarray((result + result.T) / 2.0, dtype=np.float64)


def audit_connection_laplacian(
    weights: FloatArray,
    transports: Sequence[Sequence[FloatArray]],
    *,
    tolerance: float = 1e-9,
) -> ConnectionLaplacianAudit:
    laplacian = connection_laplacian(weights, transports, tolerance=tolerance * 0.1)
    values = np.linalg.eigvalsh(laplacian)
    scale = max(float(np.max(np.abs(values))), 1.0)
    zero_count = int(np.sum(np.abs(values) <= tolerance * scale))
    positive_values = values[values > tolerance * scale]
    gap = float(np.min(positive_values)) if positive_values.size else 0.0
    target_dimension = np.asarray(transports[0][0]).shape[0]
    return ConnectionLaplacianAudit(
        laplacian=laplacian,
        eigenvalues=np.asarray(values, dtype=np.float64),
        zero_mode_count=zero_count,
        target_dimension=target_dimension,
        algebraic_gap=gap,
        collective_activated=bool(zero_count > 0),
        one_copy_activated=bool(zero_count == target_dimension),
        partial_fixed_subspace=bool(0 < zero_count < target_dimension),
    )


def identity_transport_table(port_count: int, target_dimension: int) -> tuple[tuple[FloatArray, ...], ...]:
    if port_count < 1 or target_dimension < 1:
        raise ValueError("Port and target dimensions must be positive.")
    identity = np.eye(target_dimension, dtype=np.float64)
    return tuple(tuple(identity.copy() for _ in range(port_count)) for _ in range(port_count))


def rotation_2d(angle: float) -> FloatArray:
    value = float(angle)
    return np.asarray([[np.cos(value), -np.sin(value)], [np.sin(value), np.cos(value)]], dtype=np.float64)


def rotation_z_so3(angle: float) -> FloatArray:
    """Return the ``SO(3)`` vector-representation rotation about the z-axis.

    The rotation fixes exactly ``span{e_3}``, so a cycle holonomy equal to this
    matrix leaves a one-dimensional fixed subspace of the three-dimensional target.
    """
    value = float(angle)
    cosine, sine = np.cos(value), np.sin(value)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def flat_collective_fixture(port_count: int = 4, target_dimension: int = 3) -> CollectiveCADFixture:
    if port_count < 2:
        raise ValueError("Use at least two ports.")
    weights = np.zeros((port_count, port_count), dtype=np.float64)
    for index in range(port_count - 1):
        weights[index, index + 1] = weights[index + 1, index] = 1.0
    transports = identity_transport_table(port_count, target_dimension)
    return CollectiveCADFixture(weights, transports, audit_connection_laplacian(weights, transports))


def holonomy_frustrated_cycle(angle: float = 0.4) -> CollectiveCADFixture:
    """Three two-dimensional ports with nontrivial cycle holonomy and no zero copy."""
    port_count = 3
    weights = np.ones((port_count, port_count), dtype=np.float64) - np.eye(port_count)
    identity = np.eye(2)
    rows = [[identity.copy() for _ in range(port_count)] for _ in range(port_count)]
    rows[2][0] = rotation_2d(angle)
    rows[0][2] = rows[2][0].T
    transports = tuple(tuple(block for block in row) for row in rows)
    return CollectiveCADFixture(weights, transports, audit_connection_laplacian(weights, transports))


def holonomy_partial_fixed_cycle(angle: float = 0.4) -> CollectiveCADFixture:
    """Three ``SO(3)`` vector-rep ports whose cycle holonomy is ``R_z(angle)``.

    Two edges carry the identity and one edge carries a real orthogonal rotation
    about the z-axis, so the cycle holonomy fixes exactly ``span{e_3}``. The
    resulting fixed space is one-dimensional---neither zero nor a full copy of the
    three-dimensional target---so the audit reports ``partial_fixed_subspace`` and
    declines the clean one-copy dichotomy.
    """
    port_count = 3
    weights = np.ones((port_count, port_count), dtype=np.float64) - np.eye(port_count)
    identity = np.eye(3)
    rows = [[identity.copy() for _ in range(port_count)] for _ in range(port_count)]
    rows[2][0] = rotation_z_so3(angle)
    rows[0][2] = rows[2][0].T
    transports = tuple(tuple(block for block in row) for row in rows)
    return CollectiveCADFixture(weights, transports, audit_connection_laplacian(weights, transports))


def collective_schur_effective(
    laplacian: FloatArray,
    local_blocks: Sequence[ComplexArray],
    *,
    coupling_strength: float = 1.0,
    tolerance: float = 1e-10,
) -> CollectiveSchurReport:
    """Eliminate gapped relative-copy modes by an exact Schur complement."""
    lap = _real_square(laplacian, "laplacian")
    blocks = tuple(_complex_square(block, "local block") for block in local_blocks)
    if not blocks or coupling_strength <= 0 or tolerance <= 0:
        raise ValueError("Use local blocks, positive coupling, and positive tolerance.")
    block_dimension = blocks[0].shape[0]
    if any(block.shape != (block_dimension, block_dimension) for block in blocks):
        raise ValueError("All local blocks must have one common size.")
    if lap.shape[0] != len(blocks) * block_dimension:
        raise ValueError("Laplacian dimension must equal port count times block dimension.")
    d_matrix = np.zeros_like(lap, dtype=np.complex128)
    for index, block in enumerate(blocks):
        segment = slice(index * block_dimension, (index + 1) * block_dimension)
        d_matrix[segment, segment] = block
    values, vectors = np.linalg.eigh(lap)
    scale = max(float(np.max(np.abs(values))), 1.0)
    zero_mask = np.abs(values) <= tolerance * scale
    if not np.any(zero_mask):
        raise ValueError("No collective zero sector is activated.")
    positive = values[~zero_mask]
    if not positive.size:
        raise ValueError("No relative-copy sector exists to eliminate.")
    relative_gap = float(coupling_strength * np.min(positive))
    p_vectors = vectors[:, zero_mask].astype(np.complex128)
    q_vectors = vectors[:, ~zero_mask].astype(np.complex128)
    p_block = p_vectors.conj().T @ d_matrix @ p_vectors
    q_block = coupling_strength * np.diag(positive.astype(np.complex128)) + q_vectors.conj().T @ d_matrix @ q_vectors
    cross = p_vectors.conj().T @ d_matrix @ q_vectors
    block_norm = float(np.linalg.norm(d_matrix, ord=2))
    if block_norm >= relative_gap:
        raise ValueError("Local response is not smaller than the relative conversion gap.")
    correction = cross @ np.linalg.inv(q_block) @ cross.conj().T
    effective = p_block - correction
    bound = block_norm * block_norm / (relative_gap - block_norm)
    return CollectiveSchurReport(
        effective_block=np.asarray(effective),
        weighted_average=np.asarray(p_block),
        correction=np.asarray(correction),
        correction_norm=float(np.linalg.norm(correction, ord=2)),
        correction_bound=float(bound),
        relative_gap=relative_gap,
        block_norm=block_norm,
        collective_dimension=p_block.shape[0],
    )


def _vec_conjugation(unitary: ComplexArray) -> ComplexArray:
    return np.kron(unitary.conj(), unitary)


def cq_markov_superoperator(
    rates: FloatArray,
    unitary_transports: Sequence[Sequence[ComplexArray]],
    *,
    tolerance: float = 1e-10,
) -> ComplexArray:
    """Generator of a classical--quantum port-conversion Markov semigroup."""
    q = _real_square(rates, "rates")
    if np.any(q < -tolerance):
        raise ValueError("Rates must be nonnegative.")
    np.fill_diagonal(q, 0.0)
    port_count = q.shape[0]
    if len(unitary_transports) != port_count or any(len(row) != port_count for row in unitary_transports):
        raise ValueError("Unitary transports must form a square port table.")
    internal_dimension = np.asarray(unitary_transports[0][0]).shape[0]
    identity = np.eye(internal_dimension, dtype=np.complex128)
    converted: list[list[ComplexArray]] = []
    for row in unitary_transports:
        target_row = []
        for raw in row:
            unitary = _complex_square(raw, "unitary transport")
            if unitary.shape != (internal_dimension, internal_dimension):
                raise ValueError("Unitary dimensions do not match.")
            if np.linalg.norm(unitary.conj().T @ unitary - identity, ord="fro") > tolerance:
                raise ValueError("Each transport must be unitary.")
            target_row.append(unitary)
        converted.append(target_row)
    block_size = internal_dimension * internal_dimension
    generator = np.zeros((port_count * block_size, port_count * block_size), dtype=np.complex128)
    super_identity = np.eye(block_size, dtype=np.complex128)
    for i in range(port_count):
        block_i = slice(i * block_size, (i + 1) * block_size)
        outgoing = 0.0
        for j in range(port_count):
            rate = float(q[i, j])
            if i == j or rate <= tolerance:
                continue
            block_j = slice(j * block_size, (j + 1) * block_size)
            generator[block_j, block_i] += rate * _vec_conjugation(converted[i][j])
            outgoing += rate
        generator[block_i, block_i] -= outgoing * super_identity
    return generator


def audit_cq_detailed_balance(
    rates: FloatArray,
    stationary_distribution: Sequence[float],
    unitary_transports: Sequence[Sequence[ComplexArray]],
    *,
    tolerance: float = 1e-10,
) -> CQMarkovAudit:
    q = _real_square(rates, "rates").copy()
    np.fill_diagonal(q, 0.0)
    pi = np.asarray(stationary_distribution, dtype=np.float64)
    if pi.shape != (q.shape[0],) or np.any(pi <= 0) or not np.isclose(np.sum(pi), 1.0):
        raise ValueError("Stationary distribution must be strictly positive and normalized.")
    generator = cq_markov_superoperator(q, unitary_transports, tolerance=tolerance)
    internal_dimension = np.asarray(unitary_transports[0][0]).shape[0]
    identity_density = np.eye(internal_dimension, dtype=np.complex128) / internal_dimension
    stationary = np.concatenate([(pi[index] * identity_density).reshape(-1, order="F") for index in range(q.shape[0])])
    stationary_defect = float(np.linalg.norm(generator @ stationary))
    trace_row = np.concatenate([np.eye(internal_dimension).reshape(-1, order="F") for _ in range(q.shape[0])]).conj()
    trace_defect = float(np.linalg.norm(trace_row @ generator))
    detailed = np.asarray([pi[i] * q[i, j] - pi[j] * q[j, i] for i in range(q.shape[0]) for j in range(q.shape[0])])
    detailed_defect = float(np.max(np.abs(detailed)))
    eigenvalues = np.linalg.eigvals(generator)
    real_parts = np.sort(np.real(eigenvalues))[::-1]
    nonzero_decay = -real_parts[real_parts < -tolerance]
    gap = float(np.min(nonzero_decay)) if nonzero_decay.size else 0.0
    return CQMarkovAudit(generator, trace_defect, stationary_defect, detailed_defect, gap)


def attempt_rate_certificate(
    effects: Sequence[ComplexArray], attempt_strengths: Sequence[float], *, tolerance: float = 1e-10
) -> AttemptRateCertificate:
    """Certify the uniform CP-instrument hazard ``sum kappa_i J_i^*(I) >= q I``."""
    if not effects or len(effects) != len(attempt_strengths):
        raise ValueError("Use one attempt strength per nonempty effect list.")
    arrays = tuple(_complex_square(effect, "attempt effect") for effect in effects)
    if any(effect.shape != arrays[0].shape for effect in arrays):
        raise ValueError("Attempt effects must have one common dimension.")
    rate = np.zeros_like(arrays[0])
    for effect, strength in zip(arrays, attempt_strengths):
        kappa = float(strength)
        if kappa < 0 or np.min(np.linalg.eigvalsh((effect + effect.conj().T) / 2.0)) < -tolerance:
            raise ValueError("Use nonnegative strengths and positive-semidefinite effects.")
        rate += kappa * effect
    rate = (rate + rate.conj().T) / 2.0
    minimum = float(np.min(np.linalg.eigvalsh(rate)).real)
    return AttemptRateCertificate(rate, minimum, bool(minimum > tolerance))
