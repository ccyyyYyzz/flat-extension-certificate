"""Finite-dimensional diagnostics for CF-RSP.

The fixed tensor-product structure used here is a gauge-fixed numerical probe,
not the ontology proposed in ``docs/CF_RSP_ARCHITECTURE.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PartitionResult:
    left: tuple[int, ...]
    right: tuple[int, ...]
    score: float
    within_mean: float
    across_mean: float


@dataclass(frozen=True)
class MeasurementDilationResult:
    global_state: ComplexArray
    system_state: ComplexArray
    apparatus_state: ComplexArray
    global_purity: float
    system_purity: float
    system_coherence_l1: float
    mutual_information_bits: float


def as_density_matrix(state: ComplexArray) -> ComplexArray:
    """Validate a state vector/density matrix and return a density matrix."""
    a = np.asarray(state, dtype=np.complex128)
    if a.ndim == 1:
        if not np.isclose(np.vdot(a, a).real, 1.0, atol=1e-10):
            raise ValueError("State vector must be normalised.")
        return np.outer(a, a.conj())
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Density matrix must be square.")
    if not np.allclose(a, a.conj().T, atol=1e-10):
        raise ValueError("Density matrix must be Hermitian.")
    if not np.isclose(np.trace(a), 1.0, atol=1e-10):
        raise ValueError("Density matrix must have unit trace.")
    if np.min(np.linalg.eigvalsh(a)) < -1e-10:
        raise ValueError("Density matrix must be positive semidefinite.")
    return a


def partial_trace(state: ComplexArray, keep: Sequence[int], dims: Sequence[int]) -> ComplexArray:
    """Trace out all factors except the unique, increasingly ordered ``keep`` set."""
    rho = as_density_matrix(state)
    dimensions = [int(d) for d in dims]
    if any(d <= 0 for d in dimensions) or prod(dimensions) != rho.shape[0]:
        raise ValueError("Invalid subsystem dimensions.")
    kept = sorted(set(int(i) for i in keep))
    if len(kept) != len(keep) or any(i < 0 or i >= len(dimensions) for i in kept):
        raise ValueError("Invalid kept subsystem indices.")
    tensor = rho.reshape(tuple(dimensions + dimensions))
    current = dimensions.copy()
    for index in sorted(set(range(len(dimensions))) - set(kept), reverse=True):
        tensor = np.trace(tensor, axis1=index, axis2=index + len(current))
        current.pop(index)
    out_dim = prod(dimensions[i] for i in kept) if kept else 1
    return np.asarray(tensor, dtype=np.complex128).reshape(out_dim, out_dim)


def von_neumann_entropy(state: ComplexArray, *, base: float = 2.0) -> float:
    values = np.linalg.eigvalsh(as_density_matrix(state)).real
    values = values[values > 1e-14]
    return float(-np.sum(values * np.log(values) / np.log(base))) if values.size else 0.0


def purity(state: ComplexArray) -> float:
    rho = as_density_matrix(state)
    return float(np.trace(rho @ rho).real)


def l1_coherence(state: ComplexArray) -> float:
    rho = as_density_matrix(state)
    return float(np.sum(np.abs(rho - np.diag(np.diag(rho)))))


def mutual_information(
    state: ComplexArray,
    left: Sequence[int],
    right: Sequence[int],
    dims: Sequence[int],
    *,
    base: float = 2.0,
) -> float:
    lset, rset = set(left), set(right)
    if lset & rset:
        raise ValueError("Subsystem sets must be disjoint.")
    s_left = von_neumann_entropy(partial_trace(state, sorted(lset), dims), base=base)
    s_right = von_neumann_entropy(partial_trace(state, sorted(rset), dims), base=base)
    s_joint = von_neumann_entropy(partial_trace(state, sorted(lset | rset), dims), base=base)
    return max(0.0, float(s_left + s_right - s_joint))


def pairwise_mutual_information(
    state: ComplexArray, dims: Sequence[int], *, base: float = 2.0
) -> FloatArray:
    matrix = np.zeros((len(dims), len(dims)), dtype=np.float64)
    for i, j in combinations(range(len(dims)), 2):
        matrix[i, j] = matrix[j, i] = mutual_information(
            state, [i], [j], dims, base=base
        )
    return matrix


def normalised_relation_weights(
    mutual_information_matrix: FloatArray,
    dims: Sequence[int],
    *,
    base: float = 2.0,
) -> FloatArray:
    mi = np.asarray(mutual_information_matrix, dtype=np.float64)
    if mi.shape != (len(dims), len(dims)) or np.any(mi < -1e-10):
        raise ValueError("Invalid mutual-information matrix.")
    weights = np.zeros_like(mi)
    for i, j in combinations(range(len(dims)), 2):
        cap = 2.0 * min(np.log(dims[i]) / np.log(base), np.log(dims[j]) / np.log(base))
        weights[i, j] = weights[j, i] = float(np.clip(mi[i, j] / cap, 0.0, 1.0))
    return weights


def relation_lengths(
    weights: FloatArray, *, epsilon: float = 1e-9, length_scale: float = 1.0
) -> FloatArray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or epsilon <= 0 or length_scale <= 0:
        raise ValueError("Invalid relation-length input.")
    if np.any(w < -1e-12) or np.any(w > 1 + 1e-12):
        raise ValueError("Weights must be in [0, 1].")
    lengths = -length_scale * np.log((np.clip(w, 0, 1) + epsilon) / (1 + epsilon))
    np.fill_diagonal(lengths, 0.0)
    return lengths


def shortest_path_metric(edge_lengths: FloatArray) -> FloatArray:
    """Floyd-Warshall closure; the result is an extended pseudometric."""
    distance = np.asarray(edge_lengths, dtype=np.float64).copy()
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Edge lengths must be square.")
    if np.any(distance < -1e-12) or not np.allclose(distance, distance.T, atol=1e-10):
        raise ValueError("This finite implementation expects symmetric nonnegative lengths.")
    np.fill_diagonal(distance, 0.0)
    for k in range(distance.shape[0]):
        distance = np.minimum(distance, distance[:, [k]] + distance[[k], :])
    return distance


def weighted_laplacian(weights: FloatArray) -> FloatArray:
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.ndim != 2 or w.shape[0] != w.shape[1] or np.any(w < -1e-12):
        raise ValueError("Invalid weight matrix.")
    if not np.allclose(w, w.T, atol=1e-10):
        raise ValueError("Weights must be symmetric.")
    np.fill_diagonal(w, 0.0)
    return np.diag(np.sum(w, axis=1)) - w


def heat_trace(laplacian: FloatArray, diffusion_times: Iterable[float]) -> FloatArray:
    lap = np.asarray(laplacian, dtype=np.float64)
    times = np.asarray(list(diffusion_times), dtype=np.float64)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1] or np.any(times <= 0):
        raise ValueError("Invalid heat-trace input.")
    values = np.linalg.eigvalsh(lap)
    return np.asarray([np.mean(np.exp(-t * values)) for t in times], dtype=np.float64)


def spectral_dimension(
    laplacian: FloatArray, diffusion_times: Iterable[float]
) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(list(diffusion_times), dtype=np.float64)
    if times.size < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("Use at least three strictly increasing diffusion times.")
    trace = heat_trace(laplacian, times)
    dimension = -2.0 * np.gradient(np.log(trace), np.log(times), edge_order=2)
    return trace, np.asarray(dimension, dtype=np.float64)


def partition_score(weights: FloatArray, left: Sequence[int]) -> tuple[float, float, float]:
    w = np.asarray(weights, dtype=np.float64)
    nodes, lset = set(range(w.shape[0])), set(int(i) for i in left)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not lset or lset == nodes or not lset <= nodes:
        raise ValueError("Invalid bipartition.")
    rset = nodes - lset
    within = [float(w[i, j]) for side in (sorted(lset), sorted(rset)) for i, j in combinations(side, 2)]
    across = [float(w[i, j]) for i in lset for j in rset]
    within_mean = float(np.mean(within)) if within else 0.0
    across_mean = float(np.mean(across)) if across else 0.0
    return within_mean - across_mean, within_mean, across_mean


def best_bipartition(weights: FloatArray, *, balanced: bool = True) -> PartitionResult:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("Weights must describe at least two nodes.")
    n = w.shape[0]
    sizes = {n // 2, n // 2 + (n % 2)} if balanced else set(range(1, n))
    best: PartitionResult | None = None
    for size in sorted(sizes):
        if not 1 <= size < n:
            continue
        for rest in combinations(range(1, n), size - 1):
            left = tuple(sorted((0, *rest)))
            right = tuple(sorted(set(range(n)) - set(left)))
            score, within, across = partition_score(w, left)
            candidate = PartitionResult(left, right, score, within, across)
            if best is None or score > best.score + 1e-12:
                best = candidate
    if best is None:
        raise RuntimeError("No bipartition generated.")
    return best


def kron_all(factors: Sequence[ComplexArray]) -> ComplexArray:
    if not factors:
        raise ValueError("At least one factor is required.")
    result = np.asarray(factors[0], dtype=np.complex128)
    for factor in factors[1:]:
        result = np.kron(result, np.asarray(factor, dtype=np.complex128))
    return result


def basis_state(bit_string: str) -> ComplexArray:
    if not bit_string or any(bit not in "01" for bit in bit_string):
        raise ValueError("bit_string must contain 0/1 characters.")
    vector = np.zeros(2 ** len(bit_string), dtype=np.complex128)
    vector[int(bit_string, 2)] = 1.0
    return vector


def bell_pair_state() -> ComplexArray:
    return np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def consecutive_bell_pairs(pair_count: int) -> ComplexArray:
    if pair_count < 1:
        raise ValueError("pair_count must be positive.")
    return kron_all([bell_pair_state() for _ in range(pair_count)])


def measurement_dilation_z(system_state: ComplexArray) -> MeasurementDilationResult:
    """Ideal Z measurement: apparatus |0>, CNOT coupling, no branch selection."""
    rho = as_density_matrix(system_state)
    if rho.shape != (2, 2):
        raise ValueError("Expected one qubit.")
    apparatus = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    cnot = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    global_state = cnot @ np.kron(rho, apparatus) @ cnot.conj().T
    system = partial_trace(global_state, [0], [2, 2])
    pointer = partial_trace(global_state, [1], [2, 2])
    return MeasurementDilationResult(
        global_state,
        system,
        pointer,
        purity(global_state),
        purity(system),
        l1_coherence(system),
        mutual_information(global_state, [0], [1], [2, 2]),
    )


def operator_on_qubits(operator: ComplexArray, target: int, qubit_count: int) -> ComplexArray:
    op = np.asarray(operator, dtype=np.complex128)
    if op.shape != (2, 2) or not 0 <= target < qubit_count:
        raise ValueError("Invalid embedded operator.")
    identity = np.eye(2, dtype=np.complex128)
    return kron_all([op if i == target else identity for i in range(qubit_count)])


def relation_hamiltonian(
    couplings: FloatArray, *, local_fields: Sequence[float] | None = None
) -> ComplexArray:
    """H = sum J_ij X_i X_j + sum h_i Z_i on an abstract relation matrix."""
    coupling = np.asarray(couplings, dtype=np.float64)
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("Couplings must be square.")
    if not np.allclose(coupling, coupling.T, atol=1e-10):
        raise ValueError("Couplings must be symmetric.")
    n = coupling.shape[0]
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
    hamiltonian = np.zeros((2**n, 2**n), dtype=np.complex128)
    embedded_x = [operator_on_qubits(x, i, n) for i in range(n)]
    for i, j in combinations(range(n), 2):
        hamiltonian += coupling[i, j] * (embedded_x[i] @ embedded_x[j])
    if local_fields is not None:
        fields = [float(v) for v in local_fields]
        if len(fields) != n:
            raise ValueError("local_fields length mismatch.")
        for i, field in enumerate(fields):
            hamiltonian += field * operator_on_qubits(z, i, n)
    return hamiltonian


def evolve_state(state_vector: ComplexArray, hamiltonian: ComplexArray, time: float) -> ComplexArray:
    vector = np.asarray(state_vector, dtype=np.complex128)
    as_density_matrix(vector)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if vector.ndim != 1 or h.shape != (vector.size, vector.size):
        raise ValueError("Hamiltonian/state shape mismatch.")
    if not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("Hamiltonian must be Hermitian.")
    values, vectors = np.linalg.eigh(h)
    unitary = (vectors * np.exp(-1j * float(time) * values)) @ vectors.conj().T
    result = unitary @ vector
    return np.asarray(result / np.linalg.norm(result), dtype=np.complex128)
