"""Exact finite-dimensional compatibility bridges for the CF-RSP core."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .factorization import algebra_residual

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModularHamiltonianCheck:
    beta: float
    modular_parameter: float
    physical_time: float
    error: float


def validate_density_matrix(
    state: ComplexArray, *, faithful: bool = False, tolerance: float = 1e-10
) -> ComplexArray:
    rho = np.asarray(state, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square.")
    if not np.allclose(rho, rho.conj().T, atol=tolerance):
        raise ValueError("Density matrix must be Hermitian.")
    if not np.isclose(np.trace(rho), 1.0, atol=tolerance):
        raise ValueError("Density matrix must have unit trace.")
    eigenvalues = np.linalg.eigvalsh(rho).real
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("Density matrix must be positive semidefinite.")
    if faithful and np.min(eigenvalues) <= tolerance:
        raise ValueError("A faithful state must be positive definite.")
    return rho


def _hermitian_function(matrix: ComplexArray, function: Callable[[FloatArray], FloatArray]) -> ComplexArray:
    operator = np.asarray(matrix, dtype=np.complex128)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("Operator must be square.")
    if not np.allclose(operator, operator.conj().T, atol=1e-10):
        raise ValueError("Operator must be Hermitian.")
    values, vectors = np.linalg.eigh(operator)
    transformed = function(values.astype(np.float64))
    return np.asarray((vectors * transformed) @ vectors.conj().T, dtype=np.complex128)


def matrix_log_positive(matrix: ComplexArray, *, tolerance: float = 1e-12) -> ComplexArray:
    operator = np.asarray(matrix, dtype=np.complex128)
    values = np.linalg.eigvalsh(operator).real
    if np.min(values) <= tolerance:
        raise ValueError("Matrix logarithm requires a positive-definite operator.")
    return _hermitian_function(operator, np.log)


def matrix_exponential_hermitian(matrix: ComplexArray) -> ComplexArray:
    return _hermitian_function(matrix, np.exp)


def gibbs_state(hamiltonian: ComplexArray, beta: float) -> ComplexArray:
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if beta <= 0 or h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Use a square Hamiltonian and beta > 0.")
    if not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("Hamiltonian must be Hermitian.")
    unnormalised = matrix_exponential_hermitian(-float(beta) * h)
    return np.asarray(unnormalised / np.trace(unnormalised), dtype=np.complex128)


def modular_hamiltonian(state: ComplexArray, *, traceless_gauge: bool = True) -> ComplexArray:
    rho = validate_density_matrix(state, faithful=True)
    generator = -matrix_log_positive(rho)
    if traceless_gauge:
        generator -= np.trace(generator) / generator.shape[0] * np.eye(
            generator.shape[0], dtype=np.complex128
        )
    return np.asarray(generator, dtype=np.complex128)


def _unitary_from_hermitian(generator: ComplexArray, parameter: float) -> ComplexArray:
    h = np.asarray(generator, dtype=np.complex128)
    if not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("Generator must be Hermitian.")
    values, vectors = np.linalg.eigh(h)
    return np.asarray(
        (vectors * np.exp(-1j * float(parameter) * values)) @ vectors.conj().T,
        dtype=np.complex128,
    )


def modular_flow(observable: ComplexArray, state: ComplexArray, modular_parameter: float) -> ComplexArray:
    """sigma_s(A)=rho^(is) A rho^(-is)=e^(-isK) A e^(isK)."""
    rho = validate_density_matrix(state, faithful=True)
    observable_array = np.asarray(observable, dtype=np.complex128)
    if observable_array.shape != rho.shape:
        raise ValueError("Observable/state shape mismatch.")
    generator = modular_hamiltonian(rho, traceless_gauge=False)
    unitary = _unitary_from_hermitian(generator, modular_parameter)
    return np.asarray(unitary @ observable_array @ unitary.conj().T, dtype=np.complex128)


def heisenberg_flow(observable: ComplexArray, hamiltonian: ComplexArray, time: float) -> ComplexArray:
    """alpha_t(A)=e^(itH) A e^(-itH)."""
    h = np.asarray(hamiltonian, dtype=np.complex128)
    observable_array = np.asarray(observable, dtype=np.complex128)
    if h.shape != observable_array.shape:
        raise ValueError("Observable/Hamiltonian shape mismatch.")
    schrodinger_unitary = _unitary_from_hermitian(h, time)
    return np.asarray(
        schrodinger_unitary.conj().T @ observable_array @ schrodinger_unitary,
        dtype=np.complex128,
    )


def verify_gibbs_modular_alignment(
    observable: ComplexArray,
    hamiltonian: ComplexArray,
    beta: float,
    modular_parameter: float,
) -> ModularHamiltonianCheck:
    """Check sigma_s^rho = alpha_{-beta s} for rho proportional to e^{-beta H}."""
    rho = gibbs_state(hamiltonian, beta)
    modular = modular_flow(observable, rho, modular_parameter)
    physical_time = -float(beta) * float(modular_parameter)
    heisenberg = heisenberg_flow(observable, hamiltonian, physical_time)
    denominator = max(np.linalg.norm(modular, ord="fro"), np.finfo(float).eps)
    error = float(np.linalg.norm(modular - heisenberg, ord="fro") / denominator)
    return ModularHamiltonianCheck(
        beta=float(beta),
        modular_parameter=float(modular_parameter),
        physical_time=physical_time,
        error=error,
    )


def modular_invariance_defect(
    algebra_basis: Sequence[ComplexArray],
    state: ComplexArray,
    modular_parameters: Iterable[float],
) -> float:
    """Maximum distance of modularly evolved basis elements from the algebra."""
    if not algebra_basis:
        raise ValueError("Algebra basis cannot be empty.")
    parameters = [float(value) for value in modular_parameters]
    if not parameters:
        raise ValueError("At least one modular parameter is required.")
    defects = [
        algebra_residual(modular_flow(element, state, parameter), algebra_basis)
        for parameter in parameters
        for element in algebra_basis
    ]
    return float(max(defects, default=0.0))


def born_probabilities(state: ComplexArray, effects: Sequence[ComplexArray]) -> FloatArray:
    rho = validate_density_matrix(state)
    if not effects:
        raise ValueError("At least one POVM effect is required.")
    effect_arrays = [np.asarray(effect, dtype=np.complex128) for effect in effects]
    if any(effect.shape != rho.shape for effect in effect_arrays):
        raise ValueError("POVM effects must match the state dimension.")
    if any(not np.allclose(effect, effect.conj().T, atol=1e-10) for effect in effect_arrays):
        raise ValueError("POVM effects must be Hermitian.")
    if any(np.min(np.linalg.eigvalsh(effect)) < -1e-10 for effect in effect_arrays):
        raise ValueError("POVM effects must be positive semidefinite.")
    if not np.allclose(sum(effect_arrays), np.eye(rho.shape[0]), atol=1e-10):
        raise ValueError("POVM effects must sum to the identity.")
    probabilities = np.asarray(
        [np.trace(rho @ effect).real for effect in effect_arrays], dtype=np.float64
    )
    probabilities[np.abs(probabilities) < 1e-14] = 0.0
    return probabilities


def quantum_relative_entropy(
    state: ComplexArray, reference: ComplexArray, *, base: float = np.e
) -> float:
    rho = validate_density_matrix(state, faithful=True)
    sigma = validate_density_matrix(reference, faithful=True)
    if rho.shape != sigma.shape or base <= 0 or np.isclose(base, 1.0):
        raise ValueError("Invalid relative-entropy inputs.")
    value = np.trace(rho @ (matrix_log_positive(rho) - matrix_log_positive(sigma))).real
    return float(max(0.0, value / np.log(base)))


def lindblad_rhs(
    state: ComplexArray,
    hamiltonian: ComplexArray,
    jump_operators: Sequence[ComplexArray],
) -> ComplexArray:
    rho = validate_density_matrix(state)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.shape != rho.shape or not np.allclose(h, h.conj().T, atol=1e-10):
        raise ValueError("Hamiltonian must be Hermitian and match the state.")
    derivative = -1j * (h @ rho - rho @ h)
    for raw in jump_operators:
        jump = np.asarray(raw, dtype=np.complex128)
        if jump.shape != rho.shape:
            raise ValueError("Jump-operator shape mismatch.")
        rate = jump.conj().T @ jump
        derivative += jump @ rho @ jump.conj().T - 0.5 * (rate @ rho + rho @ rate)
    return np.asarray(derivative, dtype=np.complex128)


def spohn_entropy_production(
    state: ComplexArray,
    stationary_state: ComplexArray,
    generator_action: ComplexArray,
) -> float:
    """-Tr[L(rho)(log rho-log sigma)] for a supplied QMS generator action."""
    rho = validate_density_matrix(state, faithful=True)
    sigma = validate_density_matrix(stationary_state, faithful=True)
    action = np.asarray(generator_action, dtype=np.complex128)
    if action.shape != rho.shape or sigma.shape != rho.shape:
        raise ValueError("Shape mismatch in entropy-production inputs.")
    value = -np.trace(
        action @ (matrix_log_positive(rho) - matrix_log_positive(sigma))
    ).real
    return float(0.0 if abs(value) < 1e-12 else value)


def pairwise_commutativity_defect(operators: Sequence[ComplexArray]) -> float:
    arrays = [np.asarray(operator, dtype=np.complex128) for operator in operators]
    if len(arrays) < 2:
        return 0.0
    if any(operator.shape != arrays[0].shape for operator in arrays):
        raise ValueError("Operator shapes do not match.")
    defects: list[float] = []
    for index, left in enumerate(arrays):
        for right in arrays[index + 1 :]:
            denominator = 2 * max(
                np.linalg.norm(left, ord="fro") * np.linalg.norm(right, ord="fro"),
                np.finfo(float).eps,
            )
            defects.append(np.linalg.norm(left @ right - right @ left, ord="fro") / denominator)
    return float(max(defects, default=0.0))
