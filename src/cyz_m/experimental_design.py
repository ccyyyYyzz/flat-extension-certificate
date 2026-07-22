"""Target-blind experimental design for null-sheet and response witnesses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NullSheetDesign:
    selected_indices: tuple[int, ...]
    selected_tangents: FloatArray
    linear_singular_values: FloatArray
    quadratic_singular_values: FloatArray
    linear_condition_number: float
    quadratic_condition_number: float
    combined_logdet: float


@dataclass(frozen=True)
class RamseyBudget:
    interrogation_time: float
    visibility: float
    frequency_standard_error: float
    required_shots: int


def quadratic_features(tangents: FloatArray) -> FloatArray:
    q = np.asarray(tangents, dtype=np.float64)
    if q.ndim != 2 or not np.all(np.isfinite(q)):
        raise ValueError("Tangents must be a finite two-dimensional array.")
    pairs = [(i, j) for i in range(q.shape[1]) for j in range(i, q.shape[1])]
    features = np.empty((q.shape[0], len(pairs)), dtype=np.float64)
    for column, (i, j) in enumerate(pairs):
        features[:, column] = q[:, i] * q[:, j] * (1.0 if i == j else 2.0)
    return features


def _whiten_columns(matrix: FloatArray, tolerance: float) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    scales = np.sqrt(np.mean(np.square(value), axis=0))
    if np.any(scales <= tolerance):
        raise ValueError("Candidate design has a column with no variation.")
    return value / scales


def greedy_null_sheet_design(
    candidate_tangents: FloatArray,
    sample_count: int,
    *,
    ridge: float = 1e-8,
    tolerance: float = 1e-12,
) -> NullSheetDesign:
    """Greedy block-D-optimal design for tilt and metric reconstruction.

    The midpoint and squared-radius observations have separate linear and
    quadratic design matrices. The objective therefore maximizes the sum of
    their regularized log determinants. The source dimension is inferred from
    the candidate table; no target rank, signature, or metric is supplied.
    """
    q = np.asarray(candidate_tangents, dtype=np.float64)
    if q.ndim != 2 or not np.all(np.isfinite(q)) or ridge <= 0 or tolerance <= 0:
        raise ValueError("Invalid candidate design or regularization.")
    linear = _whiten_columns(q, tolerance)
    quadratic = _whiten_columns(quadratic_features(q), tolerance)
    required = max(linear.shape[1], quadratic.shape[1])
    if not required <= sample_count <= q.shape[0]:
        raise ValueError("sample_count must identify both linear and quadratic fits.")
    gram_linear = ridge * np.eye(linear.shape[1])
    gram_quadratic = ridge * np.eye(quadratic.shape[1])
    selected: list[int] = []
    remaining = set(range(q.shape[0]))
    for _ in range(sample_count):
        best_index = None
        best_gain = -np.inf
        inverse_linear = np.linalg.inv(gram_linear)
        inverse_quadratic = np.linalg.inv(gram_quadratic)
        for index in remaining:
            row_linear = linear[index]
            row_quadratic = quadratic[index]
            gain = float(
                np.log1p(row_linear @ inverse_linear @ row_linear)
                + np.log1p(row_quadratic @ inverse_quadratic @ row_quadratic)
            )
            if gain > best_gain:
                best_gain = gain
                best_index = index
        assert best_index is not None
        selected.append(best_index)
        remaining.remove(best_index)
        row_linear = linear[best_index]
        row_quadratic = quadratic[best_index]
        gram_linear += np.outer(row_linear, row_linear)
        gram_quadratic += np.outer(row_quadratic, row_quadratic)
    chosen_linear = q[selected]
    chosen_quadratic = quadratic_features(chosen_linear)
    singular_linear = np.linalg.svd(chosen_linear, compute_uv=False)
    singular_quadratic = np.linalg.svd(chosen_quadratic, compute_uv=False)
    linear_condition = float(singular_linear[0] / singular_linear[-1])
    quadratic_condition = float(singular_quadratic[0] / singular_quadratic[-1])
    sign_linear, logdet_linear = np.linalg.slogdet(gram_linear)
    sign_quadratic, logdet_quadratic = np.linalg.slogdet(gram_quadratic)
    if sign_linear <= 0 or sign_quadratic <= 0:
        raise RuntimeError("Regularized information matrix lost positive definiteness.")
    return NullSheetDesign(
        selected_indices=tuple(selected),
        selected_tangents=np.asarray(chosen_linear),
        linear_singular_values=np.asarray(singular_linear),
        quadratic_singular_values=np.asarray(singular_quadratic),
        linear_condition_number=linear_condition,
        quadratic_condition_number=quadratic_condition,
        combined_logdet=float(logdet_linear + logdet_quadratic),
    )


def ramsey_frequency_budget(
    target_standard_error: float,
    dephasing_rate: float,
    *,
    contrast: float = 1.0,
    interrogation_time: float | None = None,
) -> RamseyBudget:
    """Idealized Ramsey shot budget under exponential contrast loss.

    The planning model uses
    ``sigma_omega = exp(gamma*t)/(contrast*t*sqrt(N))``. If time is not
    supplied, it chooses the Fisher-optimal ``t=1/gamma``. This function is
    not a hardware calibration or a substitute for a full noise simulation.
    """
    error = float(target_standard_error)
    gamma = float(dephasing_rate)
    c = float(contrast)
    if error <= 0 or gamma < 0 or not 0 < c <= 1:
        raise ValueError("Use positive error, nonnegative dephasing, and 0<contrast<=1.")
    if interrogation_time is None:
        time = 1.0 / gamma if gamma > 0 else 1.0
    else:
        time = float(interrogation_time)
    if time <= 0:
        raise ValueError("Interrogation time must be positive.")
    visibility = c * np.exp(-gamma * time)
    shots = int(np.ceil(1.0 / np.square(error * visibility * time)))
    achieved = float(1.0 / (visibility * time * np.sqrt(shots)))
    return RamseyBudget(time, float(visibility), achieved, shots)
