"""Monte-Carlo power calculations for operational response and signature witnesses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .spacetime_witness import (
    CausalSignatureWitness,
    causal_signature_from_null_sheets,
    response_rank_witness,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RankPowerPoint:
    shots_per_setting: int
    trials: int
    true_rank: int
    median_estimated_rank: float
    certification_power: float
    false_extra_rank_rate: float


@dataclass(frozen=True)
class SignaturePowerPoint:
    noise_standard_deviation: float
    trials: int
    correct_inertia_rate: float
    median_null_residual: float
    median_metric_error: float


@dataclass(frozen=True)
class SpatialRankPowerPoint:
    noise_standard_deviation: float
    trials: int
    relative_eigenvalue_threshold: float
    full_rank_power: float
    rank_deficient_false_pass_rate: float
    median_full_relative_eigenvalue: float
    median_null_relative_eigenvalue: float


def _sigmoid(value: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-value))


def simulate_response_jacobian(
    true_jacobian: FloatArray,
    offsets: Sequence[float],
    *,
    step: float,
    shots_per_setting: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Estimate a Bernoulli response Jacobian by central finite differences."""
    jacobian = np.asarray(true_jacobian, dtype=np.float64)
    baseline = np.asarray(offsets, dtype=np.float64)
    if jacobian.ndim != 2 or baseline.shape != (jacobian.shape[0],):
        raise ValueError("Offsets must provide one logistic baseline per response row.")
    if step <= 0 or shots_per_setting < 1:
        raise ValueError("Use a positive step and at least one shot.")
    estimated = np.empty_like(jacobian)
    for column in range(jacobian.shape[1]):
        plus_probability = _sigmoid(baseline + step * jacobian[:, column])
        minus_probability = _sigmoid(baseline - step * jacobian[:, column])
        plus_counts = rng.binomial(shots_per_setting, plus_probability)
        minus_counts = rng.binomial(shots_per_setting, minus_probability)
        derivative_probability = (plus_counts / shots_per_setting - minus_counts / shots_per_setting) / (2.0 * step)
        local_slope = _sigmoid(baseline) * (1.0 - _sigmoid(baseline))
        estimated[:, column] = derivative_probability / local_slope
    return estimated


def estimate_rank_power(
    true_jacobian: FloatArray,
    *,
    shots_per_setting: Sequence[int],
    trials: int = 500,
    step: float = 0.08,
    seed: int = 812731,
    false_positive_alpha: float = 0.05,
) -> tuple[RankPowerPoint, ...]:
    """Estimate target-blind rank-certification power using a zero-Jacobian null."""
    jacobian = np.asarray(true_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or trials < 20 or not 0 < false_positive_alpha < 1:
        raise ValueError("Invalid rank-power configuration.")
    true_rank = int(np.linalg.matrix_rank(jacobian))
    baseline = np.linspace(-0.5, 0.5, jacobian.shape[0])
    output = []
    for shots in shots_per_setting:
        rng = np.random.default_rng(seed + int(shots))
        null_singular = []
        for _ in range(trials):
            null_estimate = simulate_response_jacobian(
                np.zeros_like(jacobian), baseline, step=step, shots_per_setting=int(shots), rng=rng
            )
            null_singular.append(np.linalg.svd(null_estimate, compute_uv=False))
        null_array = np.asarray(null_singular)
        threshold = float(np.quantile(null_array[:, 0], 1.0 - false_positive_alpha))
        ranks = []
        for _ in range(trials):
            estimate = simulate_response_jacobian(
                jacobian, baseline, step=step, shots_per_setting=int(shots), rng=rng
            )
            ranks.append(response_rank_witness(estimate, noise_bound=threshold).certified_lower_bound)
        rank_array = np.asarray(ranks)
        output.append(
            RankPowerPoint(
                shots_per_setting=int(shots),
                trials=trials,
                true_rank=true_rank,
                median_estimated_rank=float(np.median(rank_array)),
                certification_power=float(np.mean(rank_array >= true_rank)),
                false_extra_rank_rate=float(np.mean(rank_array > true_rank)),
            )
        )
    return tuple(output)


def _noisy_signature_metric(
    tangents: FloatArray,
    tilt: FloatArray,
    metric: FloatArray,
    noise: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, float, CausalSignatureWitness]:
    radius_sq = np.einsum("ni,ij,nj->n", tangents, metric, tangents)
    radius = np.sqrt(np.maximum(radius_sq, 0.0))
    midpoint = tangents @ tilt
    plus = midpoint + radius + rng.normal(scale=noise, size=tangents.shape[0])
    minus = midpoint - radius + rng.normal(scale=noise, size=tangents.shape[0])
    witness = causal_signature_from_null_sheets(tangents, plus, minus, tolerance=1e-7)
    fitted = (witness.spatial_metric + witness.spatial_metric.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(fitted)
    relative_minimum = float(eigenvalues[0] / max(np.linalg.norm(fitted, ord="fro"), np.finfo(float).eps))
    return fitted, relative_minimum, witness


def estimate_signature_power(
    tangents: FloatArray,
    tilt: Sequence[float],
    metric: FloatArray,
    *,
    noise_standard_deviations: Sequence[float],
    trials: int = 500,
    seed: int = 812731,
) -> tuple[SignaturePowerPoint, ...]:
    """Monte-Carlo probability of recovering the inertia of noisy null sheets."""
    q = np.asarray(tangents, dtype=np.float64)
    c = np.asarray(tilt, dtype=np.float64)
    h = np.asarray(metric, dtype=np.float64)
    if q.ndim != 2 or c.shape != (q.shape[1],) or h.shape != (q.shape[1], q.shape[1]):
        raise ValueError("Tangent, tilt, and metric dimensions do not match.")
    expected_negative = int(np.linalg.matrix_rank(h))
    output = []
    for index, noise in enumerate(noise_standard_deviations):
        if noise < 0:
            raise ValueError("Noise standard deviations must be nonnegative.")
        rng = np.random.default_rng(seed + index)
        correct, residuals, errors = [], [], []
        for _ in range(trials):
            fitted, _, witness = _noisy_signature_metric(q, c, h, float(noise), rng)
            correct.append(witness.positive == 1 and witness.negative == expected_negative and witness.zero == 0)
            residuals.append(witness.null_residual)
            errors.append(
                np.linalg.norm(fitted - h, ord="fro")
                / max(np.linalg.norm(h, ord="fro"), np.finfo(float).eps)
            )
        output.append(
            SignaturePowerPoint(
                noise_standard_deviation=float(noise),
                trials=trials,
                correct_inertia_rate=float(np.mean(correct)),
                median_null_residual=float(np.median(residuals)),
                median_metric_error=float(np.median(errors)),
            )
        )
    return tuple(output)


def estimate_spatial_rank_power(
    tangents: FloatArray,
    tilt: Sequence[float],
    metric: FloatArray,
    *,
    noise_standard_deviations: Sequence[float],
    trials: int = 500,
    false_positive_alpha: float = 0.05,
    seed: int = 912731,
) -> tuple[SpatialRankPowerPoint, ...]:
    """Certify full spatial rank against a rank-deficient metric control.

    The null control is constructed by setting the smallest eigenvalue of the
    supplied positive metric to zero. A separate calibration ensemble fixes the
    relative-eigenvalue threshold, and an independent null ensemble measures the
    false-pass rate. The tested output is full rank relative to the independently
    reconstructed tangent source; no desired integer rank is passed in.
    """
    q = np.asarray(tangents, dtype=np.float64)
    c = np.asarray(tilt, dtype=np.float64)
    h = np.asarray(metric, dtype=np.float64)
    if (
        q.ndim != 2
        or c.shape != (q.shape[1],)
        or h.shape != (q.shape[1], q.shape[1])
        or trials < 20
        or not 0 < false_positive_alpha < 1
    ):
        raise ValueError("Invalid spatial-rank power configuration.")
    h = (h + h.T) / 2.0
    values, vectors = np.linalg.eigh(h)
    if float(values[0]) <= 0.0:
        raise ValueError("The positive model metric must be positive definite.")
    null_values = values.copy()
    null_values[0] = 0.0
    null_metric = np.asarray((vectors * null_values) @ vectors.T, dtype=np.float64)
    output = []
    for index, noise in enumerate(noise_standard_deviations):
        if noise < 0:
            raise ValueError("Noise standard deviations must be nonnegative.")
        rng = np.random.default_rng(seed + index)
        calibration = [
            _noisy_signature_metric(q, c, null_metric, float(noise), rng)[1]
            for _ in range(trials)
        ]
        threshold = float(np.quantile(calibration, 1.0 - false_positive_alpha))
        null_validation = [
            _noisy_signature_metric(q, c, null_metric, float(noise), rng)[1]
            for _ in range(trials)
        ]
        positive = [
            _noisy_signature_metric(q, c, h, float(noise), rng)[1]
            for _ in range(trials)
        ]
        output.append(
            SpatialRankPowerPoint(
                noise_standard_deviation=float(noise),
                trials=trials,
                relative_eigenvalue_threshold=threshold,
                full_rank_power=float(np.mean(np.asarray(positive) > threshold)),
                rank_deficient_false_pass_rate=float(
                    np.mean(np.asarray(null_validation) > threshold)
                ),
                median_full_relative_eigenvalue=float(np.median(positive)),
                median_null_relative_eigenvalue=float(np.median(null_validation)),
            )
        )
    return tuple(output)
