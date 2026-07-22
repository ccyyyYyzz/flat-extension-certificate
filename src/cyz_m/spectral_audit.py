"""Pre-registered, target-blind spectral-dimension diagnostics for CF-RSP.

A finite graph necessarily has d_s(tau) -> 0 at both diffusion endpoints.  This
module therefore treats a dimension plateau as a mesoscopic hypothesis that
must be selected without knowing the desired dimension and then survive
perturbation and null-model checks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlateauRule:
    """Frozen criteria chosen before inspecting a target dimension."""

    min_log_width_decades: float = 0.75
    min_points: int = 10
    max_standard_deviation: float = 0.15
    max_absolute_slope: float = 0.20
    edge_fraction: float = 0.10
    min_robust_fraction: float = 0.80
    min_overlap_fraction: float = 0.50
    max_replica_mean_spread: float = 0.25
    target_dimension: float | None = None
    target_tolerance: float = 0.25
    max_null_pass_rate: float = 0.05

    def validate(self) -> None:
        if self.min_log_width_decades <= 0:
            raise ValueError("min_log_width_decades must be positive.")
        if self.min_points < 3:
            raise ValueError("min_points must be at least three.")
        if self.max_standard_deviation < 0 or self.max_absolute_slope < 0:
            raise ValueError("Plateau tolerances must be nonnegative.")
        if not 0 <= self.edge_fraction < 0.5:
            raise ValueError("edge_fraction must lie in [0, 0.5).")
        if not 0 <= self.min_robust_fraction <= 1:
            raise ValueError("min_robust_fraction must lie in [0, 1].")
        if not 0 <= self.min_overlap_fraction <= 1:
            raise ValueError("min_overlap_fraction must lie in [0, 1].")
        if self.max_replica_mean_spread < 0 or self.target_tolerance < 0:
            raise ValueError("Spread/tolerance values must be nonnegative.")
        if not 0 <= self.max_null_pass_rate <= 1:
            raise ValueError("max_null_pass_rate must lie in [0, 1].")

    def without_target(self) -> "PlateauRule":
        """Return the structural rule used by target-blind window selection."""
        return replace(self, target_dimension=None)


@dataclass(frozen=True)
class SpectralPlateau:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    log_width_decades: float
    mean_dimension: float
    standard_deviation: float
    slope_per_decade: float


@dataclass(frozen=True)
class SpectralAudit:
    base_plateau: SpectralPlateau | None
    replica_plateaus: tuple[SpectralPlateau | None, ...]
    robust_fraction: float
    target_pass: bool | None
    null_pass_rate: float | None
    hypothesis_pass: bool
    reason: str


def _validate_times(diffusion_times: Iterable[float]) -> FloatArray:
    times = np.asarray(list(diffusion_times), dtype=np.float64)
    if times.ndim != 1 or times.size < 3:
        raise ValueError("At least three diffusion times are required.")
    if np.any(~np.isfinite(times)) or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Diffusion times must be finite, positive and increasing.")
    return times


def validate_laplacian(laplacian: FloatArray, *, tolerance: float = 1e-9) -> FloatArray:
    matrix = np.asarray(laplacian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("Laplacian must be a non-empty square matrix.")
    if np.any(~np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=tolerance):
        raise ValueError("Laplacian must be finite and symmetric.")
    if np.min(np.linalg.eigvalsh(matrix)) < -tolerance:
        raise ValueError("Laplacian must be positive semidefinite.")
    return matrix


def heat_trace(laplacian: FloatArray, diffusion_times: Iterable[float]) -> FloatArray:
    matrix = validate_laplacian(laplacian)
    times = _validate_times(diffusion_times)
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    return np.asarray(
        [np.mean(np.exp(-time * eigenvalues)) for time in times],
        dtype=np.float64,
    )


def spectral_dimension_curve(
    laplacian: FloatArray, diffusion_times: Iterable[float]
) -> tuple[FloatArray, FloatArray]:
    times = _validate_times(diffusion_times)
    trace = heat_trace(laplacian, times)
    dimension = -2.0 * np.gradient(np.log(trace), np.log(times), edge_order=2)
    return trace, np.asarray(dimension, dtype=np.float64)


def finite_graph_endpoint_data(laplacian: FloatArray, *, tolerance: float = 1e-9) -> dict[str, float | int]:
    """Data entering the proof that d_s tends to zero at both endpoints."""
    matrix = validate_laplacian(laplacian, tolerance=tolerance)
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    zero_modes = int(np.sum(eigenvalues <= tolerance))
    return {
        "vertex_count": int(matrix.shape[0]),
        "zero_mode_count": zero_modes,
        "mean_eigenvalue": float(np.mean(eigenvalues)),
        "small_time_heat_trace_limit": 1.0,
        "large_time_heat_trace_limit": zero_modes / matrix.shape[0],
        "small_time_spectral_dimension_limit": 0.0,
        "large_time_spectral_dimension_limit": 0.0,
    }


def _window_plateau(
    times: FloatArray,
    dimension: FloatArray,
    start: int,
    end: int,
) -> SpectralPlateau:
    x = np.log10(times[start : end + 1])
    y = dimension[start : end + 1]
    slope, _ = np.polyfit(x, y, deg=1)
    return SpectralPlateau(
        start_index=start,
        end_index=end,
        start_time=float(times[start]),
        end_time=float(times[end]),
        log_width_decades=float(x[-1] - x[0]),
        mean_dimension=float(np.mean(y)),
        standard_deviation=float(np.std(y)),
        slope_per_decade=float(slope),
    )


def detect_spectral_plateaus(
    diffusion_times: Iterable[float],
    dimensions: Sequence[float],
    *,
    rule: PlateauRule = PlateauRule(),
) -> tuple[SpectralPlateau, ...]:
    """Select plateaus without using ``rule.target_dimension``.

    Results are ranked by width, then flatness.  Consequently changing the
    desired dimension cannot change which interval is selected.
    """
    rule.validate()
    times = _validate_times(diffusion_times)
    values = np.asarray(dimensions, dtype=np.float64)
    if values.shape != times.shape or np.any(~np.isfinite(values)):
        raise ValueError("dimensions must be finite and match diffusion_times.")

    edge = int(np.floor(rule.edge_fraction * times.size))
    first = edge
    last = times.size - edge - 1
    if last - first + 1 < rule.min_points:
        return ()

    candidates: list[SpectralPlateau] = []
    for start in range(first, last - rule.min_points + 2):
        for end in range(start + rule.min_points - 1, last + 1):
            width = np.log10(times[end] / times[start])
            if width < rule.min_log_width_decades:
                continue
            plateau = _window_plateau(times, values, start, end)
            if (
                plateau.standard_deviation <= rule.max_standard_deviation
                and abs(plateau.slope_per_decade) <= rule.max_absolute_slope
                and plateau.mean_dimension >= 0.0
            ):
                candidates.append(plateau)

    candidates.sort(
        key=lambda item: (
            -item.log_width_decades,
            item.standard_deviation,
            abs(item.slope_per_decade),
            item.start_index,
        )
    )
    maximal: list[SpectralPlateau] = []
    for candidate in candidates:
        if any(
            existing.start_index <= candidate.start_index
            and existing.end_index >= candidate.end_index
            for existing in maximal
        ):
            continue
        maximal.append(candidate)
    return tuple(maximal)


def best_spectral_plateau(
    diffusion_times: Iterable[float],
    dimensions: Sequence[float],
    *,
    rule: PlateauRule = PlateauRule(),
) -> SpectralPlateau | None:
    plateaus = detect_spectral_plateaus(diffusion_times, dimensions, rule=rule)
    return plateaus[0] if plateaus else None


def _overlap_fraction(left: SpectralPlateau, right: SpectralPlateau) -> float:
    left_start, left_end = np.log10(left.start_time), np.log10(left.end_time)
    right_start, right_end = np.log10(right.start_time), np.log10(right.end_time)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    denominator = max(min(left_end - left_start, right_end - right_start), np.finfo(float).eps)
    return float(overlap / denominator)


def _target_match(plateau: SpectralPlateau | None, rule: PlateauRule) -> bool | None:
    if rule.target_dimension is None:
        return None
    if plateau is None:
        return False
    return abs(plateau.mean_dimension - rule.target_dimension) <= rule.target_tolerance


def audit_spectral_dimension(
    laplacians: Sequence[FloatArray],
    diffusion_times: Iterable[float],
    *,
    rule: PlateauRule = PlateauRule(),
    null_laplacians: Sequence[FloatArray] = (),
) -> SpectralAudit:
    """Audit one base graph plus pre-declared perturbation replicas and nulls."""
    rule.validate()
    if not laplacians:
        raise ValueError("At least one Laplacian is required.")
    times = _validate_times(diffusion_times)

    plateaus: list[SpectralPlateau | None] = []
    for laplacian in laplacians:
        _, curve = spectral_dimension_curve(laplacian, times)
        plateaus.append(best_spectral_plateau(times, curve, rule=rule))
    base = plateaus[0]

    if base is None:
        robust_fraction = 0.0
    else:
        matches = 0
        for plateau in plateaus:
            if plateau is None:
                continue
            if (
                _overlap_fraction(base, plateau) >= rule.min_overlap_fraction
                and abs(base.mean_dimension - plateau.mean_dimension)
                <= rule.max_replica_mean_spread
            ):
                matches += 1
        robust_fraction = matches / len(plateaus)

    target_pass = _target_match(base, rule)
    null_pass_rate: float | None = None
    if null_laplacians:
        null_matches = 0
        for laplacian in null_laplacians:
            _, curve = spectral_dimension_curve(laplacian, times)
            plateau = best_spectral_plateau(times, curve, rule=rule)
            match = _target_match(plateau, rule)
            if (match is True) or (match is None and plateau is not None):
                null_matches += 1
        null_pass_rate = null_matches / len(null_laplacians)

    structural_pass = base is not None and robust_fraction >= rule.min_robust_fraction
    target_gate = target_pass is not False
    null_gate = null_pass_rate is None or null_pass_rate <= rule.max_null_pass_rate
    hypothesis_pass = bool(structural_pass and target_gate and null_gate)

    if base is None:
        reason = "no target-blind plateau passed the pre-registered structural rule"
    elif robust_fraction < rule.min_robust_fraction:
        reason = "plateau failed perturbation robustness"
    elif target_pass is False:
        reason = "selected plateau exists but misses the pre-declared target"
    elif not null_gate:
        reason = "null-model false-positive rate exceeds the pre-registered bound"
    else:
        reason = "all supplied pre-registered gates passed"

    return SpectralAudit(
        base_plateau=base,
        replica_plateaus=tuple(plateaus),
        robust_fraction=float(robust_fraction),
        target_pass=target_pass,
        null_pass_rate=null_pass_rate,
        hypothesis_pass=hypothesis_pass,
        reason=reason,
    )
