"""Target-blind witnesses for operational tangent dimension and causal signature.

The functions in this module deliberately avoid a target Hilbert-space dimension.
They operate on experimentally reconstructed response Jacobians, calibrated atomic
coefficient maps, null-sheet spectroscopy, and reversible protocol generators.

The main diagnostics separate three logically independent obstructions:

* response-copy multiplicity (universality),
* atomic/full-protocol kernel mismatch (completeness), and
* new directions appearing only at greater protocol or Lie depth.

This is a response-space witness, not a Hilbert-space dimension witness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class RankWitness:
    singular_values: FloatArray
    threshold: float
    numerical_rank: int
    certified_lower_bound: int
    condition_number: float


@dataclass(frozen=True)
class ObstructionAudit:
    universality_gap: float | None
    coefficient_rank: int | None
    completeness_gap: float
    atomic_rank: int
    full_rank: int
    derivative_ranks: tuple[int, ...]
    derivative_growth: tuple[int, ...]
    orbit_ranks: tuple[int, ...]
    orbit_growth: tuple[int, ...]


@dataclass(frozen=True)
class CausalSignatureWitness:
    tilt: FloatArray
    spatial_metric: FloatArray
    characteristic_form: FloatArray
    positive: int
    negative: int
    zero: int
    spatial_rank: int
    midpoint_residual: float
    radius_residual: float
    null_residual: float


@dataclass(frozen=True)
class StableCADCertificate:
    kernel_projector_gap: float
    base_atomic_kernel_gap: float
    quotient_dimension: int
    target_dimension: int
    descended_lower_singular_bound: float
    base_coisometry_defect: float
    transported_line_residual: float
    universality_gap: float
    certified_isomorphism: bool
    # Appended for the Theorem-3 finite-error wiring (issue #3). New fields carry
    # defaults and are placed last so positional construction stays compatible.
    error_bound_vacuous: bool = False
    certificate_status: str = "inconclusive"


@dataclass(frozen=True)
class LieClosureReport:
    dimensions_by_depth: tuple[int, ...]
    closure_depth: int
    algebra_dimension: int
    basis: tuple[ComplexArray, ...]


@dataclass(frozen=True)
class RamificationWitness:
    raw_parameter: float
    qfi_raw: float
    regular_coordinate: float
    qfi_regular: float
    first_derivative_visible: bool
    second_order_visible: bool


def _real_matrix(matrix: FloatArray, name: str) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite two-dimensional real array.")
    return value


def _svd_threshold(values: FloatArray, relative_tolerance: float, noise_bound: float) -> float:
    if relative_tolerance <= 0 or noise_bound < 0:
        raise ValueError("Use relative_tolerance > 0 and noise_bound >= 0.")
    scale = float(values[0]) if values.size else 0.0
    return max(float(noise_bound), float(relative_tolerance) * max(scale, 1.0))


def response_rank_witness(
    response: FloatArray,
    *,
    noise_bound: float = 0.0,
    relative_tolerance: float = 1e-10,
) -> RankWitness:
    """Return a target-blind lower bound on response tangent dimension.

    If the operator-norm error is bounded by ``noise_bound``, Weyl's inequality
    certifies every observed singular value strictly above that bound as a true
    nonzero singular value. No desired rank is supplied.
    """
    matrix = _real_matrix(response, "response")
    values = np.linalg.svd(matrix, compute_uv=False)
    threshold = _svd_threshold(values, relative_tolerance, noise_bound)
    numerical_rank = int(np.sum(values > threshold))
    certified = int(np.sum(values > float(noise_bound))) if noise_bound > 0 else numerical_rank
    if values.size == 0 or values[-1] <= threshold:
        condition = float("inf")
    else:
        condition = float(values[0] / values[-1])
    return RankWitness(
        singular_values=np.asarray(values, dtype=np.float64),
        threshold=float(threshold),
        numerical_rank=numerical_rank,
        certified_lower_bound=certified,
        condition_number=condition,
    )


def _kernel_projector(matrix: FloatArray, tolerance: float) -> FloatArray:
    value = _real_matrix(matrix, "matrix")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    _, singular, vh = np.linalg.svd(value, full_matrices=True)
    scale = float(singular[0]) if singular.size else 1.0
    rank = int(np.sum(singular > tolerance * max(scale, 1.0)))
    kernel = vh[rank:].T
    if kernel.size == 0:
        return np.zeros((value.shape[1], value.shape[1]), dtype=np.float64)
    return np.asarray(kernel @ kernel.T, dtype=np.float64)


def kernel_completeness_gap(
    atomic_response: FloatArray,
    full_response: FloatArray,
    *,
    tolerance: float = 1e-10,
) -> float:
    """Operator-norm distance between atomic and full-protocol kernel projectors."""
    atomic = _real_matrix(atomic_response, "atomic_response")
    full = _real_matrix(full_response, "full_response")
    if atomic.shape[1] != full.shape[1]:
        raise ValueError("Atomic and full responses must share a source dimension.")
    return float(
        np.linalg.norm(
            _kernel_projector(atomic, tolerance) - _kernel_projector(full, tolerance),
            ord=2,
        )
    )


def universality_gap(
    coefficient_matrix: FloatArray,
    *,
    normalize_active_rows: bool = True,
    tolerance: float = 1e-12,
) -> tuple[float | None, int | None, FloatArray]:
    """Measure departure of calibrated response coefficients from one line."""
    matrix = _real_matrix(coefficient_matrix, "coefficient_matrix").copy()
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if normalize_active_rows:
        norms = np.linalg.norm(matrix, axis=1)
        active = norms > tolerance
        matrix[active] /= norms[active, None]
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= tolerance:
        return None, None, np.asarray([], dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    residual = float(np.sqrt(np.sum(np.square(singular[1:]))) / norm)
    rank = int(np.sum(singular > tolerance * max(float(singular[0]), 1.0)))
    return residual, rank, np.asarray(singular, dtype=np.float64)


def nested_ranks(
    blocks: Sequence[FloatArray], *, tolerance: float = 1e-10
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ranks and growth for a nested sequence of response blocks."""
    if not blocks:
        raise ValueError("At least one block is required.")
    ranks = []
    previous_columns: int | None = None
    for block in blocks:
        matrix = _real_matrix(block, "response block")
        if previous_columns is None:
            previous_columns = matrix.shape[1]
        elif matrix.shape[1] != previous_columns:
            raise ValueError("All nested blocks must share the same source dimension.")
        ranks.append(response_rank_witness(matrix, relative_tolerance=tolerance).numerical_rank)
    if any(right < left for left, right in zip(ranks, ranks[1:])):
        raise ValueError("Blocks are not numerically nested: rank decreased with depth.")
    base = ranks[0]
    return tuple(ranks), tuple(rank - base for rank in ranks)


def obstruction_audit(
    coefficient_matrix: FloatArray | None,
    atomic_response: FloatArray,
    full_response_blocks: Sequence[FloatArray],
    orbit_response_blocks: Sequence[FloatArray],
    *,
    tolerance: float = 1e-10,
) -> ObstructionAudit:
    """Audit multiplicity, completeness, derivative depth, and orbit depth."""
    if not full_response_blocks or not orbit_response_blocks:
        raise ValueError("Use nonempty derivative and orbit response sequences.")
    atomic = _real_matrix(atomic_response, "atomic_response")
    full_final = _real_matrix(full_response_blocks[-1], "full_response")
    completeness = kernel_completeness_gap(atomic, full_final, tolerance=tolerance)
    derivative_ranks, derivative_growth = nested_ranks(full_response_blocks, tolerance=tolerance)
    orbit_ranks, orbit_growth = nested_ranks(orbit_response_blocks, tolerance=tolerance)
    if coefficient_matrix is None:
        uni_gap = None
        coefficient_rank = None
    else:
        uni_gap, coefficient_rank, _ = universality_gap(coefficient_matrix, tolerance=tolerance)
    return ObstructionAudit(
        universality_gap=uni_gap,
        coefficient_rank=coefficient_rank,
        completeness_gap=completeness,
        atomic_rank=response_rank_witness(atomic, relative_tolerance=tolerance).numerical_rank,
        full_rank=response_rank_witness(full_final, relative_tolerance=tolerance).numerical_rank,
        derivative_ranks=derivative_ranks,
        derivative_growth=derivative_growth,
        orbit_ranks=orbit_ranks,
        orbit_growth=orbit_growth,
    )


def _quadratic_design(samples: FloatArray) -> tuple[FloatArray, tuple[tuple[int, int], ...]]:
    values = _real_matrix(samples, "tangent samples")
    dimension = values.shape[1]
    pairs = tuple((i, j) for i in range(dimension) for j in range(i, dimension))
    design = np.empty((values.shape[0], len(pairs)), dtype=np.float64)
    for column, (i, j) in enumerate(pairs):
        design[:, column] = values[:, i] * values[:, j] * (1.0 if i == j else 2.0)
    return design, pairs


def causal_signature_from_null_sheets(
    tangents: FloatArray,
    omega_plus: Sequence[float],
    omega_minus: Sequence[float],
    *,
    tolerance: float = 1e-9,
) -> CausalSignatureWitness:
    """Reconstruct a quadratic causal form from two measured null sheets."""
    q = _real_matrix(tangents, "tangents")
    plus = np.asarray(omega_plus, dtype=np.float64)
    minus = np.asarray(omega_minus, dtype=np.float64)
    if plus.shape != (q.shape[0],) or minus.shape != plus.shape:
        raise ValueError("Each null sheet must supply one frequency per tangent sample.")
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("Null-sheet frequencies must be finite.")
    midpoint = 0.5 * (plus + minus)
    radius_sq = np.square(0.5 * (plus - minus))
    tilt, *_ = np.linalg.lstsq(q, midpoint, rcond=None)
    design, pairs = _quadratic_design(q)
    coefficients, *_ = np.linalg.lstsq(design, radius_sq, rcond=None)
    metric = np.zeros((q.shape[1], q.shape[1]), dtype=np.float64)
    for value, (i, j) in zip(coefficients, pairs):
        metric[i, j] = metric[j, i] = value
    fitted_midpoint = q @ tilt
    fitted_radius = np.einsum("ni,ij,nj->n", q, metric, q)
    midpoint_residual = float(np.linalg.norm(midpoint - fitted_midpoint) / max(np.linalg.norm(midpoint), 1.0))
    radius_residual = float(np.linalg.norm(radius_sq - fitted_radius) / max(np.linalg.norm(radius_sq), 1.0))
    form = np.empty((q.shape[1] + 1, q.shape[1] + 1), dtype=np.float64)
    form[0, 0] = 1.0
    form[0, 1:] = form[1:, 0] = -tilt
    form[1:, 1:] = np.outer(tilt, tilt) - metric
    eigenvalues = np.linalg.eigvalsh((form + form.T) / 2.0)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    positive = int(np.sum(eigenvalues > tolerance * scale))
    negative = int(np.sum(eigenvalues < -tolerance * scale))
    zero = int(eigenvalues.size - positive - negative)
    spatial_eigenvalues = np.linalg.eigvalsh((metric + metric.T) / 2.0)
    spatial_scale = max(float(np.max(np.abs(spatial_eigenvalues))), 1.0)
    spatial_rank = int(np.sum(spatial_eigenvalues > tolerance * spatial_scale))
    full_vectors_plus = np.column_stack([plus, q])
    full_vectors_minus = np.column_stack([minus, q])
    residuals = np.concatenate([
        np.abs(np.einsum("ni,ij,nj->n", full_vectors_plus, form, full_vectors_plus)),
        np.abs(np.einsum("ni,ij,nj->n", full_vectors_minus, form, full_vectors_minus)),
    ])
    normalization = max(
        float(np.max(np.square(np.concatenate([plus, minus])))),
        float(np.max(np.sum(np.square(q), axis=1))),
        1.0,
    )
    return CausalSignatureWitness(
        tilt=np.asarray(tilt, dtype=np.float64),
        spatial_metric=np.asarray(metric, dtype=np.float64),
        characteristic_form=np.asarray(form, dtype=np.float64),
        positive=positive,
        negative=negative,
        zero=zero,
        spatial_rank=spatial_rank,
        midpoint_residual=midpoint_residual,
        radius_residual=radius_residual,
        null_residual=float(np.max(residuals) / normalization),
    )


def stable_cad_certificate(
    base_response: FloatArray,
    transported_responses: Sequence[FloatArray],
    full_response: FloatArray,
    *,
    tolerance: float = 1e-10,
    noise_norm: float | None = None,
    spectral_gap: float | None = None,
    atomic_noise_norm: float | None = None,
    atomic_spectral_gap: float | None = None,
    full_noise_norm: float | None = None,
    full_spectral_gap: float | None = None,
) -> StableCADCertificate:
    """Quantitative near-CAD certificate with an explicit quotient bound.

    A strict isomorphism verdict additionally requires that transported responses
    lie on one calibrated response line and that the base response has the same
    kernel as their stack. Without those conditions, extra copies can fill the
    base kernel and would invalidate the one-target lower bound.

    Certificate semantics (Theorem 3, ``research/C7_THEOREM_PACKAGE.md`` §4).
    ``certified_isomorphism`` reports a *bounded-angle near-isomorphism*: a
    kernel-projector mismatch ``eta < 1`` together with matching kernel
    dimensions certifies that the atomic and full-protocol kernels agree up to a
    sine-angle at most ``eta`` (descended lower singular bound
    ``sqrt(lambda_min) * sqrt(1 - eta**2) > 0``). It is **never** a claim of
    exact kernel equality; exact ``K_atomic = K_full`` holds only under the exact
    structural closure hypotheses of that package (one-step non-growth plus effect
    orbit spanning), which this numerical certificate does not by itself verify.

    Two-matrix finite-error wiring (Theorem 3, R3 §E). The package's certified
    mismatch is the **two-matrix triangle bound**

        ``eta_true <= eta_obs + b_A + b_F; near-isomorphism only; equality
        requires the exact structural closure hypotheses.``

    where ``eta_obs`` is the measured kernel-projector gap and ``b_A``, ``b_F``
    are the atomic and full subspace-error terms. Supply the four keywords
    ``atomic_noise_norm``, ``atomic_spectral_gap``, ``full_noise_norm`` and
    ``full_spectral_gap`` together (all-or-none; ``ValueError`` otherwise). For
    each matrix a rank-stable certified lower gap ``gamma_safe = observed_gap -
    eps`` is formed; if either ``gamma_safe <= 0`` no bound exists, the consumed
    error is flagged vacuous, ``certificate_status='inconclusive'`` and
    ``certified_isomorphism=False``. Otherwise ``b_A = min(1, eps_A/gamma_safe_A)``
    and ``b_F = min(1, eps_F/gamma_safe_F)`` and the consumed mismatch is the
    **sum** ``eta = eta_obs + b_A + b_F`` (never ``max``: ``max`` underestimates
    the triangle bound). If ``eta >= 1`` the bound is vacuous and the verdict is
    ``'inconclusive'``.

    Deprecated single-pair wiring. Passing the legacy ``noise_norm`` and
    ``spectral_gap`` pair instead folds a single Wedin/Davis--Kahan bound in via
    ``eta = max(measured_eta, wedin_bound)``. This underestimates the two-matrix
    triangle bound and is retained only for backward compatibility; prefer the
    four two-matrix keywords. It is a ``ValueError`` to mix the two APIs.

    ``certificate_status`` is one of ``'certified'`` (all gates pass, no vacuous
    bound), ``'inconclusive'`` (a consumed error bound is vacuous: ``eta`` at or
    beyond 1, ``noise >= gap``, or ``gamma_safe <= 0``), or ``'failed'`` (a
    structural gate fails for a non-vacuous reason).
    """
    base = _real_matrix(base_response, "base_response")
    full = _real_matrix(full_response, "full_response")
    if full.shape[1] != base.shape[1]:
        raise ValueError("Base and full responses must share a source dimension.")
    responses = [base]
    for response in transported_responses:
        value = _real_matrix(response, "transported response")
        if value.shape != base.shape:
            raise ValueError("Transported responses must have the base response shape.")
        responses.append(value)
    gram = base @ base.T
    gram_eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    lambda_min = float(np.min(gram_eigenvalues))
    if lambda_min < -tolerance:
        raise ValueError("Base response target Gram is not positive semidefinite.")
    scalar = float(np.trace(gram) / max(base.shape[0], 1))
    coisometry_defect = float(
        np.linalg.norm(gram - scalar * np.eye(base.shape[0]), ord="fro")
        / max(np.linalg.norm(gram, ord="fro"), np.finfo(float).eps)
    )
    flat = np.vstack([response.reshape(1, -1) for response in responses])
    uni_gap, _, _ = universality_gap(flat, normalize_active_rows=True, tolerance=tolerance)
    line_residual = 0.0 if uni_gap is None else float(uni_gap)
    atomic_stack = np.vstack(responses)
    measured_eta = kernel_completeness_gap(atomic_stack, full, tolerance=tolerance)
    base_atomic_gap = kernel_completeness_gap(base, atomic_stack, tolerance=tolerance)

    # Theorem-3 finite-error wiring (R3 §E). The certified mismatch is the
    # two-matrix triangle bound ``eta_true <= eta_obs + b_A + b_F``. Both the
    # atomic and full matrices contribute a subspace-error term formed from a
    # rank-stable certified lower gap ``gamma_safe = observed_gap - eps``; the two
    # terms are SUMMED (``max`` underestimates the triangle bound). A vacuous term
    # (``gamma_safe <= 0`` or ``eta >= 1``) can change kernel rank and must never
    # certify.
    finite_error_vacuous = False
    two_matrix_inputs = (
        atomic_noise_norm,
        atomic_spectral_gap,
        full_noise_norm,
        full_spectral_gap,
    )
    single_inputs = (noise_norm, spectral_gap)
    two_matrix_given = any(value is not None for value in two_matrix_inputs)
    single_given = any(value is not None for value in single_inputs)

    if two_matrix_given and single_given:
        raise ValueError(
            "Use either the deprecated single (noise_norm, spectral_gap) pair or "
            "the two-matrix (atomic_*, full_*) radii and gaps, not both."
        )

    if two_matrix_given:
        if any(value is None for value in two_matrix_inputs):
            raise ValueError(
                "Two-matrix wiring is all-or-none: provide atomic_noise_norm, "
                "atomic_spectral_gap, full_noise_norm and full_spectral_gap together."
            )
        eps_a = float(atomic_noise_norm)  # type: ignore[arg-type]
        gap_a = float(atomic_spectral_gap)  # type: ignore[arg-type]
        eps_f = float(full_noise_norm)  # type: ignore[arg-type]
        gap_f = float(full_spectral_gap)  # type: ignore[arg-type]
        if eps_a < 0 or eps_f < 0 or gap_a <= 0 or gap_f <= 0:
            raise ValueError("Use noise norms >= 0 and spectral gaps > 0.")
        gamma_safe_a = gap_a - eps_a
        gamma_safe_f = gap_f - eps_f
        if gamma_safe_a <= 0.0 or gamma_safe_f <= 0.0:
            # No rank-stable certified lower gap exists for at least one matrix:
            # the allowed perturbation can change kernel rank, so no triangle
            # bound is formable. Saturate eta at (at least) 1 so the descended
            # lower bound collapses and the certificate is inconclusive.
            finite_error_vacuous = True
            eta = float(max(measured_eta, 1.0))
        else:
            b_a = float(min(1.0, eps_a / gamma_safe_a))
            b_f = float(min(1.0, eps_f / gamma_safe_f))
            eta = float(measured_eta + b_a + b_f)  # SUM, never max
    elif single_given:
        if noise_norm is None or spectral_gap is None:
            raise ValueError("Provide both noise_norm and spectral_gap, or neither.")
        # Deprecated single-pair path: a single Wedin bound folded via ``max``.
        wedin_bound = singular_subspace_error_bound(noise_norm, spectral_gap)
        finite_error_vacuous = bool(noise_norm >= spectral_gap)
        eta = float(max(measured_eta, wedin_bound))
    else:
        eta = float(measured_eta)

    # ``eta`` is a projector-difference operator norm, hence bounded by 1; a value
    # at (or beyond) 1 is a vacuous near-isomorphism bound (kernel rank may
    # differ under the allowed perturbation), so it cannot be consumed as a
    # certified branch.
    eta_clipped = bool(eta >= 1.0)
    error_bound_vacuous = bool(eta_clipped or finite_error_vacuous)

    p_atomic = _kernel_projector(atomic_stack, tolerance)
    p_full = _kernel_projector(full, tolerance)
    kernel_dim_atomic = int(round(float(np.trace(p_atomic))))
    kernel_dim_full = int(round(float(np.trace(p_full))))
    quotient_dimension = base.shape[1] - kernel_dim_full
    line_valid = bool(line_residual <= tolerance and base_atomic_gap <= tolerance)
    lower = 0.0
    if line_valid and lambda_min > 0 and eta < 1.0:
        lower = float(np.sqrt(lambda_min) * np.sqrt(max(0.0, 1.0 - eta * eta)))
    certified = bool(
        line_valid
        and not error_bound_vacuous
        and eta < 1.0
        and kernel_dim_atomic == kernel_dim_full
        and quotient_dimension == base.shape[0]
        and lower > tolerance
    )
    if error_bound_vacuous:
        certificate_status = "inconclusive"
    elif certified:
        certificate_status = "certified"
    else:
        certificate_status = "failed"
    return StableCADCertificate(
        kernel_projector_gap=eta,
        base_atomic_kernel_gap=base_atomic_gap,
        quotient_dimension=quotient_dimension,
        target_dimension=base.shape[0],
        descended_lower_singular_bound=lower,
        base_coisometry_defect=coisometry_defect,
        transported_line_residual=line_residual,
        universality_gap=line_residual,
        certified_isomorphism=certified,
        error_bound_vacuous=error_bound_vacuous,
        certificate_status=certificate_status,
    )


def singular_subspace_error_bound(noise_norm: float, spectral_gap: float) -> float:
    """Worst-case Davis--Kahan/Wedin-type sine-angle upper bound.

    The return value is clamped to ``[0, 1]`` via ``min(1, noise/gap)`` and kept
    as a bare float for backward compatibility. When ``noise_norm >= spectral_gap``
    the bound saturates at 1 and is **vacuous**: the singular subspace can rotate
    arbitrarily and rank can change under the allowed perturbation, so no certified
    branch may consume it. Callers that need to act on this (e.g.
    ``stable_cad_certificate``) must test ``noise_norm >= spectral_gap`` explicitly
    and record the ``error_bound_vacuous`` / ``certificate_status='inconclusive'``
    flags rather than reading the clamped ``1.0`` as a valid bound.
    """
    if noise_norm < 0 or spectral_gap <= 0:
        raise ValueError("Use noise_norm >= 0 and spectral_gap > 0.")
    return float(min(1.0, noise_norm / spectral_gap))


def _matrix_inner(left: ComplexArray, right: ComplexArray) -> complex:
    return complex(np.vdot(np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)))


def _orthonormal_matrix_basis(matrices: Iterable[ComplexArray], tolerance: float) -> list[ComplexArray]:
    basis: list[ComplexArray] = []
    for raw in matrices:
        residual = np.asarray(raw, dtype=np.complex128).copy()
        if residual.ndim != 2 or residual.shape[0] != residual.shape[1]:
            raise ValueError("Lie generators must be square matrices.")
        for _ in range(2):
            for element in basis:
                residual -= _matrix_inner(element, residual) * element
        norm = float(np.linalg.norm(residual, ord="fro"))
        if norm > tolerance:
            basis.append(residual / norm)
    return basis


def matrix_lie_closure(
    generators: Sequence[ComplexArray], *, tolerance: float = 1e-10
) -> LieClosureReport:
    """Compute a matrix Lie closure and the first depth at which it stabilizes."""
    if not generators or tolerance <= 0:
        raise ValueError("Use at least one generator and a positive tolerance.")
    base = _orthonormal_matrix_basis(generators, tolerance)
    if not base:
        raise ValueError("All generators vanished at the declared tolerance.")
    basis = list(base)
    dimensions = [len(basis)]
    depth = 1
    matrix_size = basis[0].shape[0]
    maximum_dimension = matrix_size * matrix_size
    while True:
        candidates = list(basis)
        for generator in base:
            for element in basis:
                candidates.append(generator @ element - element @ generator)
        enlarged = _orthonormal_matrix_basis(candidates, tolerance)
        depth += 1
        dimensions.append(len(enlarged))
        if len(enlarged) == len(basis):
            basis = enlarged
            break
        basis = enlarged
        if len(basis) >= maximum_dimension:
            break
    return LieClosureReport(
        dimensions_by_depth=tuple(dimensions),
        closure_depth=depth,
        algebra_dimension=len(basis),
        basis=tuple(np.asarray(element) for element in basis),
    )


def finite_lie_depth_bound(lie_dimension: int, generator_span_dimension: int) -> int:
    """Maximum full-span depth under a finite-dimensional Lie-algebra promise."""
    if lie_dimension < 1 or not 1 <= generator_span_dimension <= lie_dimension:
        raise ValueError("Require 1 <= generator span dimension <= Lie dimension.")
    return int(lie_dimension - generator_span_dimension + 1)


def ramified_unitary_qfi(parameter: float, generator_variance: float) -> RamificationWitness:
    """QFI witness for ``exp(-i u^2 H)`` versus the regular coordinate ``s=u^2``."""
    u = float(parameter)
    variance = float(generator_variance)
    if variance < 0 or not np.isfinite(u) or not np.isfinite(variance):
        raise ValueError("Use a finite parameter and nonnegative generator variance.")
    qfi_raw = 16.0 * u * u * variance
    return RamificationWitness(
        raw_parameter=u,
        qfi_raw=qfi_raw,
        regular_coordinate=u * u,
        qfi_regular=4.0 * variance,
        first_derivative_visible=bool(abs(u) > 0 and qfi_raw > 0),
        second_order_visible=bool(variance > 0),
    )
