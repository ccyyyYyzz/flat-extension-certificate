"""Target-blind dimension selection and local Lorentz reconstruction for CF-RSP.

The module implements two independent selectors that do not receive a target
spatial dimension:

1. the determinant cone of the smallest noncommutative complex event atom
   ``M_2(C)``;
2. the codimension of a generic isolated two-level degeneracy.

Both return three spatial directions. A single modular frequency then produces
one time direction, and the Weyl principal symbol carries a Lorentzian cone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CausalAtomReport:
    matrix_dimension: int
    determinant_degree: int
    self_adjoint_dimension: int
    quadratic: bool
    signature_positive: int
    signature_negative: int
    signature_zero: int
    spatial_dimension: int | None
    spacetime_dimension: int | None


@dataclass(frozen=True)
class NodeDimensionReport:
    parameter_dimension: int
    degeneracy_codimension: int
    generic_degeneracy_dimension: int
    generic_degeneracy_exists: bool
    isolated: bool
    extended: bool


@dataclass(frozen=True)
class DimensionSelectionReport:
    atom_report: CausalAtomReport
    node_reports: tuple[NodeDimensionReport, ...]
    selected_spatial_dimension: int | None
    selected_spacetime_dimension: int | None
    unique: bool
    spectral_estimate: float | None
    spectral_consistent: bool | None


@dataclass(frozen=True)
class LorentzTransformAudit:
    lorentz_matrix: FloatArray
    metric_defect: float
    determinant: float
    time_component: float
    proper: bool
    orthochronous: bool
    allowed: bool


@dataclass(frozen=True)
class WeylConeReport:
    velocity: FloatArray
    contravariant_metric: FloatArray
    signature_positive: int
    signature_negative: int
    signature_zero: int
    lorentzian: bool


@dataclass(frozen=True)
class CoMarginalDrift:
    spacetime_dimension: float
    gauge_beta_canonical: float
    yukawa_beta_canonical: float
    quartic_beta_canonical: float
    normalized_gauge_drift: float
    normalized_yukawa_drift: float
    normalized_quartic_drift: float
    common_dimension_drift: float


@dataclass(frozen=True)
class ConeUniversalityAudit:
    normalized_metrics: tuple[FloatArray, ...]
    pairwise_defects: FloatArray
    maximum_defect: float
    universal: bool


def pauli_basis() -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Return ``(I, sigma_x, sigma_y, sigma_z)``."""
    identity = np.eye(2, dtype=np.complex128)
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return identity, sigma_x, sigma_y, sigma_z


def hermitian_from_coordinates(coordinates: Sequence[float]) -> ComplexArray:
    """Map four real coordinates to a Hermitian ``2 x 2`` matrix."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected four finite real coordinates.")
    return np.asarray(
        sum(
            (values[index] * basis for index, basis in enumerate(pauli_basis())),
            np.zeros((2, 2), dtype=np.complex128),
        ),
        dtype=np.complex128,
    )


def coordinates_from_hermitian(matrix: ComplexArray) -> FloatArray:
    """Recover real Pauli coordinates from a Hermitian ``2 x 2`` matrix."""
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2) or not np.allclose(value, value.conj().T, atol=1e-10):
        raise ValueError("Expected a Hermitian 2 x 2 matrix.")
    coordinates = np.asarray(
        [0.5 * np.trace(basis @ value).real for basis in pauli_basis()],
        dtype=np.float64,
    )
    reconstructed = hermitian_from_coordinates(coordinates)
    if not np.allclose(reconstructed, value, atol=1e-9):
        raise ValueError("Matrix is outside the Hermitian Pauli coordinate chart.")
    return coordinates


def determinant_norm(coordinates: Sequence[float]) -> float:
    """The determinant norm on ``H_2(C)``."""
    return float(np.linalg.det(hermitian_from_coordinates(coordinates)).real)


def determinant_quadratic_form() -> FloatArray:
    """Infer the determinant quadratic form by polarization, without a target metric."""
    dimension = len(pauli_basis())
    basis_vectors = np.eye(dimension, dtype=np.float64)
    gram = np.zeros((dimension, dimension), dtype=np.float64)
    for left in range(dimension):
        gram[left, left] = determinant_norm(basis_vectors[left])
        for right in range(left + 1, dimension):
            mixed = determinant_norm(basis_vectors[left] + basis_vectors[right])
            value = 0.5 * (
                mixed
                - determinant_norm(basis_vectors[left])
                - determinant_norm(basis_vectors[right])
            )
            gram[left, right] = gram[right, left] = value
    return gram


def quadratic_form_signature(
    form: FloatArray, *, tolerance: float = 1e-10
) -> tuple[int, int, int]:
    """Return ``(positive, negative, zero)`` eigenvalue counts."""
    matrix = np.asarray(form, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Quadratic form must be square.")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Quadratic form must be symmetric.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = int(np.sum(eigenvalues > tolerance))
    negative = int(np.sum(eigenvalues < -tolerance))
    zero = int(eigenvalues.size - positive - negative)
    return positive, negative, zero


def causal_atom_report(matrix_dimension: int) -> CausalAtomReport:
    """Audit whether ``M_n(C)`` supplies a primitive quadratic determinant cone.

    The determinant on ``M_n`` has degree ``n``. The first noncommutative factor
    with a quadratic determinant is therefore ``M_2(C)``. Its self-adjoint part
    is audited directly rather than assigning a desired signature.
    """
    if matrix_dimension < 1:
        raise ValueError("matrix_dimension must be positive.")
    quadratic = matrix_dimension == 2
    positive = negative = zero = 0
    spatial: int | None = None
    spacetime: int | None = None
    if quadratic:
        positive, negative, zero = quadratic_form_signature(determinant_quadratic_form())
        if positive == 1 and zero == 0:
            spatial = negative
            spacetime = positive + negative
    return CausalAtomReport(
        matrix_dimension=matrix_dimension,
        determinant_degree=matrix_dimension,
        self_adjoint_dimension=matrix_dimension * matrix_dimension,
        quadratic=quadratic,
        signature_positive=positive,
        signature_negative=negative,
        signature_zero=zero,
        spatial_dimension=spatial,
        spacetime_dimension=spacetime,
    )


def select_minimal_quadratic_causal_atom(
    candidates: Iterable[int] = range(1, 9),
) -> CausalAtomReport:
    """Select the smallest noncommutative matrix factor with a quadratic norm."""
    reports = [causal_atom_report(int(dimension)) for dimension in candidates]
    admissible = [report for report in reports if report.matrix_dimension > 1 and report.quadratic]
    if not admissible:
        raise ValueError("No noncommutative quadratic causal atom in candidates.")
    return min(admissible, key=lambda report: report.matrix_dimension)


def two_level_degeneracy_codimension() -> int:
    """Real codimension of an unconstrained two-level Hermitian degeneracy.

    A scalar part shifts both eigenvalues and is irrelevant to degeneracy. The
    remaining traceless Hermitian space is inferred from the event atom.
    """
    return len(pauli_basis()) - 1


def generic_two_level_node_report(parameter_dimension: int) -> NodeDimensionReport:
    """Expected dimension of the generic degeneracy set by transversality."""
    if parameter_dimension < 1:
        raise ValueError("parameter_dimension must be positive.")
    codimension = two_level_degeneracy_codimension()
    degeneracy_dimension = parameter_dimension - codimension
    exists = degeneracy_dimension >= 0
    return NodeDimensionReport(
        parameter_dimension=parameter_dimension,
        degeneracy_codimension=codimension,
        generic_degeneracy_dimension=degeneracy_dimension,
        generic_degeneracy_exists=exists,
        isolated=degeneracy_dimension == 0,
        extended=degeneracy_dimension > 0,
    )


def select_isolated_point_node_dimension(
    candidates: Iterable[int] = range(1, 9),
) -> tuple[int | None, tuple[NodeDimensionReport, ...]]:
    """Select dimensions where generic two-level nodes are isolated points."""
    reports = tuple(generic_two_level_node_report(int(dimension)) for dimension in candidates)
    isolated = [report.parameter_dimension for report in reports if report.isolated]
    return (isolated[0] if len(isolated) == 1 else None), reports


def derive_dimension_selection(
    *,
    matrix_candidates: Iterable[int] = range(1, 9),
    spatial_candidates: Iterable[int] = range(1, 9),
    spectral_estimate: float | None = None,
    spectral_tolerance: float = 0.25,
) -> DimensionSelectionReport:
    """Combine two target-blind selectors and optionally cross-check a spectrum."""
    if spectral_tolerance <= 0:
        raise ValueError("spectral_tolerance must be positive.")
    atom = select_minimal_quadratic_causal_atom(matrix_candidates)
    node_dimension, node_reports = select_isolated_point_node_dimension(spatial_candidates)
    selected = (
        atom.spatial_dimension
        if atom.spatial_dimension is not None and atom.spatial_dimension == node_dimension
        else None
    )
    spectral_consistent: bool | None = None
    if spectral_estimate is not None:
        if not np.isfinite(spectral_estimate):
            raise ValueError("spectral_estimate must be finite.")
        spectral_consistent = bool(
            selected is not None and abs(float(spectral_estimate) - selected) <= spectral_tolerance
        )
    return DimensionSelectionReport(
        atom_report=atom,
        node_reports=node_reports,
        selected_spatial_dimension=selected,
        selected_spacetime_dimension=(selected + 1 if selected is not None else None),
        unique=selected is not None,
        spectral_estimate=(float(spectral_estimate) if spectral_estimate is not None else None),
        spectral_consistent=spectral_consistent,
    )


def minkowski_metric() -> FloatArray:
    """Return the reference Minkowski form ``diag(1, -1, -1, -1)``.

    This hardcoded signature is a fixed reference used only to verify an output
    against it (for example checking ``sl2c_to_lorentz`` in ``audit_sl2c_lorentz``);
    it is never assumed by the blind estimator or the solder pipeline.
    """
    return np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.float64)


def normalize_sl2c(matrix: ComplexArray) -> ComplexArray:
    """Normalize an invertible ``2 x 2`` matrix to determinant one."""
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2):
        raise ValueError("Expected a 2 x 2 matrix.")
    determinant = np.linalg.det(value)
    if abs(determinant) <= np.finfo(float).eps:
        raise ValueError("Matrix must be invertible.")
    normalized = value / np.sqrt(determinant)
    if not np.isclose(np.linalg.det(normalized), 1.0, atol=1e-9):
        normalized = -normalized
    return np.asarray(normalized, dtype=np.complex128)


def sl2c_to_lorentz(matrix: ComplexArray) -> FloatArray:
    """Map ``SL(2,C)`` to the proper orthochronous Lorentz representation."""
    spin = normalize_sl2c(matrix)
    basis = pauli_basis()
    lorentz = np.empty((4, 4), dtype=np.float64)
    for mu, sigma_mu in enumerate(basis):
        for nu, sigma_nu in enumerate(basis):
            value = 0.5 * np.trace(sigma_mu @ spin @ sigma_nu @ spin.conj().T)
            if abs(value.imag) > 1e-8:
                raise ValueError("SL(2,C) action did not produce a real Lorentz matrix.")
            lorentz[mu, nu] = value.real
    return lorentz


def audit_sl2c_lorentz(
    matrix: ComplexArray, *, tolerance: float = 1e-9
) -> LorentzTransformAudit:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    lorentz = sl2c_to_lorentz(matrix)
    eta = minkowski_metric()
    defect = float(
        np.linalg.norm(lorentz.T @ eta @ lorentz - eta, ord="fro")
        / np.linalg.norm(eta, ord="fro")
    )
    determinant = float(np.linalg.det(lorentz))
    time_component = float(lorentz[0, 0])
    proper = bool(abs(determinant - 1.0) <= tolerance)
    orthochronous = bool(time_component >= 1.0 - tolerance)
    return LorentzTransformAudit(
        lorentz_matrix=lorentz,
        metric_defect=defect,
        determinant=determinant,
        time_component=time_component,
        proper=proper,
        orthochronous=orthochronous,
        allowed=bool(defect <= tolerance and proper and orthochronous),
    )


def sl2c_boost(rapidity: float, axis: int = 2) -> ComplexArray:
    """Return the spin representation of a boost along Pauli axis 0, 1, or 2."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    sigma = pauli_basis()[axis + 1]
    half = 0.5 * float(rapidity)
    return np.asarray(np.cosh(half) * np.eye(2) + np.sinh(half) * sigma, dtype=np.complex128)


def weyl_cone_metric(
    velocity: FloatArray, *, clock_scale: float = 1.0, tolerance: float = 1e-10
) -> WeylConeReport:
    """Return the principal Lorentz metric of a transverse Weyl node.

    For ``D(omega,k)=omega/clock_scale I - sigma^a v_ai k_i``, the
    characteristic polynomial is ``omega^2/clock_scale^2-k^T v^T v k``.
    """
    matrix = np.asarray(velocity, dtype=np.float64)
    spatial_dimension = two_level_degeneracy_codimension()
    if matrix.shape != (spatial_dimension, spatial_dimension):
        raise ValueError("velocity must be a square matrix matching the atom codimension.")
    if not np.all(np.isfinite(matrix)) or clock_scale <= 0 or tolerance <= 0:
        raise ValueError("Invalid velocity/clock data.")
    if abs(np.linalg.det(matrix)) <= tolerance:
        raise ValueError("A transverse Weyl node requires a full-rank velocity matrix.")
    spatial_metric = matrix.T @ matrix
    metric = np.zeros((spatial_dimension + 1, spatial_dimension + 1), dtype=np.float64)
    metric[0, 0] = 1.0 / (clock_scale * clock_scale)
    metric[1:, 1:] = -spatial_metric
    positive, negative, zero = quadratic_form_signature(metric, tolerance=tolerance)
    return WeylConeReport(
        velocity=matrix,
        contravariant_metric=metric,
        signature_positive=positive,
        signature_negative=negative,
        signature_zero=zero,
        lorentzian=bool(positive == 1 and negative == spatial_dimension and zero == 0),
    )


def principal_symbol_determinant(
    frequency: float, momentum: Sequence[float], velocity: FloatArray, *, clock_scale: float = 1.0
) -> float:
    """Direct determinant of the two-level Weyl principal symbol."""
    momentum_vector = np.asarray(momentum, dtype=np.float64)
    spatial_dimension = two_level_degeneracy_codimension()
    if momentum_vector.shape != (spatial_dimension,):
        raise ValueError("Momentum dimension does not match the event atom.")
    velocity_matrix = np.asarray(velocity, dtype=np.float64)
    if velocity_matrix.shape != (spatial_dimension, spatial_dimension):
        raise ValueError("Velocity has the wrong shape.")
    symbol = (float(frequency) / clock_scale) * pauli_basis()[0]
    spatial = velocity_matrix @ momentum_vector
    for index, coefficient in enumerate(spatial):
        symbol -= coefficient * pauli_basis()[index + 1]
    return float(np.linalg.det(symbol).real)


def co_marginal_canonical_drift(
    spacetime_dimension: float,
    *,
    gauge_coupling: float = 1.0,
    yukawa_coupling: float = 1.0,
    quartic_coupling: float = 1.0,
) -> CoMarginalDrift:
    """Canonical RG drift caused only by an effective spacetime dimension.

    Loop/anomalous contributions are intentionally excluded. After subtracting
    four-dimensional loop terms, a pure dimension drift obeys
    ``2 beta_g/g = 2 beta_y/y = beta_lambda/lambda = D-4``.
    """
    dimension = float(spacetime_dimension)
    couplings = np.asarray(
        [gauge_coupling, yukawa_coupling, quartic_coupling], dtype=np.float64
    )
    if not np.all(np.isfinite(couplings)) or np.any(np.abs(couplings) <= 0.0):
        raise ValueError("Couplings must be finite and nonzero.")
    drift = dimension - 4.0
    beta_g = 0.5 * drift * float(gauge_coupling)
    beta_y = 0.5 * drift * float(yukawa_coupling)
    beta_lam = drift * float(quartic_coupling)
    return CoMarginalDrift(
        spacetime_dimension=dimension,
        gauge_beta_canonical=beta_g,
        yukawa_beta_canonical=beta_y,
        quartic_beta_canonical=beta_lam,
        normalized_gauge_drift=2.0 * beta_g / float(gauge_coupling),
        normalized_yukawa_drift=2.0 * beta_y / float(yukawa_coupling),
        normalized_quartic_drift=beta_lam / float(quartic_coupling),
        common_dimension_drift=drift,
    )


def co_marginal_sum_rule_residual(
    *,
    gauge_beta_residual: float,
    yukawa_beta_residual: float,
    quartic_beta_residual: float,
    gauge_coupling: float,
    yukawa_coupling: float,
    quartic_coupling: float,
) -> float:
    """RMS violation of the dimension-drift ratio after loop subtraction."""
    values = np.asarray(
        [
            2.0 * gauge_beta_residual / gauge_coupling,
            2.0 * yukawa_beta_residual / yukawa_coupling,
            quartic_beta_residual / quartic_coupling,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Residual beta/coupling ratios must be finite.")
    return float(np.sqrt(np.mean(np.square(values - np.mean(values)))))


def _conformal_normalize_metric(metric: FloatArray) -> FloatArray:
    value = np.asarray(metric, dtype=np.float64)
    if value.shape != (4, 4) or not np.allclose(value, value.T, atol=1e-10):
        raise ValueError("Each cone metric must be a symmetric 4 x 4 matrix.")
    determinant = float(np.linalg.det(value))
    if abs(determinant) <= np.finfo(float).eps:
        raise ValueError("Cone metrics must be nondegenerate.")
    scale = abs(determinant) ** 0.25
    normalized = value / scale
    if normalized[0, 0] < 0:
        normalized = -normalized
    return normalized


def audit_cone_universality(
    metrics: Sequence[FloatArray], *, tolerance: float = 1e-6
) -> ConeUniversalityAudit:
    """Test whether species share one cone up to a positive conformal scale."""
    if len(metrics) < 2:
        raise ValueError("At least two cone metrics are required.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    normalized = tuple(_conformal_normalize_metric(metric) for metric in metrics)
    pairwise = np.zeros((len(normalized), len(normalized)), dtype=np.float64)
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            denominator = max(
                float(np.linalg.norm(normalized[left], ord="fro")), np.finfo(float).eps
            )
            defect = float(
                np.linalg.norm(normalized[left] - normalized[right], ord="fro")
                / denominator
            )
            pairwise[left, right] = pairwise[right, left] = defect
    maximum = float(np.max(pairwise))
    return ConeUniversalityAudit(
        normalized_metrics=normalized,
        pairwise_defects=pairwise,
        maximum_defect=maximum,
        universal=bool(maximum <= tolerance),
    )


def sl2c_holonomy(transitions: Sequence[ComplexArray]) -> ComplexArray:
    """Ordered spin holonomy around a closed overlap loop."""
    if not transitions:
        raise ValueError("At least one transition is required.")
    result = np.eye(2, dtype=np.complex128)
    for transition in transitions:
        result = normalize_sl2c(transition) @ result
    return normalize_sl2c(result)


def flat_spin_holonomy_defect(transitions: Sequence[ComplexArray]) -> float:
    """Distance of a loop holonomy from the spin-center ``{+I,-I}``."""
    holonomy = sl2c_holonomy(transitions)
    identity = np.eye(2, dtype=np.complex128)
    denominator = np.linalg.norm(identity, ord="fro")
    return float(
        min(
            np.linalg.norm(holonomy - identity, ord="fro"),
            np.linalg.norm(holonomy + identity, ord="fro"),
        )
        / denominator
    )
