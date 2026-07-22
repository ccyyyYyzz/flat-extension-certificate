"""Bridge currents, passive response pencils, common-cone and quotient audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResidueNormalization:
    whitening: ComplexArray
    normalized_frequency: ComplexArray
    normalized_spatial: tuple[ComplexArray, ...]
    frequency_defect: float
    condition_number: float


@dataclass(frozen=True)
class ResponsePencil:
    tilt: FloatArray
    traceless_map: FloatArray
    spatial_metric: FloatArray
    solder_matrix: FloatArray
    characteristic_form: FloatArray
    spatial_rank: int
    extended_rank: int
    spatial_kernel: FloatArray
    response_kernel: FloatArray


@dataclass(frozen=True)
class CommonConeAudit:
    connected: bool
    components: tuple[tuple[int, ...], ...]
    edge_defects: FloatArray
    maximum_edge_defect: float
    universal: bool
    representative_tilt: FloatArray | None
    representative_spatial_metric: FloatArray | None


@dataclass(frozen=True)
class OperationalQuotientAudit:
    ambient_tangent_dimension: int
    spatial_quotient_dimension: int
    spacetime_quotient_dimension: int
    spatial_kernel_dimension: int
    response_kernel_dimension: int
    protocol_through_solder_defect: float | None
    solder_through_protocol_defect: float | None
    minimal_faithful: bool | None
    response_kernel: FloatArray


@dataclass(frozen=True)
class DirichletAudit:
    form: FloatArray
    symmetry_defect: float
    minimum_eigenvalue: float
    positive_semidefinite: bool


@dataclass(frozen=True)
class PrincipalIntertwiningAudit:
    invertible: bool
    smallest_singular_value: float
    unitary_defect: float
    intertwining_defect: float
    allowed: bool


@dataclass(frozen=True)
class ConeConsensusFlow:
    times: FloatArray
    defects: FloatArray
    algebraic_connectivity: float
    fitted_decay_rate: float


def _square(matrix: ComplexArray, name: str) -> ComplexArray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite square matrix.")
    return value


def _hermitian(matrix: ComplexArray, name: str, tolerance: float = 1e-10) -> ComplexArray:
    value = _square(matrix, name)
    if not np.allclose(value, value.conj().T, atol=tolerance):
        raise ValueError(f"{name} must be Hermitian.")
    return value


def _positive_function(matrix: ComplexArray, function: Callable[[FloatArray], FloatArray]) -> ComplexArray:
    value = _hermitian(matrix, "positive matrix")
    values, vectors = np.linalg.eigh(value)
    if float(np.min(values).real) <= 0.0:
        raise ValueError("Matrix must be positive definite.")
    return np.asarray((vectors * function(values.real)) @ vectors.conj().T, dtype=np.complex128)


def positive_inverse_sqrt(matrix: ComplexArray) -> ComplexArray:
    return _positive_function(matrix, lambda values: values ** -0.5)


def matrix_log_positive(matrix: ComplexArray) -> ComplexArray:
    return _positive_function(matrix, np.log)


def normalize_positive_frequency(
    frequency_matrix: ComplexArray,
    spatial_coefficients: Sequence[ComplexArray],
    *,
    tolerance: float = 1e-10,
) -> ResidueNormalization:
    frequency = _hermitian(frequency_matrix, "frequency matrix")
    values = np.linalg.eigvalsh(frequency).real
    if tolerance <= 0 or float(np.min(values)) <= tolerance:
        raise ValueError("frequency matrix must be positive definite.")
    coefficients = tuple(_hermitian(x, "spatial coefficient") for x in spatial_coefficients)
    if any(x.shape != frequency.shape for x in coefficients):
        raise ValueError("Pencil coefficient sizes do not match.")
    whitening = positive_inverse_sqrt(frequency)
    normalized_frequency = whitening @ frequency @ whitening
    normalized_spatial = tuple(whitening @ x @ whitening for x in coefficients)
    identity = np.eye(frequency.shape[0], dtype=np.complex128)
    defect = float(np.linalg.norm(normalized_frequency - identity, ord="fro") / np.linalg.norm(identity))
    return ResidueNormalization(
        whitening,
        np.asarray(normalized_frequency),
        tuple(np.asarray(x) for x in normalized_spatial),
        defect,
        float(np.max(values) / np.min(values)),
    )


def pauli_basis() -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    return (
        np.eye(2, dtype=np.complex128),
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    )


def hermitian_two_level_coordinates(matrix: ComplexArray) -> FloatArray:
    value = _hermitian(matrix, "two-level coefficient")
    if value.shape != (2, 2):
        raise ValueError("Expected a 2 x 2 coefficient.")
    return np.asarray([0.5 * np.trace(basis @ value).real for basis in pauli_basis()])


def _rank(matrix: FloatArray, tolerance: float) -> int:
    values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return 0 if not values.size else int(np.sum(values > tolerance * max(float(values[0]), 1.0)))


def _nullspace(matrix: FloatArray, tolerance: float) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(value, full_matrices=True)
    rank = 0 if not singular.size else int(np.sum(singular > tolerance * max(float(singular[0]), 1.0)))
    return np.asarray(vh[rank:].T)


def response_pencil(
    normalized_spatial_coefficients: Sequence[ComplexArray],
    characteristic_form: FloatArray,
    *,
    tolerance: float = 1e-10,
) -> ResponsePencil:
    """Assemble the response pencil and pull the target form back along the solder.

    ``characteristic_form`` is the quadratic form on the four extended target
    coordinates ``(t, x, y, z)``; the form must be supplied from the blind
    two-sheet estimator or the H_2(C) determinant polarization; it is never
    assumed. No Minkowski signature is baked into this routine.
    """
    coefficients = tuple(normalized_spatial_coefficients)
    if tolerance <= 0 or not coefficients:
        raise ValueError("Use a positive tolerance and at least one coefficient.")
    eta = np.asarray(characteristic_form, dtype=np.float64)
    if eta.shape != (4, 4) or not np.all(np.isfinite(eta)):
        raise ValueError("characteristic_form must be a finite 4 x 4 target form.")
    coordinates = np.column_stack([hermitian_two_level_coordinates(x) for x in coefficients])
    tilt, traceless = np.asarray(coordinates[0]), np.asarray(coordinates[1:])
    metric = traceless.T @ traceless
    solder = np.zeros((4, len(coefficients) + 1), dtype=np.float64)
    solder[0, 0], solder[0, 1:], solder[1:, 1:] = 1.0, -tilt, -traceless
    return ResponsePencil(
        tilt,
        traceless,
        np.asarray(metric),
        solder,
        solder.T @ eta @ solder,
        _rank(traceless, tolerance),
        _rank(solder, tolerance),
        _nullspace(traceless, tolerance),
        _nullspace(solder, tolerance),
    )


def characteristic_value(pencil: ResponsePencil, frequency: float, tangent: Sequence[float]) -> float:
    q = np.asarray(tangent, dtype=np.float64)
    if q.shape != pencil.tilt.shape:
        raise ValueError("Tangent dimension mismatch.")
    return float((frequency - pencil.tilt @ q) ** 2 - q @ pencil.spatial_metric @ q)


def characteristic_form_from_tilt_metric(tilt: Sequence[float], metric: FloatArray) -> FloatArray:
    c, h = np.asarray(tilt, dtype=np.float64), np.asarray(metric, dtype=np.float64)
    if h.shape != (c.size, c.size) or not np.allclose(h, h.T):
        raise ValueError("Tilt and metric do not match.")
    form = np.empty((c.size + 1, c.size + 1), dtype=np.float64)
    form[0, 0] = 1.0
    form[0, 1:] = form[1:, 0] = -c
    form[1:, 1:] = np.outer(c, c) - h
    return form


def _components(count: int, edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(count)]
    for left, right in edges:
        if not (0 <= left < count and 0 <= right < count) or left == right:
            raise ValueError("Invalid species edge.")
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen, result = set(range(count)), []
    while unseen:
        stack, reached = [min(unseen)], set()
        while stack:
            node = stack.pop()
            if node not in reached:
                reached.add(node)
                stack.extend(adjacency[node] - reached)
        unseen -= reached
        result.append(tuple(sorted(reached)))
    return tuple(result)


def common_cone_soldering_audit(
    pencils: Sequence[ResponsePencil], edges: Sequence[tuple[int, int]], *, tolerance: float = 1e-6
) -> CommonConeAudit:
    if len(pencils) < 2 or tolerance <= 0:
        raise ValueError("Use at least two species and a positive tolerance.")
    if len({pencil.tilt.size for pencil in pencils}) != 1:
        raise ValueError("Species tangent dimensions do not match.")
    components = _components(len(pencils), edges)
    defects = np.full((len(pencils), len(pencils)), np.nan)
    maximum = 0.0
    for left, right in edges:
        first, second = pencils[left].characteristic_form, pencils[right].characteristic_form
        defect = float(np.linalg.norm(first - second) / max(np.linalg.norm(first), np.finfo(float).eps))
        defects[left, right] = defects[right, left] = defect
        maximum = max(maximum, defect)
    universal = bool(len(components) == 1 and bool(edges) and maximum <= tolerance)
    return CommonConeAudit(
        len(components) == 1,
        components,
        defects,
        maximum,
        universal,
        np.mean(np.stack([p.tilt for p in pencils]), axis=0) if universal else None,
        np.mean(np.stack([p.spatial_metric for p in pencils]), axis=0) if universal else None,
    )


def projective_null_transfer_defect(
    source: ResponsePencil, target: ResponsePencil, tangents: FloatArray
) -> float:
    samples = np.asarray(tangents, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != source.tilt.size or target.tilt.size != source.tilt.size:
        raise ValueError("Invalid null-transfer samples.")
    residuals = []
    for q in samples:
        radius = np.sqrt(max(0.0, float(q @ source.spatial_metric @ q)))
        for sign in (-1.0, 1.0):
            omega = float(source.tilt @ q + sign * radius)
            scale = max(omega**2 + float(q @ target.spatial_metric @ q), 1.0)
            residuals.append(abs(characteristic_value(target, omega, q)) / scale)
    return float(max(residuals, default=0.0))


def _row_projection(matrix: FloatArray, tolerance: float) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(value, full_matrices=False)
    rank = 0 if not singular.size else int(np.sum(singular > tolerance * max(float(singular[0]), 1.0)))
    basis = vh[:rank].T
    return np.zeros((value.shape[1], value.shape[1])) if rank == 0 else basis @ basis.T


def operational_quotient_audit(
    pencil: ResponsePencil, protocol_jacobian: FloatArray | None = None, *, tolerance: float = 1e-8
) -> OperationalQuotientAudit:
    through_solder = through_protocol = None
    faithful = None
    if protocol_jacobian is not None:
        responses = np.asarray(protocol_jacobian, dtype=np.float64)
        if responses.ndim != 2 or responses.shape[1] != pencil.solder_matrix.shape[1]:
            raise ValueError("Protocol tangent dimension mismatch.")
        ps, pr = _row_projection(pencil.solder_matrix, tolerance), _row_projection(responses, tolerance)
        identity = np.eye(responses.shape[1])
        through_solder = float(np.linalg.norm(responses @ (identity - ps)) / max(np.linalg.norm(responses), np.finfo(float).eps))
        through_protocol = float(np.linalg.norm(pencil.solder_matrix @ (identity - pr)) / max(np.linalg.norm(pencil.solder_matrix), np.finfo(float).eps))
        faithful = bool(through_solder <= tolerance and through_protocol <= tolerance)
    return OperationalQuotientAudit(
        pencil.tilt.size + 1,
        pencil.spatial_rank,
        pencil.extended_rank,
        pencil.spatial_kernel.shape[1],
        pencil.response_kernel.shape[1],
        through_solder,
        through_protocol,
        faithful,
        pencil.response_kernel,
    )


def modular_bridge_response_generator(
    state: ComplexArray,
    bridge_generators: Sequence[ComplexArray],
    controls: Sequence[float],
    *,
    traceless_gauge: bool = True,
) -> ComplexArray:
    rho = _hermitian(state, "state")
    if not np.isclose(np.trace(rho), 1.0) or float(np.min(np.linalg.eigvalsh(rho)).real) <= 0.0:
        raise ValueError("State must be faithful and normalized.")
    generators = tuple(_hermitian(x, "bridge generator") for x in bridge_generators)
    coefficients = np.asarray(controls, dtype=np.float64)
    if coefficients.shape != (len(generators),) or any(x.shape != rho.shape for x in generators):
        raise ValueError("Bridge controls do not match.")
    result = -matrix_log_positive(rho) + sum(
        (coefficients[i] * generators[i] for i in range(len(generators))), np.zeros_like(rho)
    )
    if traceless_gauge:
        result -= np.trace(result) / result.shape[0] * np.eye(result.shape[0])
    return np.asarray(result)


def bridge_current_action(generator: ComplexArray, observable: ComplexArray) -> ComplexArray:
    h, a = _hermitian(generator, "bridge generator"), _hermitian(observable, "observable")
    if h.shape != a.shape:
        raise ValueError("Current dimensions do not match.")
    return np.asarray(1j * (h @ a - a @ h))


def kms_inner(state: ComplexArray, left: ComplexArray, right: ComplexArray) -> complex:
    rho, a, b = _hermitian(state, "state"), _square(left, "left"), _square(right, "right")
    if a.shape != rho.shape or b.shape != rho.shape or not np.isclose(np.trace(rho), 1.0):
        raise ValueError("KMS data do not match.")
    root = _positive_function(rho, np.sqrt)
    return complex(np.trace(root @ a.conj().T @ root @ b))


def kms_dirichlet_audit(
    state: ComplexArray,
    observables: Sequence[ComplexArray],
    generator: Callable[[ComplexArray], ComplexArray],
    *,
    tolerance: float = 1e-9,
) -> DirichletAudit:
    arrays = tuple(_square(x, "observable") for x in observables)
    if not arrays:
        raise ValueError("Use at least one observable.")
    form = np.asarray(
        [[-kms_inner(state, a, np.asarray(generator(b))).real for b in arrays] for a in arrays]
    )
    symmetric = (form + form.T) / 2.0
    defect = float(np.linalg.norm(form - form.T) / max(np.linalg.norm(form), np.finfo(float).eps))
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    return DirichletAudit(symmetric, defect, minimum, bool(defect <= tolerance and minimum >= -tolerance))


def standard_liouvillean(generator: ComplexArray) -> ComplexArray:
    h = _hermitian(generator, "response generator")
    identity = np.eye(h.shape[0], dtype=np.complex128)
    return np.kron(identity, h) - np.kron(h.T, identity)


def passive_response_resolvent(
    state: ComplexArray,
    generator: ComplexArray,
    source: ComplexArray,
    detector: ComplexArray,
    spectral_parameter: complex,
    *,
    pole_tolerance: float = 1e-12,
) -> complex:
    if abs(complex(spectral_parameter).imag) <= pole_tolerance:
        raise ValueError("Choose a spectral parameter off the real poles.")
    h, a, b = _hermitian(generator, "generator"), _square(source, "source"), _square(detector, "detector")
    if a.shape != h.shape or b.shape != h.shape:
        raise ValueError("Response dimensions do not match.")
    liouvillean = standard_liouvillean(h)
    response = np.linalg.solve(
        complex(spectral_parameter) * np.eye(liouvillean.shape[0]) - liouvillean,
        a.reshape(-1, order="F"),
    ).reshape(a.shape, order="F")
    return kms_inner(state, b, response)


def bridge_current_bilinear(
    state: ComplexArray, generator: ComplexArray, source: ComplexArray, detector: ComplexArray
) -> complex:
    return kms_inner(state, detector, bridge_current_action(generator, source))


def audit_principal_intertwiner(
    source_coefficients: Sequence[ComplexArray],
    target_coefficients: Sequence[ComplexArray],
    converter: ComplexArray,
    *,
    require_unitary: bool = True,
    tolerance: float = 1e-8,
) -> PrincipalIntertwiningAudit:
    source = tuple(_hermitian(x, "source coefficient") for x in source_coefficients)
    target = tuple(_hermitian(x, "target coefficient") for x in target_coefficients)
    if not source or len(source) != len(target):
        raise ValueError("Coefficient families do not match.")
    dimension = source[0].shape[0]
    c = np.asarray(converter, dtype=np.complex128)
    if c.shape != (dimension, dimension) or any(x.shape != c.shape for x in (*source, *target)):
        raise ValueError("Converter dimensions do not match.")
    singular = np.linalg.svd(c, compute_uv=False)
    smallest = float(singular[-1])
    identity = np.eye(dimension)
    unitary_defect = float(np.linalg.norm(c.conj().T @ c - identity) / np.linalg.norm(identity))
    defects = []
    for left, right in zip(source, target):
        denominator = max(np.linalg.norm(c) * max(np.linalg.norm(left), np.linalg.norm(right)), np.finfo(float).eps)
        defects.append(float(np.linalg.norm(c @ left - right @ c) / denominator))
    defect = max(defects, default=0.0)
    allowed = bool(smallest > tolerance and defect <= tolerance and (not require_unitary or unitary_defect <= tolerance))
    return PrincipalIntertwiningAudit(smallest > tolerance, smallest, unitary_defect, defect, allowed)


def haar_averaged_inner_product(
    representation_samples: Sequence[FloatArray], *, base_metric: FloatArray | None = None
) -> FloatArray:
    matrices = [np.asarray(x, dtype=np.float64) for x in representation_samples]
    if not matrices:
        raise ValueError("Use at least one automorphism sample.")
    dimension = matrices[0].shape[0]
    if any(x.shape != (dimension, dimension) for x in matrices):
        raise ValueError("Representation dimensions do not match.")
    metric = np.eye(dimension) if base_metric is None else np.asarray(base_metric, dtype=np.float64)
    averaged = sum((x.T @ metric @ x for x in matrices), np.zeros_like(metric)) / len(matrices)
    return (averaged + averaged.T) / 2.0


def cone_consensus_flow(
    characteristic_forms: Sequence[FloatArray], edges: Sequence[tuple[int, int]], times: Iterable[float]
) -> ConeConsensusFlow:
    forms = [np.asarray(x, dtype=np.float64) for x in characteristic_forms]
    sample_times = np.asarray(list(times), dtype=np.float64)
    if len(forms) < 2 or any(x.shape != forms[0].shape for x in forms) or sample_times.size < 2:
        raise ValueError("Invalid consensus data.")
    count = len(forms)
    adjacency = np.zeros((count, count))
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    positive = values[values > 1e-12]
    gap = float(np.min(positive)) if positive.size else 0.0
    stacked = np.stack([x.reshape(-1) for x in forms])
    defects = []
    for time in sample_times:
        propagator = (vectors * np.exp(-time * values)) @ vectors.T
        evolved = propagator @ stacked
        defects.append(float(np.linalg.norm(evolved - np.mean(evolved, axis=0, keepdims=True))))
    defects_array = np.asarray(defects)
    valid = defects_array > max(np.finfo(float).eps, defects_array[0] * 1e-12)
    fitted = 0.0
    if np.sum(valid) >= 2:
        fitted = float(-np.polyfit(sample_times[valid], np.log(defects_array[valid]), 1)[0])
    return ConeConsensusFlow(sample_times, defects_array, gap, fitted)
