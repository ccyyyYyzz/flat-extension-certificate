"""Comparison methods for the C7 blind benchmark, on the same challenge data.

Four baselines are run against the frozen flat-extension certificate of
``cyz_m.blind_benchmark`` on identical challenge instances and the same
simulator backend:

    (i)   marginal QPT              -- atomic circuits only; must FAIL to detect
                                       any dark link (Theorem 1: the single-time
                                       IC marginal channels are dark-invariant).
    (ii)  restricted PTT            -- unitary-only two-slot controls (White et
                                       al. multitime witnesses from unitary
                                       sequences); detects, but scans 6x6 IC
                                       preparations x 9 Pauli pairs = 324 settings.
    (iii) full PTT                  -- the process-tensor-IC operation basis on
                                       the 2-slot instance: 16x16 = 256 = d^{4k}
                                       basis operations (the exponential ceiling
                                       of Theorem T2'); detects.
    (iv)  structured low-memory     -- the O6 strong structured comparator: fit a
                                       bond-dim-2 comb model (the true class) by
                                       least squares and test the link magnitude;
                                       detects at the same 9-setting cost as the
                                       certificate.

Every baseline keys its verdict on its *off-null* statistics (those whose
noiseless dark=False value is 0), where depolarizing noise stays unbiased and
the binomial standard error is well conditioned -- avoiding false positives from
noise-induced contrast loss on near-unit correlators.  Each returns a verdict
dict and a :class:`cyz_m.blind_benchmark.ResourceLedger`.

The ``runner`` argument is a callable ``run(circuits, shots) -> list[counts]``
supplied by the experiment (wrapping qiskit-aer + a noise model), so this module
does not import qiskit for its analysis math.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from cyz_m.blind_benchmark import (
    CircuitFamily, ChallengeInstance, FrozenEstimator, ResourceLedger, Setting,
    PREP_ORDER, PAULI_1Q, PAULI_PAIRS,
    pair_correlator_from_counts, single_expectation_from_counts, norm_ppf,
)

RunnerT = Callable[[List["object"], int], List[Dict[str, int]]]

# --------------------------------------------------------------------------- #
# numpy noiseless comb -- the dark=False (null) predictions                     #
# --------------------------------------------------------------------------- #
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}
_k0 = np.array([1, 0], complex)
_k1 = np.array([0, 1], complex)


def _dm(psi):
    psi = np.asarray(psi, complex)
    return np.outer(psi, psi.conj())


_PREP_DM = {
    "|0>": _dm(_k0), "|1>": _dm(_k1),
    "|+>": _dm((_k0 + _k1) / np.sqrt(2)), "|->": _dm((_k0 - _k1) / np.sqrt(2)),
    "|+i>": _dm((_k0 + 1j * _k1) / np.sqrt(2)), "|-i>": _dm((_k0 - 1j * _k1) / np.sqrt(2)),
}
_HALF = _I2 / 2.0
_EPLUS = _dm((_k0 + _k1) / np.sqrt(2))


def _embed1(op, pos, n=3):
    ops = [_I2] * n
    ops[pos] = op
    out = ops[0]
    for k in range(1, n):
        out = np.kron(out, ops[k])
    return out


def _cz(a, b, n=3):
    dim = 2 ** n
    diag = np.ones(dim, complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] and bits[b]:
            diag[idx] = -1
    return np.diag(diag)


def _ptrace(rho, keep, dims=(2, 2, 2)):
    n = len(dims)
    rho = rho.reshape(list(dims) + list(dims))
    keep = sorted(keep)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for ax in [i for i in range(n) if i not in keep]:
        col[ax] = row[ax]
    rho = np.einsum(rho, row + col, [row[i] for i in keep] + [col[i] for i in keep])
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


_U1 = _cz(0, 1)   # CZ_{S1,E}
_U2 = _cz(2, 1)   # CZ_{S2,E}


def null_pair_correlator(prep_s1: str, prep_s2: str, p: str, q: str) -> float:
    """Noiseless dark=False <P_S1 Q_S2> for the depth-2 comb."""
    rin = np.kron(np.kron(_PREP_DM[prep_s1], _EPLUS), _PREP_DM[prep_s2])
    V = _U2 @ _U1
    out = V @ rin @ V.conj().T
    O = _embed1(_PAULI[p], 0) @ _embed1(_PAULI[q], 2)
    return float(np.real(np.trace(O @ out)))


def null_heldout_correlator(p: str, q: str, prep_s1: str = "|+>",
                            prep_s2: str = "|+>") -> float:
    """Noiseless dark=False <P_S1 Q_S2> for the held-out deeper comb.

    Circuit word (link absent): CZ_{S1,E}, CZ_{S1,E}, CZ_{S2,E}, CZ_{S1,E}.
    """
    rin = np.kron(np.kron(_PREP_DM[prep_s1], _EPLUS), _PREP_DM[prep_s2])
    V = _U1 @ _U2 @ _U1 @ _U1           # circuit order U1, U1, U2, U1
    out = V @ rin @ V.conj().T
    O = _embed1(_PAULI[p], 0) @ _embed1(_PAULI[q], 2)
    return float(np.real(np.trace(O @ out)))


def null_atomic_expectation(slot_open: int, prep: str, pauli: str) -> float:
    """Noiseless dark=False atomic marginal expectation.

    Matches the causal-break tester: couple the open slot to a fresh |+> memory,
    reset the memory, measure the open slot.  Independent of the (inter-slot)
    dark link by construction, hence dark-invariant.
    """
    if slot_open == 1:
        sys_pos = 0
        rin = np.kron(np.kron(_PREP_DM[prep], _EPLUS), _HALF)
        U = _cz(0, 1)
    else:
        sys_pos = 2
        rin = np.kron(np.kron(_HALF, _EPLUS), _PREP_DM[prep])
        U = _cz(2, 1)
    out = U @ rin @ U.conj().T           # reset E afterwards is irrelevant to S
    r_sys = _ptrace(out, [sys_pos])
    return float(np.real(np.trace(_PAULI[pauli] @ r_sys)))


# --------------------------------------------------------------------------- #
# shared detection helpers                                                     #
# --------------------------------------------------------------------------- #
def _gate(alpha_exp: float, m_family: int, n_inst: int) -> float:
    """Bonferroni max-|z| gate at experiment-wide error ``alpha_exp``."""
    n_tests = max(1, m_family) * max(1, n_inst)
    return norm_ppf(1.0 - alpha_exp / (2.0 * n_tests))


def _z_offnull(chat: float, n_shots: int, eps: float = 1e-6) -> float:
    se = math.sqrt(max(eps, 1.0 - chat * chat) / n_shots)
    return chat / se


def _max_z_decision(offnull_chats: Dict[str, float], gate: float,
                    n_shots: int) -> Dict:
    z = {k: _z_offnull(v, n_shots) for k, v in offnull_chats.items()}
    if not z:
        return {"detected": False, "max_abs_z": 0.0, "argmax": None, "gate": gate}
    arg = max(z, key=lambda k: abs(z[k]))
    return {
        "detected": bool(abs(z[arg]) > gate),
        "max_abs_z": abs(z[arg]),
        "argmax": arg,
        "gate": gate,
    }


# --------------------------------------------------------------------------- #
# baseline (i): marginal QPT  -- must fail (Theorem 1)                          #
# --------------------------------------------------------------------------- #
def marginal_qpt_baseline(instance: ChallengeInstance, estimator: FrozenEstimator,
                          runner: RunnerT) -> Tuple[Dict, ResourceLedger]:
    """Atomic circuits only.  Blind to the dark link by Theorem 1."""
    n = estimator.n_shots
    cf = CircuitFamily(instance)
    settings_circ = cf.atomic_testers()
    circuits = [qc for _, qc in settings_circ]
    counts = runner(circuits, n)

    # off-null atomic stats (ideal dark=False expectation approx 0)
    offnull = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        prep, slot_open, pauli = setting.meta
        null = null_atomic_expectation(slot_open, prep, pauli)
        if abs(null) < 0.5:
            chat, _ = single_expectation_from_counts(cnt)
            offnull[setting.label] = chat

    gate = _gate(estimator.alpha_exp, len(offnull), estimator.n_inst_declared)
    verdict = _max_z_decision(offnull, gate, n)
    verdict["method"] = "marginal_qpt"
    verdict["note"] = "atomic marginals are dark-invariant (Theorem 1)"

    ledger = ResourceLedger("marginal_qpt")
    ledger.add("atomic", len(settings_circ), n)
    ledger.set_extra("offnull_family_size", len(offnull))
    ledger.set_extra("deterministic_identity_effects", 12)  # 6 preps x 2 slots
    return verdict, ledger


# --------------------------------------------------------------------------- #
# baseline (ii): restricted PTT  -- unitary-only 2-slot controls                #
# --------------------------------------------------------------------------- #
def restricted_ptt_baseline(instance: ChallengeInstance, estimator: FrozenEstimator,
                            runner: RunnerT) -> Tuple[Dict, ResourceLedger]:
    """Full IC unitary-preparation scan: 6 x 6 preps x 9 Pauli pairs = 324."""
    n = estimator.n_shots
    cf = CircuitFamily(instance)
    settings_circ: List[Tuple[Setting, object]] = []
    for prep_s1 in PREP_ORDER:
        for prep_s2 in PREP_ORDER:
            for (p, q) in PAULI_PAIRS:
                qc = cf.comb_correlator_circuit(prep_s1, prep_s2, p, q)
                lab = f"{prep_s1}|{prep_s2}|{p}{q}"
                settings_circ.append(
                    (Setting("restricted_ptt", lab, "pair", (prep_s1, prep_s2, p, q)), qc))
    circuits = [qc for _, qc in settings_circ]
    counts = runner(circuits, n)

    offnull = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        prep_s1, prep_s2, p, q = setting.meta
        null = null_pair_correlator(prep_s1, prep_s2, p, q)
        if abs(null) < 0.5:
            chat, _ = pair_correlator_from_counts(cnt)
            offnull[setting.label] = chat

    gate = _gate(estimator.alpha_exp, len(offnull), estimator.n_inst_declared)
    verdict = _max_z_decision(offnull, gate, n)
    verdict["method"] = "restricted_ptt"

    ledger = ResourceLedger("restricted_ptt")
    ledger.add("depth2_unitary_scan", len(settings_circ), n)
    ledger.set_extra("operation_basis", "unitary-restricted (6x6 IC preps x 9 pairs)")
    ledger.set_extra("offnull_family_size", len(offnull))
    return verdict, ledger


# --------------------------------------------------------------------------- #
# baseline (iii): full PTT  -- the 256 = d^{4k} operation-basis ceiling         #
# --------------------------------------------------------------------------- #
# process-tensor-IC operation basis per qubit slot: 4 IC input states x 4 IC
# effects = 16 (= d^4).  The 3 non-trivial Pauli effects are measured; the
# identity effect is the deterministic normalization (no shots).  Over the 2
# slots the operation basis is 16 x 16 = 256 = d^{4k} (k=2); the executed
# circuits are 4 preps x 3 Paulis per slot = 12 x 12 = 144.
_PTIC_PREPS = ["|0>", "|1>", "|+>", "|+i>"]      # 4 IC input states (a basis of B(C^2))


def full_ptt_baseline(instance: ChallengeInstance, estimator: FrozenEstimator,
                      runner: RunnerT) -> Tuple[Dict, ResourceLedger]:
    """Full process-tensor tomography on the tractable 2-slot instance."""
    n = estimator.n_shots
    cf = CircuitFamily(instance)
    settings_circ: List[Tuple[Setting, object]] = []
    for prep_s1 in _PTIC_PREPS:
        for prep_s2 in _PTIC_PREPS:
            for p in PAULI_1Q:
                for q in PAULI_1Q:
                    qc = cf.comb_correlator_circuit(prep_s1, prep_s2, p, q)
                    lab = f"{prep_s1}|{prep_s2}|{p}{q}"
                    settings_circ.append(
                        (Setting("full_ptt", lab, "pair", (prep_s1, prep_s2, p, q)), qc))
    circuits = [qc for _, qc in settings_circ]
    counts = runner(circuits, n)

    offnull = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        prep_s1, prep_s2, p, q = setting.meta
        null = null_pair_correlator(prep_s1, prep_s2, p, q)
        if abs(null) < 0.5:
            chat, _ = pair_correlator_from_counts(cnt)
            offnull[setting.label] = chat

    gate = _gate(estimator.alpha_exp, len(offnull), estimator.n_inst_declared)
    verdict = _max_z_decision(offnull, gate, n)
    verdict["method"] = "full_ptt"

    d = 2
    k = 2
    op_basis = (d ** 4) ** k              # 256
    ledger = ResourceLedger("full_ptt")
    ledger.add("depth2_ic_instrument_scan", len(settings_circ), n)
    ledger.set_extra("operation_basis_size", op_basis)   # d^{4k} = 256 ceiling
    ledger.set_extra("executed_circuits", len(settings_circ))
    ledger.set_extra("offnull_family_size", len(offnull))
    return verdict, ledger


# --------------------------------------------------------------------------- #
# baseline (iv): structured low-memory  -- O6 strong comparator                 #
# --------------------------------------------------------------------------- #
def structured_lowmem_baseline(instance: ChallengeInstance, estimator: FrozenEstimator,
                               runner: RunnerT) -> Tuple[Dict, ResourceLedger]:
    """Fit a bond-dim-2 comb model (the true class) by least squares.

    To first order the retained-S1 depth-2 correlators of the CZ--R(delta,axis)
    --CZ comb are ``<Y_S1 X_S2> = delta_x`` and ``<Z_S1 X_S2> = -delta_y`` with
    ``(delta_x, delta_y) = delta (cos axis, sin axis)``.  The least-squares fit
    of the two link components to the 9 measured correlators is therefore
    ``(delta_x_hat, delta_y_hat) = (<YX>, -<ZX>)``, and the test statistic is the
    max standardized component -- a proper structured (memory-order-prior)
    detector at the same 9-setting cost as the certificate.  This is the O6
    comparator the honesty ledger requires: it detects the planted dark link as
    cheaply as the certificate, so the certificate's proven resource advantage
    is over *unstructured* PTT, not over a structured method with the same prior.
    """
    n = estimator.n_shots
    cf = CircuitFamily(instance)
    settings_circ = cf.depth2_correlators()
    circuits = [qc for _, qc in settings_circ]
    counts = runner(circuits, n)

    corr = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        chat, _ = pair_correlator_from_counts(cnt)
        corr[setting.label] = chat

    # least-squares bond-dim-2 fit (first-order): the two link components
    delta_x_hat = corr.get("YX", 0.0)
    delta_y_hat = -corr.get("ZX", 0.0)
    fitted = {"YX_component": delta_x_hat, "ZX_component": delta_y_hat}

    gate = _gate(estimator.alpha_exp, 2, estimator.n_inst_declared)
    verdict = _max_z_decision(fitted, gate, n)
    verdict["method"] = "structured_lowmem"
    verdict["delta_hat"] = math.hypot(delta_x_hat, delta_y_hat)
    verdict["note"] = "bond-dim-2 comb LS fit; O6 structured comparator"

    ledger = ResourceLedger("structured_lowmem")
    ledger.add("depth2", len(settings_circ), n)
    ledger.set_extra("model", "bond-dim-2 comb (memory-order-2 prior)")
    ledger.set_extra("fitted_params", 2)
    return verdict, ledger


ALL_BASELINES = {
    "marginal_qpt": marginal_qpt_baseline,
    "restricted_ptt": restricted_ptt_baseline,
    "full_ptt": full_ptt_baseline,
    "structured_lowmem": structured_lowmem_baseline,
}

__all__ = [
    "null_pair_correlator", "null_atomic_expectation", "null_heldout_correlator",
    "marginal_qpt_baseline", "restricted_ptt_baseline",
    "full_ptt_baseline", "structured_lowmem_baseline", "ALL_BASELINES",
]
