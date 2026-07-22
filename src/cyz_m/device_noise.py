"""Channel-level device-noise models for the C7 depth-2 separator comb.

This module implements literature-parameterized, explicitly CPTP open-system
models of the two-qubit-plus-memory collision comb used by the C7
"operational-dimension" separator (see ``examples/run_c7_prototype.py`` and
``research/DEVICE_NOISE_PARAMETERS.md``).

Ideal construction (register order ``[S1, E, S2]``, dims ``[2, 2, 2]``):

    E initialised in |+><+|
    slot 1 (write) : U1 = CZ_{S1,E}
    dark link      : G(delta) = R_x(delta) = exp(-i * delta * X_E / 2)   on E
    slot 2 (read)  : U2 = CZ_{S2,E}
    discard E

The depth-2 separator is the correlator ``<Y_S1 X_S2>`` with S1 = S2 = |+>,
which equals ``sin(delta)`` exactly in the noiseless limit (the "dark
direction" is first-order visible at depth 2 but exactly invisible to the
complete single-time IC atomic family on S).

Noise is applied as explicit Kraus channels composed with the ideal comb
unitaries, everything on the 8-dimensional register before E is discarded:

    prep-error  ->  U1  ->  2Q depol{S1,E}  ->  R_x(delta) + 1Q depol(E)
                ->  S1 mid-circuit window  ->  U2  ->  2Q depol{S2,E}
                ->  1Q depol(S1) + 1Q depol(S2)  ->  discard E  ->  readout

The S1 mid-circuit window is the decisive term: S1 idles (as a retained
spectator) while the memory E is processed / read, and its coherence in the
X-Y plane is exactly what the ``Y_S1`` factor of the separator probes.

Two platform models are provided as ``PlatformModel`` constructors, using the
verified (checkmark) numbers from the campaign snapshot as defaults:

* ``PlatformModel.quantinuum_h2()`` -- trapped-ion QCCD (Moses PRX 13 041052,
  DeCross PRX 15 021052).  Mid-circuit window = small S1 dephasing in
  [2e-4, 4e-4] plus 5e-6 crosstalk.  ``phi_kick`` is inert for the ion model.
* ``PlatformModel.ibm_heron_r2()`` -- transmon heavy-hex (ibm_marrakesh
  snapshot arXiv:2605.24252, AbuGhanem J Supercomput 81 687).  Mid-circuit
  window = T2=96.89us idle dephasing over a 1.5us window PLUS a swept ZZ
  phase-kick dephasing channel with angle ``phi_kick`` (the UNVERIFIED
  measurement-induced-dephasing sensitivity parameter).

Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import itertools
import numpy as np

# --------------------------------------------------------------------------- #
# qubit algebra                                                               #
# --------------------------------------------------------------------------- #
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

_k0 = np.array([1, 0], dtype=complex)
_k1 = np.array([0, 1], dtype=complex)
_kp = (_k0 + _k1) / np.sqrt(2)          # |+>
_km = (_k0 - _k1) / np.sqrt(2)          # |->
_kpi = (_k0 + 1j * _k1) / np.sqrt(2)    # |+i>
_kmi = (_k0 - 1j * _k1) / np.sqrt(2)    # |-i>


def dm(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


# IC single-qubit preparations (6 Pauli eigenstates) and their labels.
PREP = {
    "|0>": dm(_k0), "|1>": dm(_k1), "|+>": dm(_kp),
    "|->": dm(_km), "|+i>": dm(_kpi), "|-i>": dm(_kmi),
}
PREP_ORDER = ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"]
EFFECT_ORDER = ["I", "X", "Y", "Z"]
HALF = I2 / 2.0

# register plumbing: [S1, E, S2]
N_REG = 3
DIMS = [2, 2, 2]
S1, E, S2 = 0, 1, 2


def rx(theta: float) -> np.ndarray:
    """R_x(theta) = exp(-i theta X / 2)."""
    return np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * X


def embed1(op: np.ndarray, pos: int, n: int = N_REG) -> np.ndarray:
    ops = [I2] * n
    ops[pos] = op
    out = ops[0]
    for k in range(1, n):
        out = np.kron(out, ops[k])
    return out


def cz(a: int, b: int, n: int = N_REG) -> np.ndarray:
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[a] == 1 and bits[b] == 1:
            diag[idx] = -1.0
    return np.diag(diag)


def partial_trace(rho: np.ndarray, keep: Sequence[int], dims: Sequence[int] = DIMS) -> np.ndarray:
    n = len(dims)
    rho = rho.reshape(list(dims) + list(dims))
    keep = sorted(keep)
    row = list(range(n))
    col = list(range(n, 2 * n))
    for ax in [i for i in range(n) if i not in keep]:
        col[ax] = row[ax]
    out_r = [row[i] for i in keep]
    out_c = [col[i] for i in keep]
    rho = np.einsum(rho, row + col, out_r + out_c)
    d = int(np.prod([dims[i] for i in keep]))
    return rho.reshape(d, d)


# --------------------------------------------------------------------------- #
# Kraus channels                                                              #
# --------------------------------------------------------------------------- #
# Every channel is a list of Kraus operators on the FULL register (8x8) so
# that they compose by simple multiplication and the whole comb remains an
# explicit CPTP map.  Single-qubit / two-qubit channels are provided as small
# Kraus sets first and then embedded into the register.

def apply_kraus(rho: np.ndarray, kraus: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(rho)
    for K in kraus:
        out = out + K @ rho @ K.conj().T
    return out


def kraus_depolarizing_1q(p: float) -> List[np.ndarray]:
    """Single-qubit depolarizing: rho -> (1-p) rho + p I/2 * Tr(rho).

    CPTP for 0 <= p <= 1.  Pauli-Kraus form:
        rho -> (1 - 3p/4) rho + (p/4) (X rho X + Y rho Y + Z rho Z).
    """
    if p < 0:
        raise ValueError("depolarizing probability must be >= 0")
    c0 = np.sqrt(max(0.0, 1.0 - 3.0 * p / 4.0))
    ck = np.sqrt(max(0.0, p / 4.0))
    return [c0 * I2, ck * X, ck * Y, ck * Z]


def kraus_depolarizing_2q(p: float) -> List[np.ndarray]:
    """Two-qubit depolarizing: rho -> (1-p) rho + p I4/4 * Tr(rho).

    CPTP for 0 <= p <= 1.  Expressed over the 16 two-qubit Paulis:
        rho -> (1 - p + p/16) rho + (p/16) sum_{P != I} P rho P.
    """
    if p < 0:
        raise ValueError("depolarizing probability must be >= 0")
    paulis = [I2, X, Y, Z]
    kraus = []
    for i, Pa in enumerate(paulis):
        for j, Pb in enumerate(paulis):
            P = np.kron(Pa, Pb)
            if i == 0 and j == 0:
                coeff = np.sqrt(max(0.0, 1.0 - p + p / 16.0))
            else:
                coeff = np.sqrt(max(0.0, p / 16.0))
            kraus.append(coeff * P)
    return kraus


def kraus_dephasing_1q(p: float) -> List[np.ndarray]:
    """Single-qubit phase-flip (dephasing): rho -> (1-p) rho + p Z rho Z.

    Off-diagonal coherence is scaled by (1 - 2p).  CPTP for 0 <= p <= 1.
    """
    if p < 0:
        raise ValueError("dephasing probability must be >= 0")
    return [np.sqrt(max(0.0, 1.0 - p)) * I2, np.sqrt(max(0.0, p)) * Z]


def dephasing_p_from_T2(t_window: float, T2: float) -> float:
    """Phase-flip probability reproducing exp(-t/T2) coherence decay.

    A pure-dephasing (Tphi = T2) channel scales the off-diagonal by (1-2p);
    matching exp(-t/T2) gives p = (1 - exp(-t/T2)) / 2.
    """
    return 0.5 * (1.0 - np.exp(-t_window / T2))


def dephasing_p_from_zz_kick(phi_kick: float) -> float:
    """Phase-flip probability for a ZZ measurement-induced phase kick.

    A conditional phase exp(-i (phi/2) Z_S1 Z_ancilla) averaged over a
    maximally-mixed measured ancilla scales the S1 off-diagonal by cos(phi),
    i.e. a phase-flip channel with p = (1 - cos(phi)) / 2 = sin^2(phi/2).
    Reduces to the identity channel at phi = 0.
    """
    return 0.5 * (1.0 - np.cos(phi_kick))


def embed_kraus_1q(kraus_1q: Sequence[np.ndarray], pos: int) -> List[np.ndarray]:
    return [embed1(K, pos) for K in kraus_1q]


def embed_kraus_2q(kraus_2q_paulis: Sequence[np.ndarray], a: int, b: int) -> List[np.ndarray]:
    """Embed a set of 2-qubit *Pauli-product* Kraus operators onto (a, b).

    Each input Kraus is a 4x4 operator built as ``coeff * kron(Pa, Pb)``; it is
    placed with ``Pa`` on qubit ``a`` and ``Pb`` on qubit ``b`` via the fact
    that Paulis on distinct qubits commute:  embed1(Pa, a) @ embed1(Pb, b).
    """
    paulis = [I2, X, Y, Z]
    out = []
    for i, Pa in enumerate(paulis):
        for j, Pb in enumerate(paulis):
            # recover the coefficient of this Pauli product from the 4x4 Kraus
            idx = 4 * i + j
            K4 = kraus_2q_paulis[idx]
            # coefficient = <Pa x Pb, K4> / 4  (Hilbert-Schmidt, orthonormal up to 4)
            coeff = np.trace(np.kron(Pa, Pb).conj().T @ K4) / 4.0
            out.append(coeff * embed1(Pa, a) @ embed1(Pb, b))
    return out


# --------------------------------------------------------------------------- #
# Choi / CPTP validation                                                      #
# --------------------------------------------------------------------------- #
def choi_from_kraus(kraus: Sequence[np.ndarray], din: int) -> np.ndarray:
    """Choi matrix J = sum_ij |i><j| (x) Phi(|i><j|) with Phi(.)=sum K . K^dag."""
    dout = kraus[0].shape[0]
    J = np.zeros((din * dout, din * dout), dtype=complex)
    for i in range(din):
        for j in range(din):
            Eij = np.zeros((din, din), dtype=complex)
            Eij[i, j] = 1.0
            phi = apply_kraus(Eij, kraus)
            J += np.kron(Eij, phi)
    return J


def choi_min_eig(kraus: Sequence[np.ndarray], din: int) -> float:
    J = choi_from_kraus(kraus, din)
    J = (J + J.conj().T) / 2.0
    return float(np.min(np.linalg.eigvalsh(J)).real)


def tp_defect(kraus: Sequence[np.ndarray], din: int) -> float:
    """max |sum_k K_k^dag K_k - I|  (0 for trace preserving)."""
    acc = np.zeros((din, din), dtype=complex)
    for K in kraus:
        acc += K.conj().T @ K
    return float(np.max(np.abs(acc - np.eye(din))))


def assert_cptp(kraus: Sequence[np.ndarray], din: int, name: str,
                tol_cp: float = -1e-12, tol_tp: float = 1e-10) -> Tuple[float, float]:
    """Validate CP (Choi min eig >= tol_cp) and TP (defect <= tol_tp)."""
    mn = choi_min_eig(kraus, din)
    tp = tp_defect(kraus, din)
    if mn < tol_cp:
        raise AssertionError(f"channel {name!r} not CP: Choi min eig {mn:.3e} < {tol_cp:.1e}")
    if tp > tol_tp:
        raise AssertionError(f"channel {name!r} not TP: defect {tp:.3e} > {tol_tp:.1e}")
    return mn, tp


# --------------------------------------------------------------------------- #
# platform model                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlatformModel:
    """Frozen literature-parameterized noise model for the separator comb.

    All error rates are per-gate / per-operation probabilities.  Provenance of
    each default is documented on the two constructors below.  ``mid_circuit``
    selects the S1 spectator-window model: ``"ion"`` (dephasing + crosstalk,
    ``phi_kick`` inert) or ``"ibm"`` (T2 idle dephasing + swept ZZ phase kick).
    """

    name: str
    citation: str
    mid_circuit: str                      # "ion" | "ibm"
    err_1q: float                         # 1Q depolarizing probability
    err_2q: float                         # 2Q depolarizing probability
    prep_error: float                     # per-qubit preparation depolarizing
    readout_error: float                  # per-qubit symmetric assignment error
    # ion window
    ion_dephasing: float = 0.0            # S1 dephasing probability over the ms window
    ion_crosstalk: float = 0.0            # S1 depolarizing crosstalk from the mid-circuit op
    # ibm window
    T2: Optional[float] = None            # seconds
    mcm_window: Optional[float] = None    # seconds (readout / MCM duration)
    # provenance / status notes
    notes: Tuple[str, ...] = field(default_factory=tuple)

    # ---- constructors ---------------------------------------------------- #
    @classmethod
    def quantinuum_h2(cls) -> "PlatformModel":
        """Trapped-ion QCCD (Quantinuum H2 / H2-1), verified snapshot.

        Sources: Moses PRX 13, 041052 (racetrack, arXiv:2305.03828);
        DeCross PRX 15, 021052 (RCS).  All defaults are checkmark-verified.

        Defaults use the Moses H2 32q figures:
          * 1Q gate error (RB)          2.5e-5     [Moses, verified]
          * 2Q gate error (RB)          1.83e-3    [Moses, verified]
          * SPAM total                  1.6e-3     [Moses, verified]
        The SPAM total is split into prep_error 6.0e-4 + symmetric readout
        assignment 1.0e-3 (sum = 1.6e-3).  The split is a modeling choice
        consistent with the (unverified) H2-1 readout split 6.0e-4 / 1.38e-3
        whose average is ~1.0e-3; the *total* is verified.
          * S1 mid-circuit dephasing    3.0e-4     midpoint of the verified
                                                   idle/memory range 2.2e-4
                                                   (2023) .. 4.0e-4 (56q)
          * S1 crosstalk                5.0e-6     (verified meas. crosstalk
                                                   4.5e-6; 5e-6 nominal)
        ``phi_kick`` has no effect on the ion model.
        """
        return cls(
            name="Quantinuum H2 (trapped ion, QCCD)",
            citation="Moses PRX 13 041052; DeCross PRX 15 021052",
            mid_circuit="ion",
            err_1q=2.5e-5,
            err_2q=1.83e-3,
            prep_error=6.0e-4,
            readout_error=1.0e-3,
            ion_dephasing=3.0e-4,
            ion_crosstalk=5.0e-6,
            notes=(
                "1Q 2.5e-5, 2Q 1.83e-3, SPAM 1.6e-3 (Moses, verified)",
                "SPAM split prep 6.0e-4 + readout 1.0e-3 (total verified)",
                "S1 dephasing 3.0e-4 = midpoint of verified [2.2e-4,4.0e-4]",
                "crosstalk 5.0e-6 (verified meas crosstalk 4.5e-6)",
                "phi_kick inert for the ion model",
            ),
        )

    @classmethod
    def ibm_heron_r2(cls) -> "PlatformModel":
        """Transmon heavy-hex (IBM Heron r2, ibm_marrakesh / ibm_fez).

        Sources: ibm_marrakesh snapshot arXiv:2605.24252 App. A;
        AbuGhanem J Supercomput 81, 687 (2025).

          * 1Q SX error                 2.9e-4     Heron ~3-4e-4 (anchored to
                                                   verified Eagle sherbrooke
                                                   2.411e-4)
          * 2Q CZ error (median)        2.334e-3   [ibm_marrakesh, verified]
          * readout assignment (median) 1.23e-3    [ibm_marrakesh, verified]
          * T2 (median)                 96.89 us   [ibm_marrakesh, verified]
          * MCM / readout window        1.5 us     [duration, unverified lower
                                                   bound]  -> idle dephasing
                                                   p = (1-exp(-t/T2))/2
          * ZZ phase-kick               swept via phi_kick (UNVERIFIED
                                                   measurement-induced spectator
                                                   dephasing; mechanism per
                                                   Govia arXiv:2204.03104)

        prep_error is set to 0: no independent verified IBM preparation number
        exists; the verified assignment error 1.23e-3 carries the SPAM budget.
        """
        return cls(
            name="IBM Heron r2 (transmon, heavy-hex)",
            citation="ibm_marrakesh arXiv:2605.24252; AbuGhanem J Supercomput 81 687",
            mid_circuit="ibm",
            err_1q=2.9e-4,
            err_2q=2.334e-3,
            prep_error=0.0,
            readout_error=1.23e-3,
            T2=96.89e-6,
            mcm_window=1.5e-6,
            notes=(
                "2Q 2.334e-3, readout 1.23e-3, T2 96.89us (marrakesh, verified)",
                "1Q 2.9e-4 (Heron ~3-4e-4; verified Eagle anchor 2.411e-4)",
                "MCM window 1.5us UNVERIFIED lower bound",
                "ZZ phase-kick phi_kick UNVERIFIED (Govia mechanism)",
                "prep_error 0 (no verified IBM prep number; readout carries SPAM)",
            ),
        )

    # ---- window channel -------------------------------------------------- #
    def ion_dephasing_p(self) -> float:
        return self.ion_dephasing

    def ibm_idle_dephasing_p(self) -> float:
        if self.T2 is None or self.mcm_window is None:
            raise ValueError("IBM idle dephasing needs T2 and mcm_window")
        return dephasing_p_from_T2(self.mcm_window, self.T2)

    def s1_window_kraus(self, phi_kick: float) -> List[np.ndarray]:
        """Kraus operators (register-embedded on S1) for the mid-circuit window."""
        kraus: List[np.ndarray] = []
        if self.mid_circuit == "ion":
            deph = kraus_dephasing_1q(self.ion_dephasing_p())
            xtalk = kraus_depolarizing_1q(self.ion_crosstalk)
            kraus = _compose_1q_on(deph, xtalk, pos=S1)
        elif self.mid_circuit == "ibm":
            deph_idle = kraus_dephasing_1q(self.ibm_idle_dephasing_p())
            deph_zz = kraus_dephasing_1q(dephasing_p_from_zz_kick(phi_kick))
            kraus = _compose_1q_on(deph_idle, deph_zz, pos=S1)
        else:
            raise ValueError(f"unknown mid_circuit model {self.mid_circuit!r}")
        return kraus


def _compose_1q_on(kraus_a: Sequence[np.ndarray], kraus_b: Sequence[np.ndarray],
                   pos: int) -> List[np.ndarray]:
    """Register-embedded Kraus set for applying channel A then channel B on ``pos``."""
    out = []
    for Kb in kraus_b:
        for Ka in kraus_a:
            out.append(embed1(Kb @ Ka, pos))
    return out


# --------------------------------------------------------------------------- #
# noisy comb                                                                  #
# --------------------------------------------------------------------------- #
_U1 = cz(S1, E)                 # write CZ_{S1,E}
_U2 = cz(S2, E)                 # read  CZ_{S2,E}
_E_INIT = dm(_kp)              # |+><+|_E


def _u(op: np.ndarray) -> List[np.ndarray]:
    """Unitary as a one-element Kraus list."""
    return [op]


def comb_channel_kraus(platform: PlatformModel, delta: float, phi_kick: float
                       ) -> List[List[np.ndarray]]:
    """Ordered list of register (8x8) Kraus stages for the noisy comb.

    The stages act on the full [S1,E,S2] register; E is discarded afterwards.
    Preparation error is applied to each freshly prepared qubit (S1, E, S2).
    """
    stages: List[List[np.ndarray]] = []
    # preparation error on each prepared qubit
    if platform.prep_error > 0:
        prep = kraus_depolarizing_1q(platform.prep_error)
        stages.append(embed_kraus_1q(prep, S1))
        stages.append(embed_kraus_1q(prep, E))
        stages.append(embed_kraus_1q(prep, S2))
    # write CZ + 2Q depol on {S1,E}
    stages.append(_u(_U1))
    if platform.err_2q > 0:
        stages.append(embed_kraus_2q(kraus_depolarizing_2q(platform.err_2q), S1, E))
    # dark link R_x(delta) on E + 1Q depol on E (physical link gate)
    stages.append(_u(embed1(rx(delta), E)))
    if platform.err_1q > 0:
        stages.append(embed_kraus_1q(kraus_depolarizing_1q(platform.err_1q), E))
    # S1 mid-circuit spectator window (the decisive term)
    stages.append(platform.s1_window_kraus(phi_kick))
    # read CZ + 2Q depol on {S2,E}
    stages.append(_u(_U2))
    if platform.err_2q > 0:
        stages.append(embed_kraus_2q(kraus_depolarizing_2q(platform.err_2q), S2, E))
    # pre-measurement 1Q basis-rotation gate noise on S1 and S2
    if platform.err_1q > 0:
        stages.append(embed_kraus_1q(kraus_depolarizing_1q(platform.err_1q), S1))
        stages.append(embed_kraus_1q(kraus_depolarizing_1q(platform.err_1q), S2))
    return stages


def noisy_comb_output(platform: PlatformModel, rho_S1: np.ndarray, rho_S2: np.ndarray,
                      delta: float, phi_kick: float) -> np.ndarray:
    """Return the 4x4 reduced output state on [S1, S2] after discarding E."""
    rho = np.kron(np.kron(rho_S1, _E_INIT), rho_S2)
    for stage in comb_channel_kraus(platform, delta, phi_kick):
        rho = apply_kraus(rho, stage)
    return partial_trace(rho, keep=[S1, S2])


def _lift_S1S2(M: np.ndarray) -> np.ndarray:
    """Lift a 4x4 operator on [S1,S2] into the register [S1,E,S2] with E=|+><+|.

    M is indexed as M[2*s1+s2, 2*t1+t2]; the ancilla E is fixed to |+>.
    """
    rho_full = np.zeros((8, 8), dtype=complex)
    basis = [_k0, _k1]
    for s1 in range(2):
        for s2 in range(2):
            for t1 in range(2):
                for t2 in range(2):
                    c = M[2 * s1 + s2, 2 * t1 + t2]
                    if c == 0:
                        continue
                    ket = np.kron(np.kron(basis[s1], _kp), basis[s2])
                    bra = np.kron(np.kron(basis[t1], _kp), basis[t2])
                    rho_full += c * np.outer(ket, bra.conj())
    return rho_full


def apply_comb_channel_joint(platform: PlatformModel, rho_S1S2: np.ndarray,
                             delta: float, phi_kick: float) -> np.ndarray:
    """The whole comb as a CPTP map on a *joint* 4x4 [S1,S2] input state.

    E is a fixed ancilla (init |+>), so the comb is a genuine channel on the
    4-dimensional system register even for entangled inputs.  Used to build the
    end-to-end Choi matrix for CP/TP validation.
    """
    rho = _lift_S1S2(rho_S1S2)
    for stage in comb_channel_kraus(platform, delta, phi_kick):
        rho = apply_kraus(rho, stage)
    return partial_trace(rho, keep=[S1, S2])


def comb_choi_on_S(platform: PlatformModel, delta: float, phi_kick: float
                   ) -> np.ndarray:
    """Choi matrix (16x16) of the whole comb channel on [S1,S2].

    J = sum_ij |i><j|_in (x) Phi(|i><j|)_out, computed by applying the channel
    to the 16 elementary input matrices (cheap: 16 channel evaluations).
    """
    J = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            Eij = np.zeros((4, 4), dtype=complex)
            Eij[i, j] = 1.0
            J += np.kron(Eij, apply_comb_channel_joint(platform, Eij, delta, phi_kick))
    return J


# --------------------------------------------------------------------------- #
# measurement (with readout assignment error) via 4 outcome probabilities     #
# --------------------------------------------------------------------------- #
def _pauli_projectors(pauli: str) -> Tuple[np.ndarray, np.ndarray]:
    P = PAULI[pauli]
    plus = (I2 + P) / 2.0
    minus = (I2 - P) / 2.0
    return plus, minus


def correlator_outcome_probs(rho_S: np.ndarray, obsA: str, obsB: str,
                             readout_error: float) -> np.ndarray:
    """Joint readout-corrupted outcome probabilities of <obsA_S1 obsB_S2>.

    Returns a 2x2 array ``p[a, b]`` for a, b in {0 -> +1, 1 -> -1} eigenvalues,
    after applying a symmetric per-qubit assignment-error confusion matrix.
    """
    Ap, Am = _pauli_projectors(obsA)
    Bp, Bm = _pauli_projectors(obsB)
    projA = [Ap, Am]
    projB = [Bp, Bm]
    p_ideal = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            proj = np.kron(projA[a], projB[b])
            p_ideal[a, b] = np.real(np.trace(proj @ rho_S))
    # symmetric assignment-error confusion on each qubit
    e = readout_error
    M = np.array([[1 - e, e], [e, 1 - e]])   # M[measured, true]
    p_meas = M @ p_ideal @ M.T
    return p_meas


def correlator_from_probs(p_meas: np.ndarray) -> float:
    """Correlator = sum_{a,b} eig(a) eig(b) p[a,b], eig(0)=+1, eig(1)=-1."""
    return float(p_meas[0, 0] + p_meas[1, 1] - p_meas[0, 1] - p_meas[1, 0])


def prob_product_plus(p_meas: np.ndarray) -> float:
    """P(product of the two +/-1 outcomes = +1) for binomial sampling."""
    return float(p_meas[0, 0] + p_meas[1, 1])


def noisy_correlator(platform: PlatformModel, obsA: str, obsB: str, delta: float,
                     phi_kick: float, rho_S1: np.ndarray, rho_S2: np.ndarray
                     ) -> Tuple[float, float]:
    """Noisy <obsA_S1 obsB_S2>; returns (correlator, P(product=+1))."""
    rho_S = noisy_comb_output(platform, rho_S1, rho_S2, delta, phi_kick)
    p_meas = correlator_outcome_probs(rho_S, obsA, obsB, platform.readout_error)
    return correlator_from_probs(p_meas), prob_product_plus(p_meas)


def noisy_separator(platform: PlatformModel, delta: float, phi_kick: float
                    ) -> float:
    """The headline depth-2 separator <Y_S1 X_S2> with S1 = S2 = |+>."""
    c, _ = noisy_correlator(platform, "Y", "X", delta, phi_kick, dm(_kp), dm(_kp))
    return c


def separator_prob_plus(platform: PlatformModel, delta: float, phi_kick: float
                        ) -> float:
    _, q = noisy_correlator(platform, "Y", "X", delta, phi_kick, dm(_kp), dm(_kp))
    return q


def ideal_separator(delta: float) -> float:
    """Noiseless reference value sin(delta)."""
    return float(np.sin(delta))


# --------------------------------------------------------------------------- #
# atomic (single-time IC) baseline family                                     #
# --------------------------------------------------------------------------- #
def _single_qubit_expectation_with_readout(rho_q: np.ndarray, obs: str,
                                           readout_error: float) -> float:
    """<obs> on a single qubit with symmetric assignment error (I -> 1)."""
    if obs == "I":
        return float(np.real(np.trace(rho_q)))
    P = PAULI[obs]
    plus = (I2 + P) / 2.0
    minus = (I2 - P) / 2.0
    p_plus = float(np.real(np.trace(plus @ rho_q)))
    p_minus = float(np.real(np.trace(minus @ rho_q)))
    e = readout_error
    p_plus_m = (1 - e) * p_plus + e * p_minus
    p_minus_m = e * p_plus + (1 - e) * p_minus
    return p_plus_m - p_minus_m


def noisy_atomic_response(platform: PlatformModel, delta: float, phi_kick: float
                          ) -> np.ndarray:
    """The 48 single-time IC atomic statistics on the noisy device.

    6 preparations x 4 Pauli effects x 2 slots.  Slot i open: prepare S_i in a
    Pauli eigenstate, close the other slot with a causal break (feed I/2),
    run the comb, discard E, read a Pauli expectation on S_i.
    """
    vals = []
    for lab in PREP_ORDER:
        rho = PREP[lab]
        # slot 1 open (S2 closed = I/2)
        rho_S = noisy_comb_output(platform, rho, HALF, delta, phi_kick)
        rS1 = partial_trace(rho_S, keep=[0], dims=[2, 2])
        for obs in EFFECT_ORDER:
            vals.append(_single_qubit_expectation_with_readout(rS1, obs, platform.readout_error))
        # slot 2 open (S1 closed = I/2)
        rho_S = noisy_comb_output(platform, HALF, rho, delta, phi_kick)
        rS2 = partial_trace(rho_S, keep=[1], dims=[2, 2])
        for obs in EFFECT_ORDER:
            vals.append(_single_qubit_expectation_with_readout(rS2, obs, platform.readout_error))
    return np.array(vals)


# --------------------------------------------------------------------------- #
# slope / contrast                                                            #
# --------------------------------------------------------------------------- #
def noisy_slope_at_zero(platform: PlatformModel, phi_kick: float, h: float = 1e-5
                        ) -> float:
    """d/ddelta of the noisy separator at delta = 0 (central difference)."""
    fp = noisy_separator(platform, +h, phi_kick)
    fm = noisy_separator(platform, -h, phi_kick)
    return (fp - fm) / (2.0 * h)


# --------------------------------------------------------------------------- #
# self-validation                                                             #
# --------------------------------------------------------------------------- #
def validate_all_channels(platform: PlatformModel, phi_kick: float = 0.05,
                          delta: float = 0.2) -> List[Tuple[str, float, float]]:
    """Validate CP (Choi min eig >= -1e-12) and TP for every channel.

    Returns a list of (name, choi_min_eig, tp_defect).
    """
    report: List[Tuple[str, float, float]] = []

    def _check(name: str, kraus: Sequence[np.ndarray], din: int):
        mn, tp = assert_cptp(kraus, din, name)
        report.append((name, mn, tp))

    _check("depol_1q(err_1q)", kraus_depolarizing_1q(platform.err_1q), 2)
    _check("depol_2q(err_2q)", kraus_depolarizing_2q(platform.err_2q), 4)
    _check("prep_depol", kraus_depolarizing_1q(platform.prep_error), 2)
    if platform.mid_circuit == "ion":
        _check("ion_dephasing", kraus_dephasing_1q(platform.ion_dephasing_p()), 2)
        _check("ion_crosstalk", kraus_depolarizing_1q(platform.ion_crosstalk), 2)
    else:
        _check("ibm_idle_dephasing", kraus_dephasing_1q(platform.ibm_idle_dephasing_p()), 2)
        _check("ibm_zz_kick", kraus_dephasing_1q(dephasing_p_from_zz_kick(phi_kick)), 2)
    # full comb as a channel on [S1,S2], validated via its 16x16 Choi matrix
    J = comb_choi_on_S(platform, delta, phi_kick)
    Jh = (J + J.conj().T) / 2.0
    mn = float(np.min(np.linalg.eigvalsh(Jh)).real)
    tr_out = partial_trace(J, keep=[0], dims=[4, 4])
    tp = float(np.max(np.abs(tr_out - np.eye(4))))
    if mn < -1e-12:
        raise AssertionError(f"full comb not CP: Choi min eig {mn:.3e}")
    if tp > 1e-10:
        raise AssertionError(f"full comb not TP: defect {tp:.3e}")
    report.append(("full_comb_process", mn, tp))
    return report


__all__ = [
    "PlatformModel",
    "PAULI", "PREP", "PREP_ORDER", "EFFECT_ORDER",
    "dm", "rx", "cz", "embed1", "partial_trace",
    "kraus_depolarizing_1q", "kraus_depolarizing_2q", "kraus_dephasing_1q",
    "dephasing_p_from_T2", "dephasing_p_from_zz_kick",
    "apply_kraus", "choi_from_kraus", "choi_min_eig", "tp_defect", "assert_cptp",
    "comb_channel_kraus", "noisy_comb_output", "apply_comb_channel_joint", "comb_choi_on_S",
    "correlator_outcome_probs", "correlator_from_probs", "prob_product_plus",
    "noisy_correlator", "noisy_separator", "separator_prob_plus", "ideal_separator",
    "noisy_atomic_response", "noisy_slope_at_zero", "validate_all_channels",
]
