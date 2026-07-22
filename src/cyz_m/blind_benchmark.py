"""Step-2 blind benchmark harness for the C7 flat-extension certificate.

This module implements the *software* side of the preregistered blind benchmark
specified in ``research/C7_THEOREM_PACKAGE.md`` §7 (kill-shot 3 hardening) and
constrained by the honesty ledger of GitHub issue #5 comment 3 (O1, O6, O7,
O8).  It is a simulator dry-run harness: no hardware credentials exist yet.

The physics is the CZ--link--CZ two-qubit-plus-memory collision comb of
``examples/run_c7_prototype.py`` and ``src/cyz_m/device_noise.py``:

    register [S1, E, S2], dims [2, 2, 2], E initialised in |+>
    slot 1 (write) : U1 = CZ_{S1,E}
    dark link      : G(delta, axis) = exp(-i (delta/2) (cos a X + sin a Y))  on E
    slot 2 (read)  : U2 = CZ_{S2,E},  discard E

The dark link is a first-jet-visible inter-slot generator on the memory that is
**exactly invisible** to the complete single-time IC atomic family on S
(Theorem 1) but first-order visible to one depth-2 retained-S1 correlator.  The
sigma_z axis is excluded by design: an R_z link commutes with both CZs (a pure
discard gauge, dark at every depth), so the challenge draws the dark axis
uniformly in the x--y plane only.

Numerically established (noiseless, S1 = S2 = |+>):

    * null (delta = 0):  <X_S1 X_S2> = 1, all eight other Pauli pairs = 0;
    * dark (inter slot): <Y_S1 X_S2> = sin(delta) cos(axis),
                         <Z_S1 X_S2> = -sin(delta) sin(axis),
      so sqrt(<YX>^2 + <ZX>^2) = |sin(delta)| is axis-invariant.

Blind discipline
----------------
* :class:`ChallengeGenerator` draws the hidden ``(dark, axis, delta, slot)`` per
  instance from a master seed and seals the answer key to canonical JSON whose
  sha256 is published in a manifest.  The public spec exposed to the analysis
  side carries **no** dark/axis/position fields.
* :class:`CircuitFamily` builds the qiskit circuits (atomic causal-break
  testers, depth-2 retained-S1 correlators, a held-out depth-3 family) for an
  instance.  The dark link is compiled into the circuit gates; the analysis
  side runs the circuits as opaque objects.
* :class:`FrozenEstimator` freezes its full config + thresholds (read from the
  Step-1 separator-power artifacts) and hashes it *before* seeing any data.
* :class:`ResourceLedger` counts distinct circuit settings and total shots.

Only the qiskit-circuit builders require qiskit; the generator, estimator,
ledger and correlator math are pure-Python/numpy so the discipline pieces are
importable and testable without qiskit installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# qiskit is imported lazily inside CircuitFamily so that the blind-discipline
# objects (generator / estimator / ledger) stay importable without it.

# --------------------------------------------------------------------------- #
# register layout and protocol constants                                       #
# --------------------------------------------------------------------------- #
S1, E, S2, SCRATCH = 0, 1, 2, 3            # qubit indices in the register
PREP_ORDER = ["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>"]
EFFECT_ORDER = ["I", "X", "Y", "Z"]        # I is the trivial (deterministic) effect
PAULI_1Q = ["X", "Y", "Z"]
PAULI_PAIRS = [(p, q) for p in PAULI_1Q for q in PAULI_1Q]   # 9 retained-S1 pairs

# The single retained-S1 depth-2 correlator null vector (dark = False), computed
# exactly from the noiseless comb: only <X_S1 X_S2> is non-zero.
CORRELATOR_NULL = {f"{p}{q}": (1.0 if (p, q) == ("X", "X") else 0.0)
                   for (p, q) in PAULI_PAIRS}
# Off-null family (all pairs whose noiseless null is 0).  This is the LEGACY
# decision family scanned by the v1 certificate (8 correlators).
OFFNULL_PAIRS = [f"{p}{q}" for (p, q) in PAULI_PAIRS if (p, q) != ("X", "X")]

# --------------------------------------------------------------------------- #
# signal subspace: the correlators carrying first-order dark signal            #
# --------------------------------------------------------------------------- #
# For a dark inter-slot link R(delta, phi) = exp(-i (delta/2)(cos phi X +
# sin phi Y)) on E, with S1 = E = S2 = |+> and the CZ-comb V = CZ_{S2,E} R
# CZ_{S1,E}, only two of the eight off-null retained-S1 correlators carry
# first-order dark signal.  Heisenberg derivation on the CZ comb (3 lines):
#
#   (1) READ leg.  Conjugating the read Pauli Q_{S2} back through CZ_{S2,E}
#       only exposes the memory when Q = X: CZ_{S2,E} X_{S2} CZ_{S2,E} =
#       X_{S2} Z_E (Q in {Y,Z} leave an identity/Z_E on E that the |+>_E
#       memory averages to 0).  => the whole signal lives in the Q = X column.
#   (2) LINK.  R(delta,phi)^dag Z_E R(delta,phi) = cos(delta) Z_E
#       - sin(delta) sin(phi) X_E + sin(delta) cos(phi) Y_E; the |+>_E average
#       kills <Z_E> = 0, leaving only the transverse X_E, Y_E components with
#       weights -sin(delta) sin(phi) and +sin(delta) cos(phi).
#   (3) WRITE leg.  Conjugating back through CZ_{S1,E} (note Y_{S1} does NOT
#       commute with CZ_{S1,E}) and averaging over |+>_{S1} |+>_E leaves
#       exactly two nonzero correlators:
#           <Y_{S1} X_{S2}> = sin(delta) cos(phi)      (separator for R_x, phi=0)
#           <Z_{S1} X_{S2}> = -sin(delta) sin(phi)
#       with <YX>^2 + <ZX>^2 = sin^2(delta) (axis-invariant).
#
# The remaining six off-null pairs {XY, XZ, YY, YZ, ZY, ZZ} are identically 0
# for every (delta, phi): they carry no dark signal and, when scanned, only
# inflate the multiple-comparison null floor.  The Marrakesh dress rehearsal
# (research/results/blind_benchmark_hardware_rehearsal) produced a false
# positive precisely there -- null instance 6's <XY> hit z = 4.17 vs the frozen
# z_gate = 3.78 -- which the signal-subspace restriction removes by construction.
# This 2-correlator subspace is exactly the one the O6 structured comparator
# fits (delta_x = <YX>, delta_y = -<ZX>), which stayed clean at 1.00 power / 0 FP.
SIGNAL_SUBSPACE_PAIRS = ["YX", "ZX"]

# insertion-slot vocabulary (memory-line links between comb operations)
SLOT_PREWRITE = "prewrite"    # link on E before U1  -- LEAKS to atomic family
SLOT_INTER = "inter"          # link on E between U1 and U2 -- the dark position
SLOT_POSTREAD = "postread"    # link on E after U2 -- discard gauge (dark all depths)
# Only ``inter`` is atomically dark AND depth-2 recoverable for the 2-slot comb.
VALID_DARK_SLOTS_2 = (SLOT_INTER,)


# --------------------------------------------------------------------------- #
# canonical hashing                                                            #
# --------------------------------------------------------------------------- #
def canonical_json(obj) -> str:
    """Deterministic JSON string for hashing (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_obj(obj) -> str:
    return sha256_hex(canonical_json(obj))


def sha256_of_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# normal quantile (Acklam) -- no scipy dependency, matches Step-1 convention   #
# --------------------------------------------------------------------------- #
def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Peter Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------- #
# challenge instances + generator                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChallengeInstance:
    """A single blind challenge.

    The hidden answer-key fields are ``dark``, ``axis``, ``delta`` and ``slot``.
    The public fields are ``instance_id`` and ``n_slots``.
    """

    instance_id: int
    n_slots: int
    dark: bool
    axis: float             # radians in the x-y plane; meaningless if not dark
    delta: float            # link magnitude; 0.0 if not dark
    slot: Optional[str]     # insertion memory-link; None if not dark

    # ---- views ---------------------------------------------------------- #
    def answer_key(self) -> Dict:
        """Full (sealed) record including the hidden fields."""
        return {
            "instance_id": self.instance_id,
            "n_slots": self.n_slots,
            "dark": self.dark,
            "axis": self.axis,
            "delta": self.delta,
            "slot": self.slot,
        }

    def public_spec(self) -> Dict:
        """Analysis-facing spec: NO dark/axis/delta/slot fields."""
        return {
            "instance_id": self.instance_id,
            "n_slots": self.n_slots,
            "families": ["atomic", "depth2", "heldout_depth3"],
        }


class ChallengeGenerator:
    """Deterministic challenge generator with a sealed answer key.

    Parameters
    ----------
    master_seed:
        Deterministic master seed.
    n_inst:
        Number of challenge instances to emit.
    delta_range:
        Declared ``(lo, hi)`` magnitude range for the dark link.
    n_slots:
        Comb size.  ``2`` is fully supported; ``3`` is stubbed in
        :class:`CircuitFamily`.
    dark_prob:
        Probability an instance carries a dark link (50/50 by default).
    """

    def __init__(self, master_seed: int, n_inst: int,
                 delta_range: Tuple[float, float] = (0.2, 0.4),
                 n_slots: int = 2, dark_prob: float = 0.5) -> None:
        self.master_seed = int(master_seed)
        self.n_inst = int(n_inst)
        self.delta_range = (float(delta_range[0]), float(delta_range[1]))
        self.n_slots = int(n_slots)
        self.dark_prob = float(dark_prob)
        self._instances = self._draw()

    # numpy is optional here; use the stdlib Mersenne-Twister for portability.
    def _draw(self) -> List[ChallengeInstance]:
        import random
        rng = random.Random(self.master_seed)
        lo, hi = self.delta_range
        valid_slots = VALID_DARK_SLOTS_2 if self.n_slots == 2 else \
            (SLOT_INTER,)  # 3-slot inter positions randomized in a later revision
        out: List[ChallengeInstance] = []
        for i in range(self.n_inst):
            dark = rng.random() < self.dark_prob
            if dark:
                axis = rng.uniform(0.0, 2.0 * math.pi)
                delta = rng.uniform(lo, hi)
                slot = valid_slots[rng.randrange(len(valid_slots))]
            else:
                axis = 0.0
                delta = 0.0
                slot = None
            out.append(ChallengeInstance(
                instance_id=i, n_slots=self.n_slots,
                dark=dark, axis=axis, delta=delta, slot=slot))
        return out

    # ---- accessors ------------------------------------------------------ #
    @property
    def instances(self) -> List[ChallengeInstance]:
        return list(self._instances)

    def sealed_key(self) -> Dict:
        """The full answer key (to be written to the sealed JSON)."""
        return {
            "protocol": "c7_blind_benchmark",
            "master_seed": self.master_seed,
            "n_inst": self.n_inst,
            "n_slots": self.n_slots,
            "delta_range": list(self.delta_range),
            "dark_prob": self.dark_prob,
            "answer_key": [inst.answer_key() for inst in self._instances],
        }

    def sealed_key_sha256(self) -> str:
        return sha256_of_obj(self.sealed_key())

    def public_manifest(self) -> Dict:
        """Public manifest: records the sealed-key sha256, leaks no hidden field."""
        return {
            "protocol": "c7_blind_benchmark",
            "master_seed": self.master_seed,
            "n_inst": self.n_inst,
            "n_slots": self.n_slots,
            "delta_range": list(self.delta_range),
            "dark_prob": self.dark_prob,
            "sealed_key_sha256": self.sealed_key_sha256(),
            "instances": [inst.public_spec() for inst in self._instances],
        }

    # ---- sealing / verification ---------------------------------------- #
    def write_sealed_key(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.sealed_key()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(payload))
        return sha256_of_obj(payload)

    @staticmethod
    def verify_sealed_key(path: str, expected_sha256: str) -> Tuple[bool, Dict]:
        """Open a sealed key file, recompute its sha256, and compare."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        got = sha256_of_obj(payload)
        return (got == expected_sha256), payload


# --------------------------------------------------------------------------- #
# qiskit circuit families                                                     #
# --------------------------------------------------------------------------- #
def _prep_gates(qc, q, label: str) -> None:
    """Prepare qubit ``q`` in an IC Pauli-eigenstate (from |0>)."""
    if label == "|0>":
        pass
    elif label == "|1>":
        qc.x(q)
    elif label == "|+>":
        qc.h(q)
    elif label == "|->":
        qc.x(q); qc.h(q)
    elif label == "|+i>":
        qc.h(q); qc.s(q)
    elif label == "|-i>":
        qc.h(q); qc.sdg(q)
    else:
        raise ValueError(f"unknown prep {label!r}")


def _meas_basis_rotation(qc, q, pauli: str) -> None:
    """Rotate qubit ``q`` so a Z-measurement reads the ``pauli`` eigenvalue."""
    if pauli == "Z":
        pass
    elif pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.sdg(q); qc.h(q)
    else:
        raise ValueError(f"unknown measurement Pauli {pauli!r}")


@dataclass(frozen=True)
class Setting:
    """A named circuit setting (a single circuit + how to read it)."""
    family: str                 # "atomic" | "depth2" | "heldout_depth3"
    label: str                  # unique within (instance, family)
    readout: str                # "single" (1 qubit) | "pair" (2 qubits)
    meta: Tuple = ()            # e.g. (prep, slot_open, pauli) or (p, q)


class CircuitFamily:
    """Builds qiskit circuits for one challenge instance.

    Families
    --------
    * ``atomic`` -- the 48-statistic single-time IC causal-break tester family
      (marginal channel per slot; the memory is reset mid-circuit to enforce the
      causal break, so the inter-slot dark link is *structurally absent* and the
      family is exactly dark-invariant -> the marginal-QPT baseline is blind by
      Theorem 1).  36 circuits are measured (6 preps x 2 slots x {X,Y,Z}); the
      12 identity effects are the deterministic value +1 and need no shots.
    * ``depth2`` -- the retained-S1 depth-2 correlator family: S1 = S2 = |+>,
      all 9 Pauli pairs <P_S1 Q_S2>.  This is the certificate's decision family.
    * ``heldout_depth3`` -- a deeper (depth-3) validation family that re-couples
      S1 to the memory a second time; used for held-out confirmation only, never
      in the frozen decision.
    """

    def __init__(self, instance: ChallengeInstance) -> None:
        self.inst = instance
        if instance.n_slots not in (2, 3):
            raise ValueError("only 2-slot (full) and 3-slot (stub) supported")
        if instance.n_slots == 3:
            raise NotImplementedError(
                "3-slot comb circuit construction is stubbed; the 3-slot "
                "extension randomizes the dark link over two inter-slot memory "
                "links and is scheduled for a later revision (see module docs).")

    # ---- lazy qiskit import -------------------------------------------- #
    @staticmethod
    def _qk():
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        return QuantumCircuit, QuantumRegister, ClassicalRegister

    # ---- the dark link -------------------------------------------------- #
    def _apply_link(self, qc, where: str) -> None:
        """Apply the dark link on E at position ``where`` if this instance is dark
        and its declared slot matches ``where``."""
        inst = self.inst
        if not inst.dark:
            return
        if inst.slot != where:
            return
        # R(delta, axis) = exp(-i (delta/2)(cos axis * X + sin axis * Y)) on E.
        qc.r(inst.delta, inst.axis, E)

    # ---- family: atomic causal-break testers --------------------------- #
    def atomic_testers(self) -> List[Tuple[Setting, "object"]]:
        QuantumCircuit, QuantumRegister, ClassicalRegister = self._qk()
        out = []
        for slot_open in (1, 2):
            sys_q = S1 if slot_open == 1 else S2
            for prep in PREP_ORDER:
                for pauli in PAULI_1Q:
                    qr = QuantumRegister(4, "q")
                    cr = ClassicalRegister(1, "c")
                    qc = QuantumCircuit(qr, cr)
                    # fresh memory in |+>
                    qc.h(E)
                    _prep_gates(qc, sys_q, prep)
                    # couple the open slot's system qubit to the memory
                    qc.cz(sys_q, E)
                    # CAUSAL BREAK: reset the memory mid-circuit.  This is the
                    # marginal single-time channel of the open slot; the
                    # inter-slot dark link (which lives after this reset) never
                    # enters -> exact dark-invariance (Theorem 1).
                    qc.reset(E)
                    _meas_basis_rotation(qc, sys_q, pauli)
                    qc.measure(sys_q, cr[0])
                    setting = Setting("atomic", f"s{slot_open}_{prep}_{pauli}",
                                      "single", (prep, slot_open, pauli))
                    out.append((setting, qc))
        return out

    # ---- general depth-2 comb correlator circuit ----------------------- #
    def comb_correlator_circuit(self, prep_s1: str, prep_s2: str, p: str, q: str):
        """Build the depth-2 comb circuit with arbitrary S1/S2 preps and Pauli
        measurements, injecting the dark link at this instance's declared slot.

        Used by the certificate depth-2 family (|+>,|+>) and by the restricted-
        and full-PTT baselines (which scan input preparations).
        """
        QuantumCircuit, QuantumRegister, ClassicalRegister = self._qk()
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(E)                           # E = |+>
        _prep_gates(qc, S1, prep_s1)
        _prep_gates(qc, S2, prep_s2)
        self._apply_link(qc, SLOT_PREWRITE)
        qc.cz(S1, E)                      # write
        self._apply_link(qc, SLOT_INTER)
        qc.cz(S2, E)                      # read
        self._apply_link(qc, SLOT_POSTREAD)
        # discard E (never measured)
        _meas_basis_rotation(qc, S1, p)
        _meas_basis_rotation(qc, S2, q)
        qc.measure(S1, cr[0])
        qc.measure(S2, cr[1])
        return qc

    # ---- family: depth-2 retained-S1 correlators ----------------------- #
    def depth2_correlators(self) -> List[Tuple[Setting, "object"]]:
        out = []
        for (p, q) in PAULI_PAIRS:
            qc = self.comb_correlator_circuit("|+>", "|+>", p, q)
            setting = Setting("depth2", f"{p}{q}", "pair", (p, q))
            out.append((setting, qc))
        return out

    # ---- family: held-out deeper validation ---------------------------- #
    def heldout_depth3(self) -> List[Tuple[Setting, "object"]]:
        """A held-out deeper protocol: S1 re-couples to the memory around the read.

        Memory word (retained-S1 spectator applied three times, S2 once):

            CZ_{S1,E} , link , CZ_{S1,E} , CZ_{S2,E} , CZ_{S1,E}.

        The link stays at the physical inter-slot position; the experimenter adds
        two extra S1--E couplings as a deeper intervention.  This depth-3 word
        keeps the same clean null (only <X_S1 X_S2> = 1) and, unlike a single
        appended coupling, retains axis-invariant sensitivity to the dark link
        (min-over-axis off-null signal ~0.15 at delta = 0.2).  It is never used
        in the frozen decision -- held out purely for cross-validation.
        """
        QuantumCircuit, QuantumRegister, ClassicalRegister = self._qk()
        out = []
        for (p, q) in PAULI_PAIRS:
            qr = QuantumRegister(4, "q")
            cr = ClassicalRegister(2, "c")
            qc = QuantumCircuit(qr, cr)
            qc.h(E)
            _prep_gates(qc, S1, "|+>")
            _prep_gates(qc, S2, "|+>")
            qc.cz(S1, E)                  # write
            self._apply_link(qc, SLOT_INTER)
            qc.cz(S1, E)                  # deeper: S1 re-couple (pre-read)
            qc.cz(S2, E)                  # read
            qc.cz(S1, E)                  # deeper: S1 re-couple (post-read)
            _meas_basis_rotation(qc, S1, p)
            _meas_basis_rotation(qc, S2, q)
            qc.measure(S1, cr[0])
            qc.measure(S2, cr[1])
            setting = Setting("heldout_depth3", f"{p}{q}", "pair", (p, q))
            out.append((setting, qc))
        return out


# --------------------------------------------------------------------------- #
# correlator / expectation extraction from qiskit counts                       #
# --------------------------------------------------------------------------- #
def pair_correlator_from_counts(counts: Dict[str, int]) -> Tuple[float, int]:
    """<A_S1 B_S2> = <(-1)^{b0+b1}> from a 2-bit measurement, plus total shots.

    qiskit count keys are big-endian bitstrings over the classical register;
    ``cr[0]`` (S1) is the rightmost character, ``cr[1]`` (S2) the next one.
    """
    total = 0
    acc = 0.0
    for key, n in counts.items():
        bits = key.replace(" ", "")
        b0 = int(bits[-1])           # cr[0] = S1
        b1 = int(bits[-2])           # cr[1] = S2
        acc += ((-1) ** (b0 + b1)) * n
        total += n
    return (acc / total if total else 0.0), total


def single_expectation_from_counts(counts: Dict[str, int]) -> Tuple[float, int]:
    """<P> = <(-1)^b> from a 1-bit measurement, plus total shots."""
    total = 0
    acc = 0.0
    for key, n in counts.items():
        bits = key.replace(" ", "")
        b0 = int(bits[-1])
        acc += ((-1) ** b0) * n
        total += n
    return (acc / total if total else 0.0), total


# --------------------------------------------------------------------------- #
# resource ledger                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ResourceLedger:
    """Counts distinct circuit settings and total shots for a method.

    ``settings`` counts the distinct circuits actually executed.  ``shots`` is
    the total shot budget.  ``extra`` carries method-specific resource metrics
    (e.g. ``operation_basis_size`` = d^{4k} for full PTT, the exponential
    ceiling the resource theorem compares against, which may exceed the number
    of executed circuits because identity effects are deterministic).
    """
    method: str
    settings: int = 0
    shots: int = 0
    by_family: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def add(self, family: str, n_settings: int, shots_each: int) -> None:
        self.settings += n_settings
        self.shots += n_settings * shots_each
        self.by_family[family] = self.by_family.get(family, 0) + n_settings

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value

    def as_dict(self) -> Dict:
        return {
            "method": self.method,
            "distinct_settings": self.settings,
            "total_shots": self.shots,
            "settings_by_family": dict(self.by_family),
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------- #
# frozen estimator (freeze-then-analyze)                                       #
# --------------------------------------------------------------------------- #
DEFAULT_STEP1_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "research", "results", "step1_separator_power", "step1_separator_power.json")


class FrozenEstimator:
    """The analysis pipeline: depth-2 correlators -> per-instance dark verdict.

    ALL thresholds are derived from the Step-1 separator-power artifact
    (``research/results/step1_separator_power/step1_separator_power.json``).  The
    full config + its sha256 are frozen *before* any challenge data is seen.

    Decision rule (per instance)
    ----------------------------
    Measure the 9 retained-S1 depth-2 correlators ``<P_S1 Q_S2>`` (S1 = S2 =
    |+>).  For each pair in the *decision family* compute the z-score
    ``z_k = (Chat_k - null_k)/se_k`` with binomial standard error
    ``se_k = sqrt(max(eps, 1 - Chat_k^2)/N_shots)``.  Declare *dark* iff

        max_k |z_k|  >  z_gate,

    where ``z_gate`` is a pre-registered blind-run family-wise gate: Bonferroni
    over the ``M_family * N_inst_declared`` tests at experiment-wide error
    ``alpha_exp``.  ``N_shots`` is set from the Step-1 shot count at the declared
    minimum-detectable delta, scaled by the matched-power factor for the gate.
    Freezing z_gate to the declared instance count controls the whole blind
    run's false-positive rate, not one test's.

    Two config versions
    -------------------
    * **v1 (legacy)** -- ``signal_subspace=False`` and
      ``readout_assignment_error=0.0`` (the defaults).  The decision family is
      the full 8-pair off-null set ``OFFNULL_PAIRS`` and the gate is the pure
      shot-noise Bonferroni gate.  Byte-identical to the original freeze, so the
      simulator dry-run config and its hash are unchanged.
    * **v2 (hardened)** -- ``signal_subspace=True`` with a device
      ``readout_assignment_error``.  Two theory-driven hardenings, both decided
      and frozen *before* any real-hardware unblinding:

        1. SIGNAL-SUBSPACE DECISION.  The decision family is restricted to the
           two correlators that carry first-order dark signal for a link with
           axis anywhere in the x-y plane, ``SIGNAL_SUBSPACE_PAIRS =
           ["YX", "ZX"]`` (see the module-level derivation).  Bonferroni is
           taken over this reduced 2-correlator set.  The six pure-noise off-null
           pairs {XY, XZ, YY, YZ, ZY, ZZ} -- which caused the Marrakesh
           rehearsal false positive (null instance 6, <XY> z=4.17 > 3.78) -- are
           no longer scanned.
        2. DEVICE-AWARE GATE.  The gate is recalibrated for the device's readout
           assignment error instead of the dry-run's negligible-readout
           assumption.  A per-qubit assignment error ``eps`` attenuates a
           ``k``-qubit correlator by ``lambda = (1 - 2 eps)^k`` (k=2 for pair
           correlators); readout-mitigating (dividing the estimate by lambda)
           inflates the null standard error by ``1/lambda`` and hence the null
           variance by ``1/lambda^2``.  The device-aware gate is therefore

               z_gate = z_gate_shot / lambda,   lambda = (1 - 2 eps)^k,

           with ``z_gate_shot`` the 2-correlator shot-noise Bonferroni gate.
           The Step-1 matched-power target is preserved by recomputing the shot
           budget against this device-aware gate.
    """

    def __init__(self,
                 target_platform: str = "ibm_heron_r2",
                 min_detectable_delta: float = 0.2,
                 n_inst_declared: int = 20,
                 alpha_exp: float = 0.01,
                 step1_artifact: str = DEFAULT_STEP1_ARTIFACT,
                 signal_subspace: bool = False,
                 readout_assignment_error: float = 0.0,
                 readout_correlator_qubits: int = 2) -> None:
        self.target_platform = target_platform
        self.min_detectable_delta = float(min_detectable_delta)
        self.n_inst_declared = int(n_inst_declared)
        self.alpha_exp = float(alpha_exp)
        self.step1_artifact = step1_artifact
        self.signal_subspace = bool(signal_subspace)
        self.readout_assignment_error = float(readout_assignment_error)
        self.readout_correlator_qubits = int(readout_correlator_qubits)
        self._config = self._freeze()

    # ---- freeze --------------------------------------------------------- #
    def _load_step1(self) -> Dict:
        """Load the Step-1 artifact; fall back to the published constants."""
        if os.path.isfile(self.step1_artifact):
            with open(self.step1_artifact, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {
                "alpha": data["metadata"]["alpha"],
                "z_alpha_over_2": data["metadata"]["z_alpha_over_2"],
                "z_beta": data["metadata"]["z_beta"],
                "n_ref": self._n_ref_from(data),
                "artifact_sha256": sha256_of_file(self.step1_artifact),
                "source": "artifact",
            }
        # published fallback (documented in the Step-1 markdown)
        return {
            "alpha": 0.05,
            "z_alpha_over_2": 1.959963986120195,
            "z_beta": 1.644853625133699,
            "n_ref": 341,               # IBM Heron delta0=0.2, phi=0
            "artifact_sha256": None,
            "source": "fallback_constants",
        }

    def _n_ref_from(self, data: Dict) -> int:
        """Step-1 single-correlator N_exact at the declared min delta / phi=0."""
        plat = data["platforms"][self.target_platform]
        sweep0 = next(s for s in plat["sweep"] if abs(s["phi_kick"]) < 1e-12)
        rec = min(sweep0["per_delta0"],
                  key=lambda r: abs(r["delta0"] - self.min_detectable_delta))
        return int(rec["N_exact_binomial"])

    @staticmethod
    def _shot_budget(z_gate: float, z_beta: float, z_alpha2_step1: float,
                     n_ref: int) -> Tuple[float, float, int]:
        """Step-1 matched-power shot budget for a given gate.

        Scale the Step-1 single-correlator N up to the (tighter) gate, then round
        up to a power of two, flooring at 2048 for a clean, comfortable freeze.
        """
        power_factor = ((z_gate + z_beta) / (z_alpha2_step1 + z_beta)) ** 2
        n_needed = n_ref * power_factor
        n_shots = 1 << max(1, int(math.ceil(math.log2(max(2.0, n_needed)))))
        n_shots = max(n_shots, 2048)
        return power_factor, n_needed, n_shots

    def _freeze(self) -> Dict:
        s1 = self._load_step1()
        alpha_step1 = s1["alpha"]
        z_beta = s1["z_beta"]
        z_alpha2_step1 = s1["z_alpha_over_2"]

        if not self.signal_subspace and self.readout_assignment_error == 0.0:
            # ------------------- v1 (legacy) freeze ------------------------- #
            # Byte-identical to the original config so the simulator dry-run and
            # its committed hash are undisturbed.
            m_offnull = len(OFFNULL_PAIRS)
            n_tests = m_offnull * self.n_inst_declared
            z_gate = norm_ppf(1.0 - self.alpha_exp / (2.0 * n_tests))
            power_factor, _, n_shots = self._shot_budget(
                z_gate, z_beta, z_alpha2_step1, s1["n_ref"])
            return {
                "estimator": "c7_frozen_flat_extension_certificate",
                "decision_family": "depth2_retained_S1_correlators",
                "correlator_null": CORRELATOR_NULL,
                "offnull_pairs": OFFNULL_PAIRS,
                "m_offnull": m_offnull,
                "target_platform": self.target_platform,
                "min_detectable_delta": self.min_detectable_delta,
                "n_inst_declared": self.n_inst_declared,
                "alpha_exp": self.alpha_exp,
                "alpha_step1": alpha_step1,
                "z_alpha_over_2_step1": z_alpha2_step1,
                "z_beta": z_beta,
                "bonferroni_tests": n_tests,
                "z_gate": z_gate,
                "power_factor": power_factor,
                "n_ref_step1": s1["n_ref"],
                "n_shots": n_shots,
                "se_floor_var": 1e-6,
                "step1_artifact_sha256": s1["artifact_sha256"],
                "step1_source": s1["source"],
            }

        # --------------------- v2 (hardened) freeze ------------------------- #
        # (1) signal-subspace decision family (2-correlator Bonferroni)
        m_signal = len(SIGNAL_SUBSPACE_PAIRS)
        n_tests = m_signal * self.n_inst_declared
        z_gate_shot = norm_ppf(1.0 - self.alpha_exp / (2.0 * n_tests))
        # (2) device-aware gate: propagate the readout assignment error into the
        # correlator SE.  eps attenuates a k-qubit correlator by lambda =
        # (1-2 eps)^k; readout-mitigation inflates the null SE by 1/lambda and
        # the null variance by 1/lambda^2, so the gate rises by 1/lambda.
        eps = self.readout_assignment_error
        k = self.readout_correlator_qubits
        lambda_ro = (1.0 - 2.0 * eps) ** k
        z_gate = z_gate_shot / lambda_ro
        # keep the Step-1 matched-power target against the device-aware gate
        power_factor, _, n_shots = self._shot_budget(
            z_gate, z_beta, z_alpha2_step1, s1["n_ref"])
        return {
            "estimator": "c7_frozen_flat_extension_certificate",
            "config_version": 2,
            "decision_family": "depth2_signal_subspace_YX_ZX",
            "correlator_null": CORRELATOR_NULL,
            "offnull_pairs": OFFNULL_PAIRS,
            "signal_subspace_pairs": list(SIGNAL_SUBSPACE_PAIRS),
            "m_signal": m_signal,
            "target_platform": self.target_platform,
            "min_detectable_delta": self.min_detectable_delta,
            "n_inst_declared": self.n_inst_declared,
            "alpha_exp": self.alpha_exp,
            "alpha_step1": alpha_step1,
            "z_alpha_over_2_step1": z_alpha2_step1,
            "z_beta": z_beta,
            "bonferroni_tests": n_tests,
            "z_gate_shot": z_gate_shot,
            "readout_assignment_error": eps,
            "readout_correlator_qubits": k,
            "readout_attenuation_lambda": lambda_ro,
            "z_gate": z_gate,
            "power_factor": power_factor,
            "n_ref_step1": s1["n_ref"],
            "n_shots": n_shots,
            "se_floor_var": 1e-6,
            "step1_artifact_sha256": s1["artifact_sha256"],
            "step1_source": s1["source"],
            "changelog": (
                "v2 (2026-07-23): hardened after the ibm_marrakesh dress "
                "rehearsal (research/results/blind_benchmark_hardware_rehearsal) "
                "produced 1 false positive on 4 nulls (0.25 FP rate): null "
                "instance 6's off-signal <XY> correlator hit z=4.17 vs the v1 "
                "frozen z_gate=3.78. Root cause: the v1 statistic scanned all 8 "
                "off-null Pauli pairs, but the device median readout error "
                "(0.99%, ~8x the dry-run assumption) inflates the off-signal "
                "noise floor. Fix 1 (signal-subspace decision): restrict the "
                "decision family to the 2 first-order-signal correlators "
                "{YX, ZX} (Bonferroni over 2, not 8). Fix 2 (device-aware gate): "
                "recalibrate z_gate for the readout assignment error "
                "(z_gate = z_gate_shot / (1-2 eps)^2) instead of the dry-run "
                "negligible-readout assumption, preserving the Step-1 power "
                "target."),
        }

    # ---- frozen views --------------------------------------------------- #
    @property
    def config(self) -> Dict:
        return dict(self._config)

    @property
    def n_shots(self) -> int:
        return int(self._config["n_shots"])

    @property
    def z_gate(self) -> float:
        return float(self._config["z_gate"])

    def config_sha256(self) -> str:
        return sha256_of_obj(self._config)

    # ---- decision ------------------------------------------------------- #
    @property
    def decision_pairs(self) -> List[str]:
        """The Pauli-pair labels the max-|z| statistic scans.

        v2 (hardened): the ``SIGNAL_SUBSPACE_PAIRS`` = {YX, ZX} carrying
        first-order dark signal.  v1 (legacy): the full 8-pair off-null family.
        """
        return list(self._config.get("signal_subspace_pairs")
                    or self._config["offnull_pairs"])

    def instance_statistic(self, correlators: Dict[str, float]) -> Dict:
        """Compute the per-instance decision statistic from measured correlators.

        ``correlators`` maps each Pauli-pair label (e.g. ``"YX"``) to its
        estimate.  Returns the max |z|, the argmax pair, and per-pair z scores
        over the frozen decision family (:attr:`decision_pairs`).
        """
        n = self.n_shots
        eps = self._config["se_floor_var"]
        z_scores = {}
        for pair in self.decision_pairs:
            chat = float(correlators.get(pair, 0.0))
            null = CORRELATOR_NULL[pair]
            se = math.sqrt(max(eps, 1.0 - chat * chat) / n)
            z_scores[pair] = (chat - null) / se
        arg = max(z_scores, key=lambda k: abs(z_scores[k]))
        return {
            "max_abs_z": abs(z_scores[arg]),
            "argmax_pair": arg,
            "z_scores": z_scores,
        }

    def decide(self, correlators: Dict[str, float]) -> Dict:
        """Return the per-instance verdict dict."""
        stat = self.instance_statistic(correlators)
        detected = stat["max_abs_z"] > self.z_gate
        return {
            "detected": bool(detected),
            "max_abs_z": stat["max_abs_z"],
            "argmax_pair": stat["argmax_pair"],
            "z_gate": self.z_gate,
        }


__all__ = [
    "S1", "E", "S2", "SCRATCH",
    "PREP_ORDER", "EFFECT_ORDER", "PAULI_1Q", "PAULI_PAIRS",
    "CORRELATOR_NULL", "OFFNULL_PAIRS", "SIGNAL_SUBSPACE_PAIRS",
    "SLOT_PREWRITE", "SLOT_INTER", "SLOT_POSTREAD", "VALID_DARK_SLOTS_2",
    "canonical_json", "sha256_hex", "sha256_of_obj", "sha256_of_file", "norm_ppf",
    "ChallengeInstance", "ChallengeGenerator",
    "Setting", "CircuitFamily",
    "pair_correlator_from_counts", "single_expectation_from_counts",
    "ResourceLedger", "FrozenEstimator", "DEFAULT_STEP1_ARTIFACT",
]
