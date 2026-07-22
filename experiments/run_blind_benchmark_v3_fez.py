#!/usr/bin/env python
"""PROTOCOL v3 -- device-referenced DIFFERENTIAL decision on ibm_fez.

Context
-------
The C7 Step-2 blind benchmark on ibm_fez
(``research/results/blind_benchmark_hardware_fez/``) reproduced every planted
dark signal (power 1.00) and stayed blind on the marginal-QPT arm, BUT the
*unreferenced absolute* certificate gate false-positived on 2 of 3 nulls: the
device carries a fixed, sign-locked, single-axis ``YX`` null floor on the
(S1,E,S2)=(0,1,2) triple.  The follow-up 6-null confirmation
(``research/results/fez_null_confirmation/``) established that this floor is
DEVICE-INTRINSIC, not drift: pooled 12288-shot ``YX`` z = -6.20 (corr -0.056),
global sign agreement 1.00, within-run half-splits stable.  Its STRUCTURE (sign,
locus, axis) is fixed; its MAGNITUDE varies day to day (the confirmation day
pooled to corr -0.056 with the strongest instance at -0.103, while the v2 run
day had two nulls at corr -0.097 and -0.108 that broke the absolute gate).

Protocol v3 is the preregistered remedy (paper supplement, "Protocol-v3
differential null referencing", Eqs. differential-reference / differential-
variance).  For each challenge instance the SAME job also runs an interleaved
REFERENCE block -- the identical certificate circuits with NO planted link
(guaranteed null by construction), on the SAME physical qubits, in the SAME
session / calibration window.  The decision is made on the DIFFERENCE of the
challenge and reference correlators, which cancels any stable device-locked
single-axis floor while preserving the rotating two-component dark response.

Design (frozen BEFORE running; see the sealed manifest + frozen config)
-----------------------------------------------------------------------
* Blind challenge set: fresh master seed 20260729, 8 sealed instances,
  dark_prob 0.5, delta_range (0.2, 0.4).  Certificate arm ONLY.  Per instance:
  9 challenge correlators + 9 reference correlators, 1024 shots each.
  8 x (9 + 9) x 1024 = 147,456 shots.
* One JOB PER INSTANCE (18 PUBs: 9 challenge + 9 reference), PUB order
  interleaved by a deterministic per-job shuffle (O7).  Challenge and reference
  therefore share one calibration window and the same physical layout (VF2 is
  deterministic -> phys (0,1,2)).
* v3 differential estimator (frozen + hashed before submission):
    - decision vector  Delta C_hat_p = C_hat_p^challenge - C_hat_p^reference,
      p in {YX, ZX} (signal subspace);
    - conservative independent-block component variance
      var_diff_p = (1 - (C_ch_p)^2)/N_ch + (1 - (C_ref_p)^2)/N_ref;
    - decision statistic  z_diff_p = Delta C_hat_p / sqrt(var_diff_p);
    - detection iff  max_p |z_diff_p| > z_gate_diff.
  The device-aware family-wise DIFFERENTIAL GATE is derived as
      z_gate_diff = z_gate_shot / lambda_fez         (in z_diff units)
                  = sqrt(2) * z_gate_shot / lambda_fez   (in single-block-sigma units),
  where z_gate_shot = Phi^{-1}(1 - alpha_exp / (2 * m_signal * n_inst)) is the
  same 2-correlator Bonferroni shot gate as v2, and lambda_fez = (1 - 2 eps)^2
  is fez's OWN readout attenuation (eps = fez median measure assignment error
  from its calibration snapshot; NO marrakesh constant).  The "sqrt(2)
  adjustment" is exactly the combined-variance factor 2 already carried in the
  z_diff denominator (var_ch + var_ref = 2 * var_single when the two blocks
  match), so the two forms are the identical test; both numbers are recorded.
    - INCONCLUSIVE (reference-drift guard, preregistered): if the reference
      block's stabilizer witness |<XX>_ref| < 0.5 the differential null model is
      untrustworthy for that instance -> mark INCONCLUSIVE (neither detection
      nor clean null).  Under normal execution <XX>_ref ~ 0.9 and never fires.
* v2-absolute rescoring OF THE SAME DATA (for the paper's comparison): apply the
  frozen v2 fez certificate classifier (z_gate = 3.5599) to the CHALLENGE
  correlators only, both (a) as the frozen v2 estimator (est.decide, its native
  n_shots) and (b) with an honest z recomputed at the actual shot count.

Dress rehearsal (AerSimulator.from_backend(ibm_fez) + INJECTED synthetic floor)
-------------------------------------------------------------------------------
The device-calibrated AerSimulator carries fez's INCOHERENT noise but NOT the
COHERENT multitime YX floor (that is precisely why the null-confirmation
rehearsal expected YX ~ 0).  So the rehearsal injects the floor by hand: a
coherent, sign-fixed, single-axis R(theta_floor, 0) link on E at the inter slot,
added IDENTICALLY to every challenge AND reference block of every instance
(the floor is a device property present in both).  Two documented conditions:
  - floor corr -0.056: the confirmation-day POOLED floor (task-specified).  v3
    must give power 1.00 / FP 0.00 (floor cancelled).  This amplitude is BELOW
    the absolute gate at 1024 shots (z ~ 1.8), so v2-absolute stays clean too --
    documenting that the typical pooled floor alone does not breach the gate at
    this shot count.
  - floor corr -0.12: a worst-case STRONG-floor day (>= the -0.097/-0.108 that
    actually broke v2 on hardware; the confirmation established the magnitude
    varies day to day with fixed structure).  v3 must STILL give power 1.00 /
    FP 0.00, while the v2 absolute gate FALSE-POSITIVES on the nulls.  This is
    the contrast the paper needs, shown with an honest 1024-shot rescore.

The real run submits ONLY if the rehearsal shows exactly that contrast.  The
real ``--arm`` path injects NO floor: the device supplies its own.

SAFETY / how to run
-------------------
* Default (NO ``--arm``): full dress rehearsal against a local
  ``AerSimulator.from_backend(ibm_fez)`` with the injected floor(s).  Nothing is
  submitted.  Output under ``research/results/blind_benchmark_v3_fez/`` with a
  ``rehearsal_`` prefix.
* ``--arm``: the ONLY switch that submits to real ibm_fez.  (Project-lead
  authorization for one v3 submission is on record; quota budget < 80 QPU-s.)

    PYTHONPATH=src python experiments/run_blind_benchmark_v3_fez.py          # rehearsal
    PYTHONPATH=src python experiments/run_blind_benchmark_v3_fez.py --arm     # REAL fez
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from cyz_m.blind_benchmark import (
    ChallengeGenerator, ChallengeInstance, FrozenEstimator,
    _prep_gates, _meas_basis_rotation,
    PAULI_PAIRS, SIGNAL_SUBSPACE_PAIRS, SLOT_INTER,
    pair_correlator_from_counts, norm_ppf,
    canonical_json, sha256_of_obj, sha256_hex,
    S1, E, S2,
)

# --------------------------------------------------------------------------- #
# constants (frozen)                                                           #
# --------------------------------------------------------------------------- #
TARGET_BACKEND = "ibm_fez"
MASTER_SEED = 20260729                       # fresh v3 blind seed
N_INST = 8                                   # sealed blind instances
DELTA_RANGE = (0.2, 0.4)
N_SLOTS = 2
DARK_PROB = 0.5
CERT_SHOTS = 1536                            # per block per correlator setting
CERT_SETTINGS = 9                            # depth-2 retained-S1 correlators
REHEARSAL_SIM_SEED = 20260729                # deterministic AerSimulator noise seed
ALPHA_EXP = 0.01
N_INST_DECLARED = N_INST                     # 8 (family-wise gate declared count)

# fez readout assignment error, from ibm_fez's OWN calibration snapshot
# (research/results/blind_benchmark_hardware_fez/calibration_snapshot.json:
#  measure median_error = 0.00872802734375 over 156 qubits).  Device-level figure
# available BEFORE unblinding -> legitimate pre-registration.  NO marrakesh const.
FEZ_READOUT_ASSIGNMENT_ERROR = 0.00872802734375
FEZ_READOUT_PER_QUBIT_012 = {"0": 0.0118408203125, "1": 0.017822265625,
                             "2": 0.0052490234375}
READOUT_CORRELATOR_QUBITS = 2               # pair correlators <P_S1 Q_S2>

# frozen v2 fez certificate classifier parameters (reproduce z_gate = 3.5599 for
# the v2-absolute rescoring comparison).  v2 fez used marrakesh's median readout
# eps in its deployed gate -- reproduced here EXACTLY as deployed, only for the
# comparison rescore (the v3 gate above uses fez's own eps).
V2_READOUT_ASSIGNMENT_ERROR = 0.0098876953125
V2_N_INST_DECLARED = 8

# rehearsal injected-floor conditions (correlator units; sign-fixed negative).
REHEARSAL_FLOORS = (-0.056, -0.12)
POOLED_FLOOR = -0.056                        # confirmation-day pooled value
STRONG_FLOOR = -0.12                         # worst-case strong-floor day (contrast)

# reference-drift INCONCLUSIVE guard (frozen): stabilizer witness threshold.
XX_REF_INCONCLUSIVE_FLOOR = 0.5

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO, "research", "results", "blind_benchmark_v3_fez")

# QPU-seconds estimate: the standard per-shot/per-job model, PLUS an empirical
# 2-point fit from the two prior fez runs (null-confirmation: 110592 shots / 1
# job / 31 QPU-s; v2 blind: 331776 shots / 8 jobs / 106 QPU-s) ->
# qpu ~ 0.0002568 * shots + 2.6 * jobs.
PER_SHOT_OVERHEAD_S = (50e-6, 150e-6)
PER_JOB_OVERHEAD_S = (1.0, 3.0)
EMPIRICAL_PER_SHOT_S = 0.0002568
EMPIRICAL_PER_JOB_S = 2.6
QPU_BUDGET_S = 80                            # task-mandated ceiling for this run

_SE_FLOOR_VAR = 1e-6


# --------------------------------------------------------------------------- #
# frozen v3 differential gate                                                   #
# --------------------------------------------------------------------------- #
def v3_gate_config() -> Dict:
    """Derive + freeze the device-referenced differential gate for ibm_fez."""
    m_signal = len(SIGNAL_SUBSPACE_PAIRS)
    n_tests = m_signal * N_INST_DECLARED
    z_gate_shot = norm_ppf(1.0 - ALPHA_EXP / (2.0 * n_tests))
    eps = FEZ_READOUT_ASSIGNMENT_ERROR
    k = READOUT_CORRELATOR_QUBITS
    lambda_ro = (1.0 - 2.0 * eps) ** k
    # differential gate in z_diff (combined-variance) units:
    z_gate_diff = z_gate_shot / lambda_ro
    # equivalent single-block-sigma-units form (the literal "sqrt(2)-adjusted
    # shot gate x readout inflation"); identical test because sqrt(var_ch +
    # var_ref) = sqrt(2) * sigma_single when the two blocks match.
    z_gate_diff_singleblock = math.sqrt(2.0) * z_gate_shot / lambda_ro
    return {
        "estimator": "c7_v3_device_referenced_differential",
        "config_version": 3,
        "decision_family": "differential_signal_subspace_YX_ZX",
        "signal_subspace_pairs": list(SIGNAL_SUBSPACE_PAIRS),
        "m_signal": m_signal,
        "n_inst_declared": N_INST_DECLARED,
        "alpha_exp": ALPHA_EXP,
        "bonferroni_tests": n_tests,
        "z_gate_shot": z_gate_shot,
        "readout_source": "ibm_fez calibration snapshot (median measure error)",
        "readout_assignment_error_fez": eps,
        "readout_per_qubit_012": FEZ_READOUT_PER_QUBIT_012,
        "readout_correlator_qubits": k,
        "readout_attenuation_lambda_fez": lambda_ro,
        "z_gate_diff": z_gate_diff,
        "z_gate_diff_units": "z_diff = DeltaC / sqrt(var_ch + var_ref)",
        "z_gate_diff_singleblock_sigma_units": z_gate_diff_singleblock,
        "cert_shots_per_block": CERT_SHOTS,
        "se_floor_var": _SE_FLOOR_VAR,
        "component_variance": ("var_diff_p = (1-(C_ch_p)^2)/N_ch + "
                               "(1-(C_ref_p)^2)/N_ref  (paper Eq. "
                               "differential-variance)"),
        "inconclusive_rule": (f"reference-drift guard: |<XX>_ref| < "
                              f"{XX_REF_INCONCLUSIVE_FLOOR} -> INCONCLUSIVE"),
        "xx_ref_inconclusive_floor": XX_REF_INCONCLUSIVE_FLOOR,
        "derivation": (
            "z_gate_shot = Phi^-1(1 - alpha_exp/(2*m_signal*n_inst)); "
            "lambda_fez = (1-2*eps_fez)^k, k=2; differential gate = "
            "z_gate_shot/lambda_fez in z_diff units, equivalently "
            "sqrt(2)*z_gate_shot/lambda_fez in single-block-sigma units. The "
            "sqrt(2) is the combined-variance factor in the z_diff denominator; "
            "eps_fez is fez's own median readout error (no marrakesh constant)."),
    }


def v3_config_sha256(cfg: Dict) -> str:
    return sha256_of_obj(cfg)


# --------------------------------------------------------------------------- #
# hypotheses block (hashed into the manifest before execution)                 #
# --------------------------------------------------------------------------- #
def hypotheses_block(v3cfg: Dict) -> Dict:
    return {
        "protocol": "blind_benchmark_v3_device_referenced_differential",
        "target_backend": TARGET_BACKEND,
        "question": ("Does device-referenced differential decision restore "
                     "specificity on ibm_fez (whose fixed single-axis YX null "
                     "floor breaks the unreferenced absolute gate)?"),
        "master_seed": MASTER_SEED,
        "n_inst": N_INST,
        "dark_prob": DARK_PROB,
        "delta_range": list(DELTA_RANGE),
        "cert_shots_per_block": CERT_SHOTS,
        "decision": ("per instance, per pair p in {YX,ZX}: z_diff_p = "
                     "(C_ch_p - C_ref_p)/sqrt(var_ch_p+var_ref_p); detect iff "
                     f"max_p |z_diff_p| > z_gate_diff = {v3cfg['z_gate_diff']:.6f}"),
        "z_gate_diff": v3cfg["z_gate_diff"],
        "z_gate_shot": v3cfg["z_gate_shot"],
        "readout_assignment_error_fez": v3cfg["readout_assignment_error_fez"],
        "v2_absolute_gate": 3.559932645771468,
        "H_v3_specific": ("v3 differential: power 1.00 on darks AND FP 0.00 on "
                          "nulls -- the fixed floor cancels in the difference"),
        "H_v2_fails": ("the unreferenced absolute gate (3.5599) false-positives "
                       "on >= 1 null under a device-locked floor at its "
                       "v2-breaking magnitude"),
        "rehearsal_injected_floors_corr": list(REHEARSAL_FLOORS),
        "rehearsal_floor_provenance": {
            str(POOLED_FLOOR): "confirmation-day pooled floor (task-specified)",
            str(STRONG_FLOOR): ("worst-case strong-floor day >= the -0.097/-0.108 "
                                "that broke v2 on hardware"),
        },
        "submit_gate": ("submit real fez ONLY if rehearsal shows v3 power 1.00 / "
                        "FP 0.00 in BOTH floor conditions AND v2-absolute FPs on "
                        ">=1 null in the strong-floor condition"),
    }


# --------------------------------------------------------------------------- #
# circuit builder: floor-injected depth-2 correlators                          #
# --------------------------------------------------------------------------- #
def build_correlator_circuits(instance: ChallengeInstance,
                              theta_floor: Optional[float]):
    """Build the 9 depth-2 retained-S1 correlator circuits for ``instance``,
    optionally injecting a coherent single-axis floor R(theta_floor, 0) on E at
    the inter slot.

    With ``theta_floor is None`` and the instance's real dark/slot, the circuits
    are byte-identical to ``CircuitFamily(instance).depth2_correlators()`` (the
    established certificate circuits used on hardware).  The floor gate, when
    present, sits at the same inter-slot memory position as a planted link, so it
    reproduces a device-locked coherent multitime bias affecting both blocks.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    out = []
    for (p, q) in PAULI_PAIRS:
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.h(E)                                        # E = |+>
        _prep_gates(qc, S1, "|+>")
        _prep_gates(qc, S2, "|+>")
        # (prewrite planted link would go here; the blind set only uses inter)
        if instance.dark and instance.slot == "prewrite":
            qc.r(instance.delta, instance.axis, E)
        qc.cz(S1, E)                                   # write
        if instance.dark and instance.slot == SLOT_INTER:
            qc.r(instance.delta, instance.axis, E)     # planted inter-slot link
        if theta_floor is not None:
            qc.r(theta_floor, 0.0, E)                  # injected device floor (YX axis)
        qc.cz(S2, E)                                   # read
        if instance.dark and instance.slot == "postread":
            qc.r(instance.delta, instance.axis, E)
        _meas_basis_rotation(qc, S1, p)
        _meas_basis_rotation(qc, S2, q)
        qc.measure(S1, cr[0])
        qc.measure(S2, cr[1])
        out.append((f"{p}{q}", qc))
    return out


def reference_instance(instance_id: int) -> ChallengeInstance:
    """A guaranteed-null instance (NO planted link) with the same id/layout."""
    return ChallengeInstance(instance_id=instance_id, n_slots=N_SLOTS,
                             dark=False, axis=0.0, delta=0.0, slot=None)


# --------------------------------------------------------------------------- #
# correlator / z helpers                                                        #
# --------------------------------------------------------------------------- #
def corr_from_bitstrings(bitstrings: List[str]) -> Tuple[float, int]:
    return pair_correlator_from_counts(Counter(bitstrings))


def var_of_corr(chat: float, n: int) -> float:
    return max(_SE_FLOOR_VAR, 1.0 - chat * chat) / n


def z_from_corr(chat: float, n: int, null: float = 0.0) -> float:
    return (chat - null) / math.sqrt(var_of_corr(chat, n))


# --------------------------------------------------------------------------- #
# physical-qubit extraction (verifies deterministic VF2 layout -> phys 0,1,2)   #
# --------------------------------------------------------------------------- #
def physical_qubits(isa_circ) -> Dict:
    fil = list(isa_circ.layout.final_index_layout(filter_ancillas=True))
    used = set()
    two_q = []
    for inst in isa_circ.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        idx = [isa_circ.find_bit(qb).index for qb in inst.qubits]
        used.update(idx)
        if name in ("cz", "cx", "ecr", "rzz"):
            two_q.append(tuple(idx))
    return {
        "phys_S1": fil[S1] if len(fil) > S1 else None,
        "phys_E": fil[E] if len(fil) > E else None,
        "phys_S2": fil[S2] if len(fil) > S2 else None,
        "final_index_layout": fil,
        "active_physical_qubits": sorted(used),
        "two_qubit_edges": two_q,
    }


# --------------------------------------------------------------------------- #
# backend acquisition + rehearsal simulator                                    #
# --------------------------------------------------------------------------- #
def fetch_real_backend(target_backend: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    return service.backend(target_backend), service


def make_sampler(mode):
    from qiskit_ibm_runtime import SamplerV2
    return SamplerV2(mode=mode)


def rehearsal_backend(target_backend: str):
    from qiskit_aer import AerSimulator
    backend, _ = fetch_real_backend(target_backend)
    sim = AerSimulator.from_backend(backend)
    sim.set_options(seed_simulator=REHEARSAL_SIM_SEED)   # deterministic, auditable
    return backend, sim, (f"AerSimulator.from_backend({backend.name}) "
                          f"seed={REHEARSAL_SIM_SEED}")


def calibration_snapshot(backend) -> Dict:
    from cyz_m.hardware_runner import calibration_snapshot as _cs
    return _cs(backend)


# --------------------------------------------------------------------------- #
# QPU-seconds preflight                                                         #
# --------------------------------------------------------------------------- #
def qpu_seconds_estimate(total_shots: int, job_count: int) -> Dict:
    lo = total_shots * PER_SHOT_OVERHEAD_S[0] + job_count * PER_JOB_OVERHEAD_S[0]
    hi = total_shots * PER_SHOT_OVERHEAD_S[1] + job_count * PER_JOB_OVERHEAD_S[1]
    emp = total_shots * EMPIRICAL_PER_SHOT_S + job_count * EMPIRICAL_PER_JOB_S
    return {"low_s": lo, "high_s": hi, "empirical_s": emp, "budget_s": QPU_BUDGET_S,
            "under_budget": emp < QPU_BUDGET_S and hi < QPU_BUDGET_S,
            "note": ("empirical fit from the two prior fez runs "
                     "(0.0002568 s/shot + 2.6 s/job) is the authoritative "
                     "estimate; both it and the model high estimate stay < 80")}


# --------------------------------------------------------------------------- #
# build + run one instance's job (9 challenge + 9 reference PUBs, interleaved)   #
# --------------------------------------------------------------------------- #
def build_and_run_instance(instance, transpile_backend, sampler, armed,
                           theta_floor):
    """Return (challenge_bits, reference_bits, layout, job_record) for one
    instance.  theta_floor is None on the armed path (device supplies the floor);
    a coherent injected floor on the rehearsal path."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1,
                                      target=transpile_backend.target)

    ref_inst = reference_instance(instance.instance_id)
    ch_circ = build_correlator_circuits(instance, theta_floor)
    ref_circ = build_correlator_circuits(ref_inst, theta_floor)

    ch_isa = pm.run([qc for _, qc in ch_circ])
    ref_isa = pm.run([qc for _, qc in ref_circ])

    layout = {"challenge": {}, "reference": {}}
    pubs = []                                     # (isa, None, shots)
    pub_index: List[Tuple[str, str]] = []         # (block, pair_label)
    for (lab, _), isa in zip(ch_circ, ch_isa):
        layout["challenge"][lab] = physical_qubits(isa)
        pubs.append((isa, None, CERT_SHOTS))
        pub_index.append(("challenge", lab))
    for (lab, _), isa in zip(ref_circ, ref_isa):
        layout["reference"][lab] = physical_qubits(isa)
        pubs.append((isa, None, CERT_SHOTS))
        pub_index.append(("reference", lab))

    # O7 deterministic interleave of this job's 18 PUBs.
    shuffle_seed = int(sha256_hex(f"{MASTER_SEED}:job{instance.instance_id}")[:8], 16)
    rng = random.Random(shuffle_seed)
    order = list(range(len(pubs)))
    rng.shuffle(order)
    shuffled_pubs = [pubs[k] for k in order]
    shuffled_index = [pub_index[k] for k in order]

    submit_ts = datetime.now(timezone.utc).isoformat()
    job_obj = sampler.run(shuffled_pubs)
    job_id = job_obj.job_id()
    result = job_obj.result()
    recv_ts = datetime.now(timezone.utc).isoformat()
    try:
        usage = job_obj.usage()
    except Exception:
        usage = None

    ch_bits: Dict[str, List[str]] = {}
    ref_bits: Dict[str, List[str]] = {}
    for i, (block, lab) in enumerate(shuffled_index):
        data = result[i].data
        ba = getattr(data, "c", None)
        if ba is None:
            for f in list(getattr(data, "__dict__", {})) or []:
                cand = getattr(data, f)
                if hasattr(cand, "get_bitstrings"):
                    ba = cand
                    break
        bits = ba.get_bitstrings()
        (ch_bits if block == "challenge" else ref_bits)[lab] = bits

    job_record = {
        "instance_id": instance.instance_id, "job_id": job_id, "armed": armed,
        "n_pubs": len(pubs), "shuffle_seed": shuffle_seed, "pub_order": order,
        "submit_ts_utc": submit_ts, "result_ts_utc": recv_ts,
        "usage_qpu_s": usage,
    }
    return ch_bits, ref_bits, layout, job_record


# --------------------------------------------------------------------------- #
# per-instance analysis                                                         #
# --------------------------------------------------------------------------- #
def analyse_instance(instance, ch_bits, ref_bits, layout, v3cfg, v2_est) -> Dict:
    iid = instance.instance_id

    def block_corr(bits_map):
        out = {}
        for lab in ("XX",) + tuple(SIGNAL_SUBSPACE_PAIRS):
            bits = bits_map[lab]
            c, n = corr_from_bitstrings(bits)
            out[lab] = {"corr": c, "n": n}
        return out

    ch = block_corr(ch_bits)
    ref = block_corr(ref_bits)

    # v3 differential per signal pair
    z_diff = {}
    for p in SIGNAL_SUBSPACE_PAIRS:
        c_ch, n_ch = ch[p]["corr"], ch[p]["n"]
        c_ref, n_ref = ref[p]["corr"], ref[p]["n"]
        v_ch = var_of_corr(c_ch, n_ch)
        v_ref = var_of_corr(c_ref, n_ref)
        dc = c_ch - c_ref
        z = dc / math.sqrt(v_ch + v_ref)
        z_diff[p] = {"delta_corr": dc, "c_challenge": c_ch, "c_reference": c_ref,
                     "var_challenge": v_ch, "var_reference": v_ref, "z_diff": z}
    arg = max(z_diff, key=lambda k: abs(z_diff[k]["z_diff"]))
    max_abs_zdiff = abs(z_diff[arg]["z_diff"])

    # INCONCLUSIVE reference-drift guard
    xx_ref = ref["XX"]["corr"]
    inconclusive = abs(xx_ref) < v3cfg["xx_ref_inconclusive_floor"]
    v3_detected = (not inconclusive) and (max_abs_zdiff > v3cfg["z_gate_diff"])

    # v2-absolute rescoring on the CHALLENGE correlators only
    ch_corr_map = {p: ch[p]["corr"] for p in SIGNAL_SUBSPACE_PAIRS}
    v2_frozen = v2_est.decide(ch_corr_map)      # native n_shots (2048)
    honest_z = {p: z_from_corr(ch[p]["corr"], ch[p]["n"], 0.0)
                for p in SIGNAL_SUBSPACE_PAIRS}
    honest_arg = max(honest_z, key=lambda k: abs(honest_z[k]))
    v2_honest_detected = abs(honest_z[honest_arg]) > v2_est.z_gate

    lay = layout["challenge"][arg]
    return {
        "instance_id": iid,
        "dark": instance.dark,
        "phys_triple": (lay["phys_S1"], lay["phys_E"], lay["phys_S2"]),
        # v3
        "v3_detected": bool(v3_detected),
        "v3_inconclusive": bool(inconclusive),
        "v3_argmax_pair": arg,
        "v3_max_abs_zdiff": max_abs_zdiff,
        "z_diff": z_diff,
        "xx_reference": xx_ref,
        "xx_challenge": ch["XX"]["corr"],
        # v2-absolute rescore
        "v2_frozen_detected": bool(v2_frozen["detected"]),
        "v2_frozen_max_abs_z": v2_frozen["max_abs_z"],
        "v2_frozen_argmax": v2_frozen["argmax_pair"],
        "v2_honest_detected": bool(v2_honest_detected),
        "v2_honest_max_abs_z": abs(honest_z[honest_arg]),
        "v2_honest_argmax": honest_arg,
        "v2_honest_z": honest_z,
        "challenge_corr": {p: ch[p]["corr"] for p in ("XX",) + tuple(SIGNAL_SUBSPACE_PAIRS)},
        "reference_corr": {p: ref[p]["corr"] for p in ("XX",) + tuple(SIGNAL_SUBSPACE_PAIRS)},
    }


def score_family(per_instance, key: str) -> Dict:
    """power/FP for a detected-key ('v3_detected' | 'v2_frozen_detected' |
    'v2_honest_detected'); INCONCLUSIVE instances excluded from both counts."""
    usable = [p for p in per_instance if not p.get("v3_inconclusive", False)] \
        if key == "v3_detected" else per_instance
    n_dark = sum(1 for p in usable if p["dark"])
    n_null = sum(1 for p in usable if not p["dark"])
    tp = sum(1 for p in usable if p["dark"] and p[key])
    fp = sum(1 for p in usable if (not p["dark"]) and p[key])
    return {
        "n_dark": n_dark, "n_null": n_null,
        "true_positives": tp, "false_positives": fp,
        "power": (tp / n_dark) if n_dark else float("nan"),
        "fp_rate": (fp / n_null) if n_null else float("nan"),
        "n_inconclusive": sum(1 for p in per_instance
                              if p.get("v3_inconclusive", False)),
    }


# --------------------------------------------------------------------------- #
# run one full pass (all 8 instances) for a given floor condition               #
# --------------------------------------------------------------------------- #
def run_pass(instances, transpile_backend, sampler, armed, theta_floor,
             v3cfg, v2_est) -> Dict:
    per_instance = []
    job_records = []
    layouts = {}
    for inst in instances:
        ch_bits, ref_bits, layout, jr = build_and_run_instance(
            inst, transpile_backend, sampler, armed, theta_floor)
        per_instance.append(
            analyse_instance(inst, ch_bits, ref_bits, layout, v3cfg, v2_est))
        job_records.append(jr)
        layouts[inst.instance_id] = layout
    scores = {
        "v3_differential": score_family(per_instance, "v3_detected"),
        "v2_absolute_frozen": score_family(per_instance, "v2_frozen_detected"),
        "v2_absolute_honest": score_family(per_instance, "v2_honest_detected"),
    }
    total_usage = None
    usages = [jr["usage_qpu_s"] for jr in job_records
              if isinstance(jr["usage_qpu_s"], (int, float))]
    if usages:
        total_usage = sum(usages)
    return {"floor_corr": theta_floor, "per_instance": per_instance,
            "scores": scores, "job_records": job_records, "layouts": layouts,
            "total_usage_qpu_s": total_usage}


# --------------------------------------------------------------------------- #
# markdown                                                                      #
# --------------------------------------------------------------------------- #
def _fmt(x, p=2):
    return (f"{x:+.{p}f}" if isinstance(x, (int, float)) else str(x))


def score_table_md(scores) -> List[str]:
    md = ["| classifier | power | FP rate | TP/dark | FP/null | inconclusive |",
          "|---|---|---|---|---|---|"]
    order = [("v3_differential", "v3 device-referenced differential"),
             ("v2_absolute_honest", "v2 absolute gate (honest N rescore)"),
             ("v2_absolute_frozen", "v2 absolute gate (frozen v2 estimator)")]
    for k, note in order:
        s = scores[k]
        md.append(f"| {note} | {s['power']:.2f} | {s['fp_rate']:.2f} | "
                  f"{s['true_positives']}/{s['n_dark']} | "
                  f"{s['false_positives']}/{s['n_null']} | {s['n_inconclusive']} |")
    return md


def per_instance_table_md(per_instance) -> List[str]:
    md = ["| inst | truth | phys | v3 z_diff(YX,ZX) | max|z_diff| | v3? | "
          "v2 |z|(honest) | v2? | <XX>ref |", "|---|---|---|---|---|---|---|---|---|"]
    for p in per_instance:
        zy = p["z_diff"]["YX"]["z_diff"]
        zz = p["z_diff"]["ZX"]["z_diff"]
        truth = "dark" if p["dark"] else "null"
        v3 = "INCONCL" if p["v3_inconclusive"] else ("DET" if p["v3_detected"] else "-")
        v2 = "DET" if p["v2_honest_detected"] else "-"
        md.append(f"| {p['instance_id']} | {truth} | {p['phys_triple']} | "
                  f"({zy:+.2f},{zz:+.2f}) | {p['v3_max_abs_zdiff']:.2f} | {v3} | "
                  f"{p['v2_honest_max_abs_z']:.2f} | {v2} | {p['xx_reference']:+.3f} |")
    return md


def build_markdown(meta, v3cfg, passes, submit_decision) -> str:
    md: List[str] = []
    tag = "REHEARSAL" if not meta["armed"] else "HARDWARE"
    md.append(f"# Blind benchmark PROTOCOL v3 (device-referenced differential) -- {tag} run")
    md.append("")
    md.append(f"- Mode: **{tag}** ({meta['exec_source']})")
    md.append(f"- Date: {meta['date']}  |  target: {meta['target_backend']}")
    md.append(f"- Master seed: {meta['master_seed']}  |  qiskit {meta['qiskit_version']}, "
              f"qiskit-ibm-runtime {meta['ibm_runtime_version']}, aer {meta['aer_version']}")
    md.append(f"- Blind set: {N_INST} instances (dark_prob {DARK_PROB}), certificate "
              f"arm only, {N_INST}x(9 challenge + 9 reference)x{CERT_SHOTS} = "
              f"{meta['total_shots']} shots, 1 job/instance ({N_INST} jobs x 18 PUBs)")
    md.append(f"- sealed_key_sha256: `{meta['sealed_key_sha256']}`  |  unblind "
              f"verified: **{meta['unblind_verified']}**")
    md.append(f"- v3 estimator config_sha256: `{meta['v3_config_sha256']}`  |  "
              f"hypotheses_sha256: `{meta['hypotheses_sha256']}`  |  manifest_sha256: "
              f"`{meta['manifest_sha256']}` (all sealed pre-exec)")
    md.append("")
    md.append("## Frozen v3 differential gate")
    md.append("")
    md.append(f"- z_gate_shot (Bonferroni over {v3cfg['m_signal']}x{v3cfg['n_inst_declared']} "
              f"= {v3cfg['bonferroni_tests']} tests, alpha_exp {v3cfg['alpha_exp']}): "
              f"**{v3cfg['z_gate_shot']:.4f}**")
    md.append(f"- fez readout eps (median measure, from fez calibration snapshot): "
              f"{v3cfg['readout_assignment_error_fez']}  -> lambda_fez = "
              f"(1-2eps)^2 = {v3cfg['readout_attenuation_lambda_fez']:.6f}")
    md.append(f"- **differential gate z_gate_diff = z_gate_shot/lambda_fez = "
              f"{v3cfg['z_gate_diff']:.4f}** (z_diff units); equivalently "
              f"sqrt(2)*z_gate_shot/lambda_fez = "
              f"{v3cfg['z_gate_diff_singleblock_sigma_units']:.4f} in "
              f"single-block-sigma units (identical test)")
    md.append(f"- component variance: {v3cfg['component_variance']}")
    md.append(f"- INCONCLUSIVE: {v3cfg['inconclusive_rule']}")
    md.append(f"- v2-absolute comparison gate (frozen fez v2 classifier): "
              f"z_gate = {meta['v2_z_gate']:.4f}")
    md.append("")
    for pr in passes:
        floor = pr["floor_corr"]
        if floor is None:
            head = "## Results -- REAL device floor (no injection)"
        else:
            prov = ("confirmation-day POOLED floor" if abs(floor - POOLED_FLOOR) < 1e-9
                    else ("worst-case STRONG-floor day (v2-breaking; the contrast)"
                          if abs(floor - STRONG_FLOOR) < 1e-9 else "injected floor"))
            head = f"## Rehearsal condition -- injected YX floor corr = {floor:+.3f} ({prov})"
        md.append(head)
        md.append("")
        if pr.get("total_usage_qpu_s") is not None:
            md.append(f"- QPU-s this pass (sum job.usage()): **{pr['total_usage_qpu_s']}**")
            md.append("")
        md += score_table_md(pr["scores"])
        md.append("")
        md += per_instance_table_md(pr["per_instance"])
        md.append("")
    md.append("## Submit decision")
    md.append("")
    md.append(f"- {submit_decision['statement']}")
    md.append(f"- contrast satisfied: **{submit_decision['contrast_satisfied']}**")
    md.append("")
    md.append("## QPU usage / budget")
    md.append("")
    q = meta["qpu_estimate"]
    md.append(f"- preflight estimate: model {q['low_s']:.1f}-{q['high_s']:.1f} s, "
              f"empirical-fit **{q['empirical_s']:.1f} s** (budget {QPU_BUDGET_S} s, "
              f"under_budget {q['under_budget']})")
    if meta.get("quota_consumed_s") is not None:
        md.append(f"- quota consumed (svc.usage delta): **{meta['quota_consumed_s']} s**  "
                  f"|  remaining after: {meta.get('quota_remaining_after_s')} s")
    md.append("")
    md.append("## One-sentence result (paper's marked slot)")
    md.append("")
    md.append(f"> {submit_decision['paper_sentence']}")
    md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# submit-gate evaluation                                                        #
# --------------------------------------------------------------------------- #
def evaluate_contrast(passes, armed) -> Dict:
    """Rehearsal submit gate: v3 power 1.00 / FP 0.00 in BOTH floor conditions
    AND v2-absolute FP >= 1 null in the strong-floor condition."""
    if armed:
        # real run: report v3 result (device provides floor)
        pr = passes[0]
        s3 = pr["scores"]["v3_differential"]
        s2h = pr["scores"]["v2_absolute_honest"]
        contrast = (s3["fp_rate"] == 0.0 and s3["power"] == 1.0)
        sent = (f"On ibm_fez, device-referenced differential decision (protocol "
                f"v3) recovered full specificity -- power {s3['power']:.2f} on "
                f"{s3['n_dark']} planted-dark instances and false-positive rate "
                f"{s3['fp_rate']:.2f} on {s3['n_null']} sealed nulls -- while the "
                f"unreferenced absolute gate false-positived on "
                f"{s2h['false_positives']}/{s2h['n_null']} of the same nulls.")
        return {"contrast_satisfied": bool(contrast),
                "statement": ("REAL fez run: v3 differential scored "
                              f"power={s3['power']:.2f} FP={s3['fp_rate']:.2f}; "
                              f"v2-absolute FP={s2h['false_positives']}/{s2h['n_null']}."),
                "paper_sentence": sent}
    # rehearsal gate
    v3_all_clean = all(pr["scores"]["v3_differential"]["power"] == 1.0 and
                       pr["scores"]["v3_differential"]["fp_rate"] == 0.0
                       for pr in passes)
    strong = next((pr for pr in passes
                   if pr["floor_corr"] is not None
                   and abs(pr["floor_corr"] - STRONG_FLOOR) < 1e-9), None)
    v2_fails = (strong is not None and
                strong["scores"]["v2_absolute_honest"]["false_positives"] >= 1)
    contrast = bool(v3_all_clean and v2_fails)
    if strong is not None:
        s3 = strong["scores"]["v3_differential"]
        s2 = strong["scores"]["v2_absolute_honest"]
        sent = (f"Under an injected device-locked YX null floor, protocol-v3 "
                f"differential decision held power {s3['power']:.2f} / "
                f"false-positive rate {s3['fp_rate']:.2f} while the unreferenced "
                f"absolute gate false-positived on {s2['false_positives']}/"
                f"{s2['n_null']} nulls, confirming the differential cancels the "
                f"fixed floor the absolute gate cannot.")
    else:
        sent = "n/a"
    return {"contrast_satisfied": contrast,
            "statement": (f"Rehearsal: v3 power=1.00/FP=0.00 in all floor "
                          f"conditions = {v3_all_clean}; v2-absolute FP>=1 null in "
                          f"strong-floor condition = {v2_fails}. "
                          f"{'PROCEED to --arm.' if contrast else 'DO NOT submit.'}"),
            "paper_sentence": sent}


# --------------------------------------------------------------------------- #
# CLI + main                                                                    #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Blind benchmark protocol v3 (fez)")
    ap.add_argument("--arm", action="store_true",
                    help="REQUIRED to submit to real ibm_fez. Without it the full "
                         "pipeline runs against a local AerSimulator.from_backend "
                         "with the injected floor(s).")
    ap.add_argument("--target-backend", default=TARGET_BACKEND)
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    import qiskit
    import qiskit_aer
    import qiskit_ibm_runtime

    armed = bool(args.arm)
    prefix = "" if armed else "rehearsal_"
    os.makedirs(OUT_DIR, exist_ok=True)

    # frozen v3 config + hypotheses (sealed pre-exec)
    v3cfg = v3_gate_config()
    v3cfg_sha = v3_config_sha256(v3cfg)
    hyp = hypotheses_block(v3cfg)
    hyp_sha = sha256_of_obj(hyp)

    # blind challenge set + sealed key
    gen = ChallengeGenerator(MASTER_SEED, N_INST, DELTA_RANGE, N_SLOTS, DARK_PROB)
    instances = gen.instances
    sealed_key = gen.sealed_key()
    sealed_sha = gen.sealed_key_sha256()

    total_shots = N_INST * 2 * CERT_SETTINGS * CERT_SHOTS
    qest = qpu_seconds_estimate(total_shots, job_count=N_INST)

    print("### PROTOCOL v3 preflight")
    print(f"- {N_INST} instances x (9 challenge + 9 reference) x {CERT_SHOTS} shots "
          f"= {total_shots} shots, {N_INST} jobs")
    print(f"- QPU-seconds: model {qest['low_s']:.1f}-{qest['high_s']:.1f}s, "
          f"empirical {qest['empirical_s']:.1f}s (budget {QPU_BUDGET_S}s) -> "
          f"under_budget={qest['under_budget']}")
    print(f"- v3 gate z_gate_diff={v3cfg['z_gate_diff']:.4f} (z_diff units); "
          f"fez eps={v3cfg['readout_assignment_error_fez']}")
    assert qest["under_budget"], (
        f"preflight QPU-s estimate exceeds the {QPU_BUDGET_S}s budget")

    # v2 frozen fez classifier (for the absolute rescoring comparison)
    v2_est = FrozenEstimator(
        target_platform="ibm_heron_r2", min_detectable_delta=DELTA_RANGE[0],
        n_inst_declared=V2_N_INST_DECLARED, alpha_exp=ALPHA_EXP,
        signal_subspace=True, readout_assignment_error=V2_READOUT_ASSIGNMENT_ERROR,
        readout_correlator_qubits=READOUT_CORRELATOR_QUBITS)

    # choose path (SAFETY: --arm is the ONLY hardware gate)
    if armed:
        print(f"\n[ARMED] submitting to REAL hardware {args.target_backend}.")
        backend, service = fetch_real_backend(args.target_backend)
        transpile_backend = backend
        sampler = make_sampler(mode=backend)
        exec_source = f"IBM hardware {backend.name}"
        floor_conditions = [None]                    # device supplies the floor
    else:
        print("\n[REHEARSAL] no --arm: local AerSimulator.from_backend + injected "
              f"floors {list(REHEARSAL_FLOORS)} (no submit).")
        transpile_backend, exec_backend, exec_source = \
            rehearsal_backend(args.target_backend)
        sampler = make_sampler(mode=exec_backend)
        service = None
        floor_conditions = list(REHEARSAL_FLOORS)
    print(f"[exec source] {exec_source}")

    # SEAL manifest (sealed key + v3 config + hypotheses hashes) BEFORE execution
    manifest = {
        "protocol": "blind_benchmark_v3_device_referenced_differential",
        "target_backend": args.target_backend,
        "master_seed": MASTER_SEED,
        "n_inst": N_INST, "dark_prob": DARK_PROB, "delta_range": list(DELTA_RANGE),
        "cert_shots_per_block": CERT_SHOTS, "cert_settings": CERT_SETTINGS,
        "total_shots": total_shots, "job_count": N_INST,
        "sealed_key_sha256": sealed_sha,
        "v3_config_sha256": v3cfg_sha,
        "hypotheses_sha256": hyp_sha,
        "qpu_seconds_estimate": qest,
        "armed": armed,
        "rehearsal_injected_floors_corr": (None if armed else list(REHEARSAL_FLOORS)),
        "sealed_before_execution": True,
        "date": datetime.now(timezone.utc).isoformat(),
    }
    manifest_sha = sha256_of_obj({k: v for k, v in manifest.items()
                                  if k not in ("date",)})
    manifest["manifest_sha256"] = manifest_sha
    with open(os.path.join(OUT_DIR, prefix + "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(OUT_DIR, prefix + "sealed_key.json"), "w",
              encoding="utf-8") as fh:
        fh.write(canonical_json(sealed_key))
    with open(os.path.join(OUT_DIR, prefix + "frozen_v3_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": v3cfg, "config_sha256": v3cfg_sha,
                   "hypotheses": hyp, "hypotheses_sha256": hyp_sha}, fh, indent=2)
    print(f"[sealed] sealed_key_sha256={sealed_sha[:16]}...  "
          f"v3_config_sha256={v3cfg_sha[:16]}...  manifest_sha256={manifest_sha[:16]}...")

    # calibration snapshot (pre-submission)
    try:
        cal = calibration_snapshot(transpile_backend)
    except Exception as exc:
        cal = {"error": f"{type(exc).__name__}: {exc}"}
    with open(os.path.join(OUT_DIR, prefix + "calibration_snapshot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)

    # quota before (armed)
    quota_before = None
    if armed and service is not None:
        try:
            quota_before = service.usage().get("usage_remaining_seconds")
        except Exception:
            quota_before = None

    # execute each floor condition (rehearsal) or the single device-floor pass (armed)
    passes = []
    for theta_floor in floor_conditions:
        print(f"\n--- pass: floor={theta_floor} ---")
        passes.append(run_pass(instances, transpile_backend, sampler, armed,
                               theta_floor, v3cfg, v2_est))

    quota_after = quota_consumed = None
    if armed and service is not None:
        try:
            quota_after = service.usage().get("usage_remaining_seconds")
            if quota_before is not None and quota_after is not None:
                quota_consumed = quota_before - quota_after
        except Exception:
            pass

    # unblind verification
    ok, _payload = ChallengeGenerator.verify_sealed_key(
        os.path.join(OUT_DIR, prefix + "sealed_key.json"), sealed_sha)

    submit_decision = evaluate_contrast(passes, armed)

    meta = {
        "armed": armed, "exec_source": exec_source,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "target_backend": args.target_backend, "master_seed": MASTER_SEED,
        "total_shots": total_shots,
        "qiskit_version": qiskit.__version__, "aer_version": qiskit_aer.__version__,
        "ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "sealed_key_sha256": sealed_sha, "unblind_verified": bool(ok),
        "v3_config_sha256": v3cfg_sha, "hypotheses_sha256": hyp_sha,
        "manifest_sha256": manifest_sha,
        "v2_z_gate": v2_est.z_gate, "v2_config_sha256": v2_est.config_sha256(),
        "qpu_estimate": qest,
        "quota_remaining_before_s": quota_before,
        "quota_remaining_after_s": quota_after, "quota_consumed_s": quota_consumed,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    # answer key only written to the results JSON post-unblind (not the sealed file)
    report = {
        "metadata": meta, "manifest": manifest,
        "v3_config": v3cfg, "hypotheses": hyp,
        "answer_key": [inst.answer_key() for inst in instances],
        "passes": passes,
        "submit_decision": submit_decision,
        "calibration_snapshot": cal,
    }
    with open(os.path.join(OUT_DIR, prefix + "blind_benchmark_v3_fez.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = build_markdown(meta, v3cfg, passes, submit_decision)
    with open(os.path.join(OUT_DIR, prefix + "blind_benchmark_v3_fez.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print("=" * 78)
    print(f"mode            : {'ARMED (hardware)' if armed else 'REHEARSAL (local)'}")
    print(f"unblind verified: {ok}")
    print(f"contrast/gate   : {submit_decision['statement']}")
    print(f"quota consumed  : {quota_consumed} s")
    print(f"out dir         : {OUT_DIR}")
    print(f"runtime         : {meta['runtime_seconds']}s")


if __name__ == "__main__":
    main()
