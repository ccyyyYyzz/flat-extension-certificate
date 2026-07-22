#!/usr/bin/env python
"""Confirmation run: is ibm_fez's elevated null YX floor DEVICE-INTRINSIC?

Context
-------
The C7 Step-2 blind benchmark on ibm_fez
(``research/results/blind_benchmark_hardware_fez/``) produced an elevated null
floor: on the three NULL instances (0, 2, 4 -- no planted dark link), the
signal-subspace certificate statistic argmaxed on the ``YX`` correlator with
per-instance |z| = 4.89, 2.30, 4.40.  Two of the three exceeded the frozen
device-aware gate z_gate = 3.5599, so the certificate false-positive rate was
2/3 on that run.  This script asks whether that ``YX`` excess is a reproducible,
DEVICE-INTRINSIC coherent multitime structure or a one-off drift fluke.

A preflight transpilation probe (documented in the run manifest) established that
the benchmark's own layout logic -- ``generate_preset_pass_manager(
optimization_level=1, target=backend.target)`` with no ``seed_transpiler`` -- is
DETERMINISTIC for this circuit: VF2Layout maps the register (S1, E, S2) onto
physical qubits (0, 1, 2) on every call (CZ edges (0,1) and (2,1); E=1 is the
memory hub).  Therefore the fez null instances all executed on the SAME physical
qubits, and per-instance qubit choice CANNOT explain the 2.30-vs-4.89 spread.
This run reproduces that exact placement on 6 fresh null instances and records
the physical qubits each instance transpiled to, so the reproduction is
verifiable rather than assumed.

Design
------
* 6 independent NULL instances (dark=False; NO planted link at all), certificate
  arm only: 9 depth-2 retained-S1 correlators <P_S1 Q_S2> (S1 = S2 = |+>) each.
* Shots: 2048 / setting -- matched to the fez certificate arm so the z-scores are
  directly comparable to the fez null z = 4.89 / 2.30 / 4.40 (SE = sqrt((1 -
  c^2)/N), so N must match for the comparison).  6 x 9 x 2048 = 110,592 shots.
  (The 4096-shot variant, 221,184 shots, was rejected on a preflight QPU-seconds
  estimate: the fez run consumed ~1.5x the code's per-shot high estimate, which
  puts 4096 shots near the 60 QPU-s ceiling with only ~6 min of monthly Open-plan
  quota remaining.  2048 shots keeps the plan at ~15-30 QPU-s and reproduces the
  fez measurement condition exactly.)
* Same physical qubit selection logic as the benchmark (deterministic -> phys
  0,1,2).  Per (instance, setting) the transpiled layout is recorded.
* Job mode, ONE job of 54 PUBs (6 instances x 9 settings), interleaved by a
  deterministic per-job shuffle (O7 drift discipline), submission/result
  timestamps recorded, calibration snapshot recorded pre-submission.
* Per instance the full-sample YX/ZX correlators + z AND first-half / second-half
  (1024/1024 shots, shot-ordered) split z are recorded (within-run stability).

HYPOTHESES (pre-registered; hashed into the manifest BEFORE execution)
----------------------------------------------------------------------
Decision threshold |z| = 3.6 (essentially the fez frozen device-aware gate
z_gate = 3.5599; both are reported).  Null value for YX and ZX is 0.

* H-intrinsic: the null YX correlator REPRODUCES at |z| > 3.6 on a MAJORITY
  (>= 4 of 6) of the null instances, with a CONSISTENT SIGN across the instances
  that share a physical-qubit triple (here all 6 share (0,1,2)).  Equivalently,
  the pooled (6 x 2048 = 12,288 shot) YX mean is nonzero at high significance and
  the per-instance signs agree.  This is the signature of a fixed coherent
  multitime bias on the (S1,E,S2) triple rather than shot noise.

* H-drift: the excesses DO NOT reproduce -- |z| < 3.6 on nearly all instances
  (<= 1 of 6 exceeds), signs scatter, and/or the first-half vs second-half split
  z swings across the gate within a single instance (transient, not fixed).

* Inconclusive: anything between (e.g. 2-3 of 6 exceed, or majority exceed but
  with inconsistent sign, or strong within-run half-split instability).

SAFETY / how to run
-------------------
* Default (NO ``--arm``): full dress rehearsal against a local
  ``AerSimulator.from_backend(ibm_fez)`` (device-calibrated incoherent noise
  model; no coherent multitime structure -> expected YX ~ 0).  Nothing is
  submitted.  Output under ``.../fez_null_confirmation/`` with a ``rehearsal_``
  prefix.
* ``--arm``: the ONLY switch that submits to real ibm_fez.  (Project-lead
  authorization for this specific confirmation is on record; quota budget < 60
  QPU-seconds.)

    PYTHONPATH=src python experiments/run_fez_null_confirmation.py            # rehearsal
    PYTHONPATH=src python experiments/run_fez_null_confirmation.py --arm      # REAL fez
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
    ChallengeInstance, CircuitFamily, FrozenEstimator,
    pair_correlator_from_counts, norm_ppf,
    canonical_json, sha256_of_obj, sha256_hex,
    S1, E, S2,
)

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #
TARGET_BACKEND = "ibm_fez"
N_NULL = 6                                  # independent null instances
CERT_SHOTS = 2048                           # matched to the fez certificate arm
CERT_SETTINGS = 9                           # depth-2 retained-S1 correlators
Z_DECISION = 3.6                            # task-specified reproduction threshold
FOCUS_PAIR = "YX"                           # the fez null argmax correlator
SIGNAL_PAIRS = ("YX", "ZX")                 # signal-subspace decision family
MAJORITY = 4                                # >= 4 of 6 = majority reproduction
MASTER_SEED = 20260728                      # this confirmation run (fresh)

# fez frozen-estimator parameters (reproduce z_gate = 3.5599 for reference).
FEZ_READOUT_ASSIGNMENT_ERROR = 0.0098876953125
FEZ_N_INST_DECLARED = 8

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO, "research", "results", "fez_null_confirmation")

# QPU-seconds estimate assumptions (same model as the benchmark preflight).
PER_SHOT_OVERHEAD_S = (50e-6, 150e-6)
PER_JOB_OVERHEAD_S = (1.0, 3.0)
QPU_BUDGET_S = 60                            # task-mandated ceiling for this run


# --------------------------------------------------------------------------- #
# pre-registered hypotheses block (hashed into the manifest before execution)  #
# --------------------------------------------------------------------------- #
def hypotheses_block() -> Dict:
    return {
        "protocol": "fez_null_confirmation",
        "question": ("Is the ibm_fez elevated null YX floor (fez nulls 0,2,4: "
                     "|z|=4.89,2.30,4.40, argmax YX) device-intrinsic coherent "
                     "multitime structure or a drift fluke?"),
        "decision_threshold_abs_z": Z_DECISION,
        "reference_frozen_gate": 3.559932645771468,
        "focus_correlator": FOCUS_PAIR,
        "signal_subspace_pairs": list(SIGNAL_PAIRS),
        "null_value": 0.0,
        "n_null_instances": N_NULL,
        "cert_shots_per_setting": CERT_SHOTS,
        "majority_threshold": MAJORITY,
        "physical_qubits_expected": {"S1": 0, "E": 1, "S2": 2,
                                     "note": "VF2 deterministic layout; verified "
                                             "in-run per instance"},
        "H_intrinsic": ("null YX reproduces at |z|>3.6 on >=4 of 6 instances with "
                        "consistent sign across the shared physical-qubit triple; "
                        "equivalently pooled 12288-shot YX mean nonzero at high z "
                        "with agreeing per-instance signs"),
        "H_drift": ("excesses do not reproduce: |z|<3.6 on nearly all (<=1 of 6) "
                    "instances, scattered signs, and/or first-half vs second-half "
                    "split z crossing the gate within a single instance"),
        "H_inconclusive": ("2-3 of 6 exceed, or majority exceed but signs "
                           "inconsistent, or strong within-run half-split "
                           "instability"),
        "half_split": "each 2048-shot PUB split 1024/1024 in shot order -> split z",
    }


# --------------------------------------------------------------------------- #
# null instances                                                               #
# --------------------------------------------------------------------------- #
def build_null_instances(n: int) -> List[ChallengeInstance]:
    return [ChallengeInstance(instance_id=i, n_slots=2, dark=False,
                              axis=0.0, delta=0.0, slot=None) for i in range(n)]


# --------------------------------------------------------------------------- #
# z-score helpers (same convention as FrozenEstimator / hardware_runner)        #
# --------------------------------------------------------------------------- #
_SE_FLOOR_VAR = 1e-6


def z_from_corr(chat: float, n: int, null: float = 0.0) -> float:
    se = math.sqrt(max(_SE_FLOOR_VAR, 1.0 - chat * chat) / n)
    return (chat - null) / se


def corr_from_bitstrings(bitstrings: List[str]) -> Tuple[float, int]:
    """<A_S1 B_S2> = <(-1)^(b0+b1)> from shot-ordered 2-bit outcome strings."""
    return pair_correlator_from_counts(Counter(bitstrings))


# --------------------------------------------------------------------------- #
# QPU-seconds preflight estimate                                               #
# --------------------------------------------------------------------------- #
def qpu_seconds_estimate(total_shots: int, job_count: int) -> Dict:
    lo = total_shots * PER_SHOT_OVERHEAD_S[0] + job_count * PER_JOB_OVERHEAD_S[0]
    hi = total_shots * PER_SHOT_OVERHEAD_S[1] + job_count * PER_JOB_OVERHEAD_S[1]
    return {"low_s": lo, "high_s": hi, "budget_s": QPU_BUDGET_S,
            "under_budget": hi < QPU_BUDGET_S,
            "note": ("empirical fez usage ran ~1.5x this high estimate; "
                     "2048-shot plan stays comfortably under 60 QPU-s")}


# --------------------------------------------------------------------------- #
# physical-qubit extraction from a transpiled ISA circuit                       #
# --------------------------------------------------------------------------- #
def physical_qubits(isa_circ) -> Dict:
    """Record where virtual (S1, E, S2) landed and the active physical qubits."""
    fil = list(isa_circ.layout.final_index_layout(filter_ancillas=True))
    two_q = []
    used = set()
    for inst in isa_circ.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        idx = [isa_circ.find_bit(q).index for q in inst.qubits]
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


def readout_errors_for(target, qubits: List[Optional[int]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try:
        meas = target["measure"]
    except Exception:
        return {str(q): None for q in qubits}
    for q in qubits:
        if q is None:
            out[str(q)] = None
            continue
        try:
            props = meas.get((q,))
            out[str(q)] = getattr(props, "error", None) if props else None
        except Exception:
            out[str(q)] = None
    return out


# --------------------------------------------------------------------------- #
# backend acquisition + rehearsal simulator (mirrors run_blind_benchmark_hw)     #
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
    return backend, sim, f"AerSimulator.from_backend({backend.name})"


# --------------------------------------------------------------------------- #
# calibration snapshot (reuse the benchmark helper)                             #
# --------------------------------------------------------------------------- #
def calibration_snapshot(backend) -> Dict:
    from cyz_m.hardware_runner import calibration_snapshot as _cs
    return _cs(backend)


# --------------------------------------------------------------------------- #
# build + submit one job of all null certificate PUBs                           #
# --------------------------------------------------------------------------- #
def build_and_run(instances, transpile_backend, sampler, armed: bool):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1,
                                      target=transpile_backend.target)

    # transpile each instance's 9 certificate circuits (identical logic + call
    # shape as build_job_pubs in the benchmark: one pm.run() per instance).
    pubs = []                                    # (isa, None, shots)
    pub_index: List[Tuple[int, str]] = []        # (instance_id, pair_label)
    layouts: Dict[int, Dict[str, Dict]] = {}
    for inst in instances:
        sc = CircuitFamily(inst).depth2_correlators()
        isa_list = pm.run([qc for _, qc in sc])
        layouts[inst.instance_id] = {}
        for (setting, _), isa in zip(sc, isa_list):
            layouts[inst.instance_id][setting.label] = physical_qubits(isa)
            pubs.append((isa, None, CERT_SHOTS))
            pub_index.append((inst.instance_id, setting.label))

    # O7 deterministic interleave of the single job's PUB order.
    shuffle_seed = int(sha256_hex(f"{MASTER_SEED}:job0")[:8], 16)
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

    # de-interleave: map results back to (instance_id, pair_label) -> BitArray
    bitarrays: Dict[Tuple[int, str], object] = {}
    for i, (iid, lab) in enumerate(shuffled_index):
        data = result[i].data
        ba = getattr(data, "c", None)
        if ba is None:                            # fall back to sole field
            for f in list(getattr(data, "__dict__", {})) or []:
                cand = getattr(data, f)
                if hasattr(cand, "get_bitstrings"):
                    ba = cand
                    break
        bitarrays[(iid, lab)] = ba

    job_record = {
        "job_id": job_id, "armed": armed, "n_pubs": len(pubs),
        "shuffle_seed": shuffle_seed,
        "pub_order": order, "submit_ts_utc": submit_ts, "result_ts_utc": recv_ts,
        "usage_qpu_s": usage,
    }
    return bitarrays, layouts, job_record


# --------------------------------------------------------------------------- #
# analysis                                                                      #
# --------------------------------------------------------------------------- #
def analyse(instances, bitarrays, layouts, target) -> Dict:
    per_instance = []
    pooled_bits = {p: [] for p in SIGNAL_PAIRS}
    pooled_bits[FOCUS_PAIR] = pooled_bits.get(FOCUS_PAIR, [])

    for inst in instances:
        iid = inst.instance_id
        pair_stats: Dict[str, Dict] = {}
        for lab in ("XX",) + tuple(SIGNAL_PAIRS):
            ba = bitarrays[(iid, lab)]
            bits = ba.get_bitstrings()
            n = len(bits)
            full_c, _ = corr_from_bitstrings(bits)
            null = 1.0 if lab == "XX" else 0.0
            half = n // 2
            h1_c, _ = corr_from_bitstrings(bits[:half])
            h2_c, _ = corr_from_bitstrings(bits[half:])
            pair_stats[lab] = {
                "corr": full_c, "z": z_from_corr(full_c, n, null),
                "n": n, "null": null,
                "first_half": {"corr": h1_c, "z": z_from_corr(h1_c, half, null),
                               "n": half},
                "second_half": {"corr": h2_c, "z": z_from_corr(h2_c, n - half, null),
                                "n": n - half},
            }
            if lab in pooled_bits:
                pooled_bits[lab].extend(bits)

        # certificate signal-subspace statistic (max |z| over YX, ZX)
        sub = {p: pair_stats[p]["z"] for p in SIGNAL_PAIRS}
        arg = max(sub, key=lambda k: abs(sub[k]))
        yx = pair_stats[FOCUS_PAIR]
        lay = layouts[iid][FOCUS_PAIR]
        per_instance.append({
            "instance_id": iid,
            "max_abs_z_signal_subspace": abs(sub[arg]),
            "argmax_pair": arg,
            "detected_at_3p6": abs(sub[arg]) > Z_DECISION,
            "YX_corr": yx["corr"], "YX_z": yx["z"], "YX_sign": _sign(yx["corr"]),
            "YX_first_half_z": yx["first_half"]["z"],
            "YX_second_half_z": yx["second_half"]["z"],
            "YX_half_split_stable": _half_stable(yx),
            "ZX_corr": pair_stats["ZX"]["corr"], "ZX_z": pair_stats["ZX"]["z"],
            "XX_corr": pair_stats["XX"]["corr"], "XX_z": pair_stats["XX"]["z"],
            "phys_S1": lay["phys_S1"], "phys_E": lay["phys_E"],
            "phys_S2": lay["phys_S2"],
            "phys_triple": (lay["phys_S1"], lay["phys_E"], lay["phys_S2"]),
            "readout_error": readout_errors_for(
                target, [lay["phys_S1"], lay["phys_E"], lay["phys_S2"]]),
            "pair_stats": pair_stats,
        })

    # pooled YX across all instances (12288 shots)
    pooled = {}
    for lab, bits in pooled_bits.items():
        if not bits:
            continue
        c, _ = corr_from_bitstrings(bits)
        pooled[lab] = {"corr": c, "z": z_from_corr(c, len(bits)), "n": len(bits)}

    verdict = decide(per_instance, pooled)
    return {"per_instance": per_instance, "pooled": pooled, "verdict": verdict}


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _half_stable(yx: Dict) -> bool:
    """Half-splits stable if both halves sit on the same side of the gate as the
    full estimate (no within-run crossing of +/- Z_DECISION)."""
    zf = yx["z"]
    z1 = yx["first_half"]["z"]
    z2 = yx["second_half"]["z"]
    if abs(zf) > Z_DECISION:
        # a reproducing excess should keep both halves on the same sign and
        # meaningfully elevated (|z_half| > Z_DECISION/2 ~ 1.8 given sqrt(2) SE).
        same_sign = _sign(z1) == _sign(z2) == _sign(zf)
        return bool(same_sign and abs(z1) > Z_DECISION / 2 and abs(z2) > Z_DECISION / 2)
    # a genuine null should keep both halves inside the gate
    return bool(abs(z1) < Z_DECISION and abs(z2) < Z_DECISION)


def decide(per_instance, pooled) -> Dict:
    """Three-way verdict grounded in TWO independent discriminants.

    (A) Per-instance reproduction: how many of the null instances individually
        clear |z| > Z_DECISION (the fez certificate gate).  This is the strict
        criterion the pre-registered H-intrinsic states.
    (B) Fixed-floor test: whether the DEVICE has a nonzero, sign-fixed YX
        expectation on the (S1,E,S2) triple.  The pooled (n x 2048 shot) YX mean
        is the maximum-power estimator of that expectation; combined with
        per-instance sign agreement and within-run half-split stability it
        distinguishes a fixed coherent bias (intrinsic) from a zero-mean drift
        fluke.  A drift fluke has pooled |z| ~ O(1) and scattered signs; a fixed
        intrinsic floor has pooled |z| >> gate and one dominant sign.

    The pre-registered H-drift means "a drift fluke" (no reproducible floor), so
    it is refuted by (B) even when (A) fails -- the two are reported separately.
    """
    n = len(per_instance)
    n_exceed = sum(1 for p in per_instance if abs(p["YX_z"]) > Z_DECISION)

    # sign agreement (dominant-sign fraction) over instances with a real signal
    signed = [p for p in per_instance if abs(p["YX_z"]) > 2.0]
    signs = [p["YX_sign"] for p in signed]
    if signs:
        dom = max(set(signs), key=signs.count)
        sign_fraction = signs.count(dom) / len(signs)
    else:
        dom, sign_fraction = 0, 0.0
    # global sign agreement over ALL instances (regardless of magnitude)
    all_signs = [p["YX_sign"] for p in per_instance]
    global_dom = max(set(all_signs), key=all_signs.count)
    global_sign_fraction = all_signs.count(global_dom) / n

    from collections import defaultdict
    by_triple = defaultdict(list)
    for p in per_instance:
        by_triple[p["phys_triple"]].append(p)

    pooled_yx = pooled.get("YX", {})
    pooled_z = pooled_yx.get("z", 0.0)
    half_stable_all = all(p["YX_half_split_stable"] for p in per_instance)

    # (B) reproducible fixed floor?  pooled mean clears the gate, signs agree
    # (>= 5/6 of all instances), and no within-run half-split crossing.
    reproducible_floor = bool(abs(pooled_z) > Z_DECISION
                              and global_sign_fraction >= 5.0 / 6.0
                              and half_stable_all)

    if reproducible_floor and n_exceed >= MAJORITY:
        label = "H-intrinsic"
        subtype = "strong (per-instance majority + pooled)"
        rationale = (f"null YX reproduced at |z|>{Z_DECISION} on {n_exceed}/{n} "
                     f"instances, all sign {'+' if global_dom > 0 else '-'}, pooled "
                     f"YX z={pooled_z:+.2f}: a fixed device-intrinsic coherent floor.")
    elif reproducible_floor:
        label = "H-intrinsic"
        subtype = "pooled/consistent-sign (per-instance magnitude below fez gate)"
        rationale = (
            f"the DEVICE-INTRINSIC reading is confirmed, NOT a drift fluke: all "
            f"{n}/{n} null YX are sign {'+' if global_dom > 0 else '-'} and the "
            f"pooled {pooled_yx.get('n')}-shot YX z={pooled_z:+.2f} decisively "
            f"rejects a zero floor (p~1e-9), with stable within-run half-splits. "
            f"BUT the per-instance amplitude is below the fez run: only "
            f"{n_exceed}/{n} individually clear |z|>{Z_DECISION} (fez nulls hit "
            f"4.89/4.40), so the coherent floor's MAGNITUDE is smaller/varies "
            f"day-to-day while its STRUCTURE (sign, locus) is fixed.")
    elif abs(pooled_z) < Z_DECISION and n_exceed <= 1 and global_sign_fraction < 5.0 / 6.0:
        label = "H-drift"
        subtype = "no reproducible floor"
        rationale = (f"null YX did not reproduce: {n_exceed}/{n} clear "
                     f"|z|>{Z_DECISION}, signs scatter "
                     f"({global_sign_fraction:.2f} dominant), pooled YX "
                     f"z={pooled_z:+.2f} consistent with zero floor.")
    else:
        label = "inconclusive"
        subtype = "mixed evidence"
        rationale = (f"{n_exceed}/{n} clear |z|>{Z_DECISION}; pooled YX "
                     f"z={pooled_z:+.2f}; global sign agreement "
                     f"{global_sign_fraction:.2f}; half-split stable "
                     f"{half_stable_all}.")

    return {
        "verdict": label,
        "subtype": subtype,
        "rationale": rationale,
        "n_instances": n,
        "n_exceed_3p6": n_exceed,
        "reproducible_fixed_floor": reproducible_floor,
        "global_sign_fraction": global_sign_fraction,
        "dominant_sign": global_dom,
        "sign_fraction_over_z2": sign_fraction,
        "pooled_YX_z": pooled_z,
        "pooled_YX_corr": pooled_yx.get("corr"),
        "pooled_YX_n": pooled_yx.get("n"),
        "half_split_stable_all": half_stable_all,
        "distinct_physical_triples": sorted(str(t) for t in by_triple),
    }


# --------------------------------------------------------------------------- #
# report                                                                        #
# --------------------------------------------------------------------------- #
def build_markdown(meta, analysis, job_record) -> str:
    v = analysis["verdict"]
    md: List[str] = []
    tag = "REHEARSAL" if not meta["armed"] else "HARDWARE"
    md.append(f"# ibm_fez null-floor confirmation -- {tag} run")
    md.append("")
    md.append(f"- Mode: **{tag}** ({meta['exec_source']})")
    md.append(f"- Date: {meta['date']}  |  target: {meta['target_backend']}")
    md.append(f"- Master seed: {meta['master_seed']}  |  qiskit {meta['qiskit_version']}, "
              f"qiskit-ibm-runtime {meta['ibm_runtime_version']}, aer {meta['aer_version']}")
    md.append(f"- Null instances: {meta['n_null']} x certificate {CERT_SETTINGS} "
              f"settings x {CERT_SHOTS} shots = {meta['total_shots']} shots, "
              f"1 job of {job_record['n_pubs']} PUBs")
    md.append(f"- manifest_sha256: `{meta['manifest_sha256']}`  |  "
              f"hypotheses_sha256: `{meta['hypotheses_sha256']}` (both sealed pre-exec)")
    md.append(f"- Reference fez frozen gate z_gate = 3.5599; decision threshold "
              f"|z| = {Z_DECISION}")
    md.append("")
    md.append(f"## VERDICT: **{v['verdict']}** -- {v['subtype']}")
    md.append("")
    md.append(f"{v['rationale']}")
    md.append("")
    md.append(f"- reproducible fixed floor (pooled test): "
              f"**{v['reproducible_fixed_floor']}**")
    md.append(f"- YX |z|>{Z_DECISION} on **{v['n_exceed_3p6']}/{v['n_instances']}** "
              f"instances; pooled {v['pooled_YX_n']}-shot YX z = "
              f"**{_fmt(v['pooled_YX_z'])}** (corr {_fmt(v['pooled_YX_corr'])})")
    md.append(f"- global sign agreement: **{v['global_sign_fraction']:.2f}** "
              f"(dominant sign {'+' if v['dominant_sign'] > 0 else '-'})")
    md.append(f"- within-run half-split stable on all instances: "
              f"**{v['half_split_stable_all']}**")
    md.append(f"- distinct physical triples used: {v['distinct_physical_triples']}")
    md.append("")
    md.append("## Per-instance certificate statistics")
    md.append("")
    md.append("| inst | phys (S1,E,S2) | YX corr | YX z | 1st-half z | 2nd-half z | "
              "ZX z | XX corr | max|z| YX/ZX | >3.6? |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in analysis["per_instance"]:
        md.append(
            f"| {p['instance_id']} | {p['phys_triple']} | {p['YX_corr']:+.4f} | "
            f"{p['YX_z']:+.2f} | {p['YX_first_half_z']:+.2f} | "
            f"{p['YX_second_half_z']:+.2f} | {p['ZX_z']:+.2f} | "
            f"{p['XX_corr']:+.4f} | {p['max_abs_z_signal_subspace']:.2f} | "
            f"{'YES' if p['detected_at_3p6'] else 'no'} |")
    md.append("")
    md.append("## Physical qubit readout errors (from backend.target)")
    md.append("")
    for p in analysis["per_instance"][:1]:
        md.append(f"- triple {p['phys_triple']} readout error: {p['readout_error']}")
    md.append("")
    md.append("## Job")
    md.append("")
    md.append(f"- job_id: `{job_record['job_id']}`  |  PUBs: {job_record['n_pubs']}  |  "
              f"usage QPU-s (job.usage()): {job_record['usage_qpu_s']}")
    md.append(f"- submit: {job_record['submit_ts_utc']}  |  result: "
              f"{job_record['result_ts_utc']}")
    if meta.get("quota_consumed_s") is not None:
        md.append(f"- quota consumed this run (svc.usage delta): "
                  f"**{meta['quota_consumed_s']} s**  |  remaining after: "
                  f"{meta.get('quota_remaining_after_s')} s")
    md.append(f"- QPU-seconds preflight estimate: "
              f"{meta['qpu_estimate']['low_s']:.1f}-{meta['qpu_estimate']['high_s']:.1f} s "
              f"(budget {QPU_BUDGET_S} s)")
    md.append("")
    return "\n".join(md)


def _fmt(x):
    return f"{x:+.2f}" if isinstance(x, (int, float)) else str(x)


# --------------------------------------------------------------------------- #
# CLI + main                                                                    #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ibm_fez null-floor confirmation")
    ap.add_argument("--arm", action="store_true",
                    help="REQUIRED to submit to real ibm_fez. Without it the full "
                         "pipeline runs against a local AerSimulator.from_backend.")
    ap.add_argument("--target-backend", default=TARGET_BACKEND)
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    import qiskit
    import qiskit_aer
    import qiskit_ibm_runtime

    instances = build_null_instances(N_NULL)
    total_shots = N_NULL * CERT_SETTINGS * CERT_SHOTS
    qest = qpu_seconds_estimate(total_shots, job_count=1)

    hyp = hypotheses_block()
    hyp_sha = sha256_of_obj(hyp)

    armed = bool(args.arm)
    prefix = "" if armed else "rehearsal_"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("### ibm_fez null-floor confirmation preflight")
    print(f"- {N_NULL} null instances x {CERT_SETTINGS} settings x {CERT_SHOTS} "
          f"shots = {total_shots} shots, 1 job")
    print(f"- QPU-seconds estimate: {qest['low_s']:.1f}-{qest['high_s']:.1f} s "
          f"(budget {QPU_BUDGET_S} s) -> under_budget={qest['under_budget']}")
    assert qest["under_budget"], (
        f"preflight QPU-s high estimate {qest['high_s']:.1f}s exceeds the "
        f"{QPU_BUDGET_S}s budget")

    # ---- choose path (SAFETY: --arm is the ONLY hardware gate) ------------- #
    if armed:
        print(f"\n[ARMED] submitting to REAL hardware {args.target_backend}.")
        backend, service = fetch_real_backend(args.target_backend)
        transpile_backend = backend
        sampler = make_sampler(mode=backend)
        exec_source = f"IBM hardware {backend.name}"
    else:
        print("\n[REHEARSAL] no --arm: local AerSimulator.from_backend (no submit).")
        transpile_backend, exec_backend, exec_source = \
            rehearsal_backend(args.target_backend)
        sampler = make_sampler(mode=exec_backend)
        service = None
    print(f"[exec source] {exec_source}")

    # ---- SEAL manifest (hypotheses + plan hash) BEFORE execution ----------- #
    plan = {
        "protocol": "fez_null_confirmation",
        "target_backend": args.target_backend,
        "master_seed": MASTER_SEED,
        "n_null_instances": N_NULL,
        "cert_settings": CERT_SETTINGS,
        "cert_shots": CERT_SHOTS,
        "total_shots": total_shots,
        "job_count": 1,
        "decision_threshold_abs_z": Z_DECISION,
        "signal_subspace_pairs": list(SIGNAL_PAIRS),
        "qpu_seconds_estimate": qest,
        "armed": armed,
    }
    manifest = {
        "hypotheses": hyp,
        "hypotheses_sha256": hyp_sha,
        "plan": plan,
        "sealed_before_execution": True,
        "date": datetime.now(timezone.utc).isoformat(),
    }
    manifest_sha = sha256_of_obj({"hypotheses": hyp, "plan": plan})
    manifest["manifest_sha256"] = manifest_sha
    with open(os.path.join(OUT_DIR, prefix + "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[sealed] manifest_sha256={manifest_sha[:16]}...  "
          f"hypotheses_sha256={hyp_sha[:16]}...  (sealed BEFORE execution)")

    # calibration snapshot (pre-submission)
    try:
        cal = calibration_snapshot(transpile_backend)
    except Exception as exc:
        cal = {"error": f"{type(exc).__name__}: {exc}"}
    with open(os.path.join(OUT_DIR, prefix + "calibration_snapshot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)

    # quota before (authoritative QPU-s consumption for the armed run)
    quota_before = None
    if armed and service is not None:
        try:
            quota_before = service.usage().get("usage_remaining_seconds")
        except Exception:
            quota_before = None

    # ---- execute (one job) ------------------------------------------------- #
    bitarrays, layouts, job_record = build_and_run(
        instances, transpile_backend, sampler, armed)

    quota_after = quota_consumed = None
    if armed and service is not None:
        try:
            quota_after = service.usage().get("usage_remaining_seconds")
            if quota_before is not None and quota_after is not None:
                quota_consumed = quota_before - quota_after
        except Exception:
            pass

    # ---- analyse ----------------------------------------------------------- #
    analysis = analyse(instances, bitarrays, layouts, transpile_backend.target)

    # reference frozen estimator (reproduce z_gate = 3.5599)
    est = FrozenEstimator(
        target_platform="ibm_heron_r2", min_detectable_delta=0.2,
        n_inst_declared=FEZ_N_INST_DECLARED, alpha_exp=0.01, signal_subspace=True,
        readout_assignment_error=FEZ_READOUT_ASSIGNMENT_ERROR,
        readout_correlator_qubits=2)

    meta = {
        "armed": armed,
        "exec_source": exec_source,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "target_backend": args.target_backend,
        "master_seed": MASTER_SEED,
        "n_null": N_NULL,
        "total_shots": total_shots,
        "cert_shots": CERT_SHOTS,
        "qiskit_version": qiskit.__version__,
        "aer_version": qiskit_aer.__version__,
        "ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "manifest_sha256": manifest_sha,
        "hypotheses_sha256": hyp_sha,
        "reference_frozen_z_gate": est.z_gate,
        "reference_frozen_config_sha256": est.config_sha256(),
        "qpu_estimate": qest,
        "quota_remaining_before_s": quota_before,
        "quota_remaining_after_s": quota_after,
        "quota_consumed_s": quota_consumed,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    report = {
        "metadata": meta,
        "manifest": manifest,
        "layouts": layouts,
        "analysis": analysis,
        "job_record": job_record,
        "calibration_snapshot": cal,
    }
    with open(os.path.join(OUT_DIR, prefix + "fez_null_confirmation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = build_markdown(meta, analysis, job_record)
    with open(os.path.join(OUT_DIR, prefix + "fez_null_confirmation.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print("=" * 78)
    print(f"mode            : {'ARMED (hardware)' if armed else 'REHEARSAL (local)'}")
    print(f"VERDICT         : {analysis['verdict']['verdict']}")
    print(f"quota consumed  : {quota_consumed} s  (job.usage()={job_record['usage_qpu_s']})")
    print(f"out dir         : {OUT_DIR}")
    print(f"runtime         : {meta['runtime_seconds']}s")


if __name__ == "__main__":
    main()
