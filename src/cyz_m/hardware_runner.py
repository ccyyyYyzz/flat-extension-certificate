"""IBM-hardware run-plan, PUB construction, and scoring for the C7 blind benchmark.

This is the *hardware* counterpart of the simulator dry run in
``experiments/run_blind_benchmark_dryrun.py``.  It reuses the blind-discipline
objects of :mod:`cyz_m.blind_benchmark` (deterministic sealed challenges, the
frozen flat-extension certificate, the resource ledger) and the null-prediction
math of :mod:`cyz_m.benchmark_baselines`, and adds everything specific to a real
IBM Quantum submission:

    * a deterministic, quota-fitting :class:`HardwareRunPlan` (target
      ``ibm_marrakesh``) with a pre-flight shots/jobs/QPU-seconds table;
    * a ``qiskit-ibm-runtime`` SamplerV2 **job-mode** PUB builder (Open plan does
      not allow dedicated sessions -- each instance is one job of PUBs);
    * an O7 drift discipline (deterministic per-job circuit-order interleave,
      recorded submission timestamps, a ``backend.target``-derived calibration
      snapshot);
    * the same scoring pipeline as the dry run (certificate / structured / the
      marginal-QPT blindness null / the held-out depth-3 confirmation).

SAFETY.  This module *never* submits a job on its own.  Submission happens only
in :func:`experiments.run_blind_benchmark_hardware.main` behind an explicit
``--arm`` switch; without ``--arm`` the identical PUB path runs against a local
``AerSimulator.from_backend(backend)`` (noise-model-from-device dress rehearsal).
The plan / estimate / calibration helpers here import qiskit lazily so the plan
math stays importable and testable without qiskit.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cyz_m.blind_benchmark import (
    ChallengeGenerator, ChallengeInstance, FrozenEstimator, CircuitFamily,
    ResourceLedger, sha256_of_obj,
    pair_correlator_from_counts, single_expectation_from_counts,
    OFFNULL_PAIRS, SIGNAL_SUBSPACE_PAIRS, CORRELATOR_NULL, norm_ppf,
)
from cyz_m.benchmark_baselines import (
    null_atomic_expectation, null_heldout_correlator,
)

# --------------------------------------------------------------------------- #
# run-plan constants (quota-fitting; deterministic from a master seed)          #
# --------------------------------------------------------------------------- #
TARGET_BACKEND = "ibm_marrakesh"          # IBM Heron r2, 156 qubits (Open plan)
ESTIMATOR_PLATFORM = "ibm_heron_r2"       # Step-1 separator-power platform key
N_INST_HW = 8                             # blind challenge instances on hardware
DELTA_RANGE = (0.2, 0.4)
N_SLOTS = 2
ALPHA_EXP = 0.01

# ibm_marrakesh median readout (measure) assignment error, from the calibration
# snapshot (research/results/blind_benchmark_hardware_rehearsal/
# rehearsal_calibration_snapshot.json: measure median_error = 0.00989 over 156
# qubits).  This is a device-calibration figure available BEFORE any challenge
# unblinding, so freezing the gate against it is legitimate pre-registration.
# The dry-run had assumed a negligible (~0.12%) readout floor.
DEVICE_READOUT_ASSIGNMENT_ERROR = 0.0098876953125
READOUT_CORRELATOR_QUBITS = 2             # pair correlators <P_S1 Q_S2>

CERT_SETTINGS = 9                         # depth-2 retained-S1 correlators
MARGINAL_SETTINGS = 36                    # 6 preps x 2 slots x 3 Paulis
HELDOUT_SETTINGS = 9                      # depth-3 confirmation family
MARGINAL_SHOTS = 512                      # blindness null needs fewer shots
HELDOUT_SHOTS = 2048
HELDOUT_INSTANCE_IDS = (0, 1)             # held-out depth-3 on 2 instances only

# fresh master seed for the hardware run (distinct from the dry-run seed
# 20260721 in experiments/run_blind_benchmark_dryrun.py).
HW_MASTER_SEED = 20260722

# QPU-seconds estimate assumptions (stated in the pre-flight table).
PER_SHOT_OVERHEAD_S = (50e-6, 150e-6)     # 50-150 us / shot (init + gate + RO)
PER_JOB_OVERHEAD_S = (1.0, 3.0)           # circuit load / classical setup per job
QPU_BUDGET_S = 7 * 60                     # 7-min Open-plan reserve guard (=420 s)

_Z95 = 1.959963984540054
_Z99 = 2.5758293035489004


# --------------------------------------------------------------------------- #
# arms of the run plan                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arm:
    """One measurement arm of the hardware run.

    ``builder`` is the :class:`~cyz_m.blind_benchmark.CircuitFamily` method that
    produces this arm's ``(Setting, circuit)`` list for an instance.
    """
    name: str
    builder: str
    n_settings: int
    shots: int
    instance_ids: Tuple[int, ...]
    readout: str                          # "single" | "pair"


def _seed_int(master_seed: int, tag: str) -> int:
    """Deterministic 32-bit sub-seed (portable across processes/platforms)."""
    h = hashlib.sha256(f"{master_seed}:{tag}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 32)


class HardwareRunPlan:
    """Deterministic quota-fitting run plan for the C7 blind benchmark.

    * ``certificate`` arm  : 9 settings x 2048 shots, all instances.
    * ``marginal_qpt`` arm : 36 settings x 512 shots, all instances (blindness
      null; fewer shots -- see :meth:`marginal_ci`).
    * ``heldout_depth3``   : 9 settings x 2048 shots, first ``HELDOUT_INSTANCE_IDS``
      instances only.
    * ``structured``       : analysis-only, re-uses the certificate counts.
    * ``full_ptt``         : simulator-only, never on hardware (excluded here).

    JOB MODE: one job per instance, all of that instance's PUBs batched together;
    PUB order is interleaved by a deterministic per-job shuffle (O7).
    """

    def __init__(self, master_seed: int = HW_MASTER_SEED, n_inst: int = N_INST_HW,
                 target_backend: str = TARGET_BACKEND,
                 estimator: Optional[FrozenEstimator] = None) -> None:
        self.master_seed = int(master_seed)
        self.n_inst = int(n_inst)
        self.target_backend = target_backend
        self.gen = ChallengeGenerator(self.master_seed, self.n_inst,
                                      DELTA_RANGE, N_SLOTS)
        self.instances = self.gen.instances
        # v2 hardened certificate: signal-subspace decision + device-aware gate
        # (frozen BEFORE unblinding; see FrozenEstimator docstring / changelog).
        self.estimator = estimator or FrozenEstimator(
            target_platform=ESTIMATOR_PLATFORM,
            min_detectable_delta=DELTA_RANGE[0],
            n_inst_declared=self.n_inst, alpha_exp=ALPHA_EXP,
            signal_subspace=True,
            readout_assignment_error=DEVICE_READOUT_ASSIGNMENT_ERROR,
            readout_correlator_qubits=READOUT_CORRELATOR_QUBITS)
        # certificate shots are the frozen estimator's n_shots (== 2048 here).
        self.cert_shots = int(self.estimator.n_shots)
        heldout_ids = tuple(i for i in HELDOUT_INSTANCE_IDS if i < self.n_inst)
        self.arms: List[Arm] = [
            Arm("certificate", "depth2_correlators",
                CERT_SETTINGS, self.cert_shots, tuple(range(self.n_inst)), "pair"),
            Arm("marginal_qpt", "atomic_testers",
                MARGINAL_SETTINGS, MARGINAL_SHOTS, tuple(range(self.n_inst)), "single"),
            Arm("heldout_depth3", "heldout_depth3",
                HELDOUT_SETTINGS, HELDOUT_SHOTS, heldout_ids, "pair"),
        ]
        self.jobs = self._plan_jobs()

    # ---- instance lookup ------------------------------------------------- #
    def instance_by_id(self, instance_id: int) -> ChallengeInstance:
        return self.instances[instance_id]

    # ---- job planning (one job per instance, O7 interleave) -------------- #
    def _plan_jobs(self) -> List[Dict]:
        jobs = []
        for inst in self.instances:
            pub_specs: List[Tuple[str, int, int]] = []   # (arm, setting_idx, shots)
            for arm in self.arms:
                if inst.instance_id in arm.instance_ids:
                    for s in range(arm.n_settings):
                        pub_specs.append((arm.name, s, arm.shots))
            seed_i = _seed_int(self.master_seed, f"job{inst.instance_id}")
            import random
            rng = random.Random(seed_i)
            order = list(range(len(pub_specs)))
            rng.shuffle(order)                          # O7 deterministic interleave
            jobs.append({
                "instance_id": inst.instance_id,
                "n_pubs": len(pub_specs),
                "shuffle_seed": seed_i,
                "pub_order": order,
                # PUB references in submission (shuffled) order:
                "pubs": [pub_specs[k] for k in order],
            })
        return jobs

    # ---- totals / estimates ---------------------------------------------- #
    def totals(self) -> Dict:
        total_shots = 0
        total_pubs = 0
        by_arm: Dict[str, Dict] = {}
        for arm in self.arms:
            n_inst_arm = len(arm.instance_ids)
            pubs = n_inst_arm * arm.n_settings
            shots = pubs * arm.shots
            by_arm[arm.name] = {
                "instances": n_inst_arm,
                "settings": arm.n_settings,
                "shots_per_setting": arm.shots,
                "pubs": pubs,
                "shots": shots,
            }
            total_shots += shots
            total_pubs += pubs
        return {
            "total_shots": total_shots,
            "total_pubs": total_pubs,
            "job_count": len(self.jobs),
            "by_arm": by_arm,
        }

    def qpu_seconds_estimate(self) -> Dict:
        t = self.totals()
        s, j = t["total_shots"], t["job_count"]
        lo = s * PER_SHOT_OVERHEAD_S[0] + j * PER_JOB_OVERHEAD_S[0]
        hi = s * PER_SHOT_OVERHEAD_S[1] + j * PER_JOB_OVERHEAD_S[1]
        return {
            "low_s": lo,
            "high_s": hi,
            "budget_s": QPU_BUDGET_S,
            "under_budget": hi < QPU_BUDGET_S,
            "assumptions": {
                "per_shot_overhead_s": list(PER_SHOT_OVERHEAD_S),
                "per_job_overhead_s": list(PER_JOB_OVERHEAD_S),
                "model": "qpu_s = total_shots * per_shot + job_count * per_job",
                "note": "excludes queue wait; SamplerV2 job-mode active QPU time.",
            },
        }

    def marginal_ci(self) -> Dict:
        """CI half-width of a marginal-QPT off-null Pauli expectation.

        Under the blindness null the atomic off-null expectations are ~0, so the
        binomial standard error is ``sqrt((1 - E^2)/N) -> sqrt(1/N)`` at
        ``N = MARGINAL_SHOTS`` shots.  With 512 shots the blindness confirmation
        certifies |E| < the reported half-width rather than = 0.
        """
        se = math.sqrt(1.0 / MARGINAL_SHOTS)
        return {
            "marginal_shots": MARGINAL_SHOTS,
            "se_null": se,
            "ci_halfwidth_95": _Z95 * se,
            "ci_halfwidth_99": _Z99 * se,
        }

    # ---- deterministic serialization + hash ------------------------------ #
    def as_dict(self) -> Dict:
        return {
            "protocol": "c7_blind_benchmark_hardware",
            "target_backend": self.target_backend,
            "master_seed": self.master_seed,
            "n_inst": self.n_inst,
            "delta_range": list(DELTA_RANGE),
            "n_slots": N_SLOTS,
            "alpha_exp": ALPHA_EXP,
            "cert_shots": self.cert_shots,
            "arms": [
                {"name": a.name, "builder": a.builder, "n_settings": a.n_settings,
                 "shots": a.shots, "instance_ids": list(a.instance_ids),
                 "readout": a.readout}
                for a in self.arms
            ],
            "jobs": [
                {"instance_id": jb["instance_id"], "n_pubs": jb["n_pubs"],
                 "shuffle_seed": jb["shuffle_seed"], "pub_order": jb["pub_order"]}
                for jb in self.jobs
            ],
            "totals": self.totals(),
            "qpu_seconds_estimate": self.qpu_seconds_estimate(),
            "marginal_ci": self.marginal_ci(),
            "structured_comparator": "analysis-only on certificate counts",
            "full_ptt": "simulator-only; excluded from the hardware plan",
            "estimator_config_sha256": self.estimator.config_sha256(),
            "sealed_key_sha256": self.gen.sealed_key_sha256(),
        }

    def plan_sha256(self) -> str:
        return sha256_of_obj(self.as_dict())

    # ---- pre-flight table ------------------------------------------------ #
    def preflight_markdown(self) -> str:
        t = self.totals()
        q = self.qpu_seconds_estimate()
        ci = self.marginal_ci()
        md: List[str] = []
        md.append("### Pre-flight run plan "
                  f"(target {self.target_backend}, seed {self.master_seed})")
        md.append("")
        md.append("| arm | instances | settings | shots/setting | PUBs | shots |")
        md.append("|---|---|---|---|---|---|")
        for a in self.arms:
            b = t["by_arm"][a.name]
            md.append(f"| {a.name} | {b['instances']} | {b['settings']} | "
                      f"{b['shots_per_setting']} | {b['pubs']} | {b['shots']} |")
        md.append(f"| **total** | - | - | - | **{t['total_pubs']}** | "
                  f"**{t['total_shots']}** |")
        md.append("")
        md.append(f"- Jobs (job mode, 1 per instance): **{t['job_count']}**")
        md.append(f"- QPU-seconds estimate: **{q['low_s']:.1f}-{q['high_s']:.1f} s** "
                  f"(per-shot {PER_SHOT_OVERHEAD_S[0]*1e6:.0f}-"
                  f"{PER_SHOT_OVERHEAD_S[1]*1e6:.0f} us + per-job "
                  f"{PER_JOB_OVERHEAD_S[0]:.0f}-{PER_JOB_OVERHEAD_S[1]:.0f} s)")
        md.append(f"- Under 7-min Open-plan reserve ({QPU_BUDGET_S} s): "
                  f"**{q['under_budget']}**")
        md.append(f"- Structured comparator: analysis-only on certificate counts "
                  f"(0 extra shots).  Full PTT: simulator-only (never hardware).")
        md.append(f"- Marginal-QPT blindness null @ {ci['marginal_shots']} shots: "
                  f"se={ci['se_null']:.4f}, 95% CI half-width="
                  f"**+/-{ci['ci_halfwidth_95']:.4f}** "
                  f"(99%: +/-{ci['ci_halfwidth_99']:.4f})")
        md.append("")
        return "\n".join(md)


# --------------------------------------------------------------------------- #
# calibration snapshot (backend.properties-equivalent from backend.target)      #
# --------------------------------------------------------------------------- #
def calibration_snapshot(backend) -> Dict:
    """A JSON-able calibration snapshot derived from ``backend.target``.

    Records per-qubit T1/T2/frequency, per-operation error/duration statistics,
    and per-qubit readout (measure) error -- a ``backend.properties``-equivalent
    that also works for targets without a legacy ``properties()`` object.
    """
    target = backend.target
    n = int(getattr(target, "num_qubits", 0) or 0)

    def _median(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    qprops = []
    qp_list = getattr(target, "qubit_properties", None)
    for q in range(n):
        qp = qp_list[q] if (qp_list and q < len(qp_list)) else None
        qprops.append({
            "qubit": q,
            "t1": getattr(qp, "t1", None) if qp else None,
            "t2": getattr(qp, "t2", None) if qp else None,
            "frequency": getattr(qp, "frequency", None) if qp else None,
        })

    op_stats: Dict[str, Dict] = {}
    readout: List[Dict] = []
    for op_name in sorted(getattr(target, "operation_names", []) or []):
        try:
            props = target[op_name]
        except Exception:
            continue
        if not props:
            continue
        errs, durs = [], []
        for qargs, inst_props in props.items():
            if inst_props is None:
                continue
            e = getattr(inst_props, "error", None)
            d = getattr(inst_props, "duration", None)
            if e is not None:
                errs.append(e)
            if d is not None:
                durs.append(d)
            if op_name == "measure" and e is not None:
                readout.append({
                    "qubit": (qargs[0] if qargs else None),
                    "error": e, "duration": d,
                })
        if errs or durs:
            op_stats[op_name] = {
                "n_calibrated": len(errs),
                "median_error": _median(errs),
                "min_error": min(errs) if errs else None,
                "max_error": max(errs) if errs else None,
                "median_duration_s": _median(durs),
            }

    return {
        "backend_name": getattr(backend, "name", None),
        "num_qubits": n,
        "dt": getattr(target, "dt", None),
        "operation_names": sorted(getattr(target, "operation_names", []) or []),
        "qubit_properties": qprops,
        "operation_error_stats": op_stats,
        "readout_error": readout,
    }


# --------------------------------------------------------------------------- #
# PUB construction (job mode, transpiled with a preset pass manager)            #
# --------------------------------------------------------------------------- #
@dataclass
class PubItem:
    arm: str
    setting_label: str
    readout: str
    shots: int
    creg_name: str


def build_job_pubs(plan: HardwareRunPlan, job: Dict, pass_manager) -> Tuple[list, List[PubItem]]:
    """Build the SamplerV2 PUB list (and metadata) for one instance's job.

    Transpiles each arm's circuits once with ``pass_manager`` (preset,
    optimization_level=1, target=backend.target), then assembles PUBs in the
    job's deterministic shuffled order with per-PUB shots.
    """
    inst = plan.instance_by_id(job["instance_id"])
    cf = CircuitFamily(inst)

    # transpile each arm's circuits once (efficient); index into isa lists.
    arm_settings: Dict[str, list] = {}
    arm_isa: Dict[str, list] = {}
    arm_creg: Dict[str, list] = {}
    for arm in plan.arms:
        if inst.instance_id not in arm.instance_ids:
            continue
        sc = getattr(cf, arm.builder)()
        arm_settings[arm.name] = sc
        cregs = [qc.cregs[0].name for _, qc in sc]
        arm_creg[arm.name] = cregs
        arm_isa[arm.name] = pass_manager.run([qc for _, qc in sc])

    pubs = []
    meta: List[PubItem] = []
    for (arm_name, s_idx, shots) in job["pubs"]:
        setting, _ = arm_settings[arm_name][s_idx]
        isa = arm_isa[arm_name][s_idx]
        pubs.append((isa, None, shots))
        meta.append(PubItem(arm=arm_name, setting_label=setting.label,
                            readout=setting.readout, shots=shots,
                            creg_name=arm_creg[arm_name][s_idx]))
    return pubs, meta


def counts_from_pub_result(pub_result, creg_name: str = "c") -> Dict[str, int]:
    """Extract counts from a SamplerV2 PUB result's classical register."""
    data = pub_result.data
    ba = getattr(data, creg_name, None)
    if ba is None:
        # fall back to the sole BitArray field
        try:
            fields = list(data.keys())          # DataBin mapping interface
        except Exception:
            fields = [f for f in getattr(data, "__dict__", {})]
        for f in fields:
            cand = data[f] if hasattr(data, "__getitem__") else getattr(data, f)
            if hasattr(cand, "get_counts"):
                ba = cand
                break
    return ba.get_counts()


# --------------------------------------------------------------------------- #
# scoring (same statistics as the dry run; per-arm shot counts)                 #
# --------------------------------------------------------------------------- #
def _gate(alpha_exp: float, m_family: int, n_inst: int) -> float:
    n_tests = max(1, m_family) * max(1, n_inst)
    return norm_ppf(1.0 - alpha_exp / (2.0 * n_tests))


def _z_offnull(chat: float, n_shots: int, eps: float = 1e-6) -> float:
    se = math.sqrt(max(eps, 1.0 - chat * chat) / n_shots)
    return chat / se


def certificate_correlators(counts_by_label: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {lab: pair_correlator_from_counts(c)[0]
            for lab, c in counts_by_label.items()}


def score_certificate(estimator: FrozenEstimator,
                      correlators: Dict[str, float]) -> Dict:
    verdict = estimator.decide(correlators)
    verdict["method"] = "certificate"
    return verdict


def score_structured(estimator: FrozenEstimator,
                     correlators: Dict[str, float]) -> Dict:
    """O6 bond-dim-2 comb LS fit on the certificate correlators (0 extra shots)."""
    dx = correlators.get("YX", 0.0)
    dy = -correlators.get("ZX", 0.0)
    gate = _gate(estimator.alpha_exp, 2, estimator.n_inst_declared)
    z = {"YX_component": _z_offnull(dx, estimator.n_shots),
         "ZX_component": _z_offnull(dy, estimator.n_shots)}
    arg = max(z, key=lambda k: abs(z[k]))
    return {
        "method": "structured_lowmem",
        "detected": bool(abs(z[arg]) > gate),
        "max_abs_z": abs(z[arg]),
        "argmax": arg,
        "gate": gate,
        "delta_hat": math.hypot(dx, dy),
    }


def score_marginal(estimator: FrozenEstimator,
                   atomic_counts: Dict[str, Tuple[str, int, str, Dict[str, int]]],
                   shots: int) -> Dict:
    """Marginal-QPT blindness null on the atomic off-null expectations.

    ``atomic_counts`` maps each atomic setting label to
    ``(prep, slot_open, pauli, counts)``.  Must FAIL to detect (Theorem 1).
    """
    offnull: Dict[str, float] = {}
    for label, (prep, slot_open, pauli, cnt) in atomic_counts.items():
        null = null_atomic_expectation(slot_open, prep, pauli)
        if abs(null) < 0.5:
            chat, _ = single_expectation_from_counts(cnt)
            offnull[label] = chat
    gate = _gate(estimator.alpha_exp, len(offnull), estimator.n_inst_declared)
    z = {k: _z_offnull(v, shots) for k, v in offnull.items()}
    if z:
        arg = max(z, key=lambda k: abs(z[k]))
        detected = abs(z[arg]) > gate
        max_z = abs(z[arg])
        max_abs_expect = max(abs(v) for v in offnull.values())
    else:
        arg, detected, max_z, max_abs_expect = None, False, 0.0, 0.0
    return {
        "method": "marginal_qpt",
        "detected": bool(detected),
        "max_abs_z": max_z,
        "argmax": arg,
        "gate": gate,
        "offnull_family_size": len(offnull),
        "max_abs_offnull_expectation": max_abs_expect,
        "note": "atomic marginals are dark-invariant (Theorem 1)",
    }


def score_heldout(estimator: FrozenEstimator,
                  pair_counts: Dict[str, Tuple[str, str, Dict[str, int]]],
                  shots: int) -> Dict:
    """Held-out depth-3 off-null detector (confirmation only)."""
    offnull: Dict[str, float] = {}
    for label, (p, q, cnt) in pair_counts.items():
        null = null_heldout_correlator(p, q)
        if abs(null) < 0.5:
            chat, _ = pair_correlator_from_counts(cnt)
            offnull[label] = chat
    gate = _gate(estimator.alpha_exp, len(offnull), estimator.n_inst_declared)
    z = {k: _z_offnull(v, shots) for k, v in offnull.items()}
    if z:
        arg = max(z, key=lambda k: abs(z[k]))
        detected = abs(z[arg]) > gate
        max_z = abs(z[arg])
    else:
        arg, detected, max_z = None, False, 0.0
    return {
        "method": "heldout_depth3",
        "detected": bool(detected),
        "max_abs_z": max_z,
        "argmax": arg,
        "gate": gate,
    }


def score_method_power(per_instance: List[Dict], instances: Sequence[ChallengeInstance]) -> Dict:
    """TP/FP/power/FP-rate against the (unblinded) answer key."""
    n_dark = sum(1 for i in instances if i.dark)
    n_null = sum(1 for i in instances if not i.dark)
    tp = sum(1 for v, inst in zip(per_instance, instances)
             if inst.dark and v["detected"])
    fp = sum(1 for v, inst in zip(per_instance, instances)
             if (not inst.dark) and v["detected"])
    return {
        "n_dark": n_dark, "n_null": n_null,
        "true_positives": tp, "false_positives": fp,
        "power": (tp / n_dark) if n_dark else float("nan"),
        "fp_rate": (fp / n_null) if n_null else float("nan"),
    }


__all__ = [
    "TARGET_BACKEND", "ESTIMATOR_PLATFORM", "N_INST_HW", "HW_MASTER_SEED",
    "DEVICE_READOUT_ASSIGNMENT_ERROR", "READOUT_CORRELATOR_QUBITS",
    "DELTA_RANGE", "ALPHA_EXP",
    "CERT_SETTINGS", "MARGINAL_SETTINGS", "HELDOUT_SETTINGS",
    "MARGINAL_SHOTS", "HELDOUT_SHOTS", "HELDOUT_INSTANCE_IDS",
    "PER_SHOT_OVERHEAD_S", "PER_JOB_OVERHEAD_S", "QPU_BUDGET_S",
    "Arm", "HardwareRunPlan", "PubItem",
    "calibration_snapshot", "build_job_pubs", "counts_from_pub_result",
    "certificate_correlators", "score_certificate", "score_structured",
    "score_marginal", "score_heldout", "score_method_power",
]
