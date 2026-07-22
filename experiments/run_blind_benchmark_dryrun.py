"""End-to-end SIMULATOR dry run of the C7 Step-2 blind benchmark.

This is a software dry run only -- no hardware credentials exist yet.  It
exercises the full preregistered pipeline of ``research/C7_THEOREM_PACKAGE.md``
Sec.7 against the qiskit-aer simulator:

    deterministic master seed
      -> 20 blind challenge instances (sealed answer key + public manifest)
      -> qiskit circuits (atomic testers, depth-2 correlators, held-out depth-3)
      -> qiskit-aer  (IBM Heron noise approximation: depolarizing + readout from
                      research/DEVICE_NOISE_PARAMETERS.md; also a noiseless control)
      -> frozen flat-extension certificate + four comparison baselines
      -> UNBLIND: open the sealed key, verify its sha256 against the manifest
      -> score: detection power / false-positive rate per method, resource
                ledgers, and an O7 drift diagnostic (circuit-order randomization
                + first-half/second-half consistency check).

Outputs a markdown score report and a JSON artifact under
``research/results/blind_benchmark_dryrun/``.

Run:  PYTHONPATH=src python experiments/run_blind_benchmark_dryrun.py
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import date
from typing import Callable, Dict, List, Tuple

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from cyz_m.blind_benchmark import (
    ChallengeGenerator, FrozenEstimator, CircuitFamily, ResourceLedger,
    pair_correlator_from_counts, PAULI_PAIRS, norm_ppf,
)
from cyz_m.benchmark_baselines import (
    marginal_qpt_baseline, restricted_ptt_baseline, full_ptt_baseline,
    structured_lowmem_baseline, null_heldout_correlator,
)

# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #
MASTER_SEED = 20260721                    # deterministic; matches Step-1
N_INST = 20
DELTA_RANGE = (0.2, 0.4)
N_SLOTS = 2
ALPHA_EXP = 0.01
TARGET_PLATFORM = "ibm_heron_r2"

# IBM Heron r2 approximation (verified numbers from DEVICE_NOISE_PARAMETERS.md /
# device_noise.PlatformModel.ibm_heron_r2)
HERON_1Q = 2.9e-4                          # 1Q depolarizing probability
HERON_2Q = 2.334e-3                        # 2Q (CZ) depolarizing probability
HERON_RO = 1.23e-3                         # symmetric readout assignment error

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "results", "blind_benchmark_dryrun")


# --------------------------------------------------------------------------- #
# aer backend + O7 order-randomizing runner                                    #
# --------------------------------------------------------------------------- #
def heron_noise_model() -> NoiseModel:
    nm = NoiseModel()
    one_q = ["h", "x", "y", "z", "s", "sdg", "r", "sx", "rz", "u", "id"]
    nm.add_all_qubit_quantum_error(depolarizing_error(HERON_1Q, 1), one_q)
    nm.add_all_qubit_quantum_error(depolarizing_error(HERON_2Q, 2), ["cz", "cx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - HERON_RO, HERON_RO], [HERON_RO, 1 - HERON_RO]]))
    return nm


class AerRunner:
    """qiskit-aer runner with O7 circuit-order randomization.

    Each ``run`` call shuffles the submitted circuits into a randomized
    execution order (time-randomization per honesty-ledger O7), executes them,
    and returns the counts in the *original* order.  Tallies total shots.
    """

    def __init__(self, noise_model=None, seed: int = 0) -> None:
        self.sim = AerSimulator(noise_model=noise_model)
        self.rng = random.Random(seed)
        self.total_shots = 0
        self.total_circuits = 0

    def run(self, circuits: List["object"], shots: int) -> List[Dict[str, int]]:
        n = len(circuits)
        order = list(range(n))
        self.rng.shuffle(order)                       # O7 time-randomization
        shuffled = [circuits[i] for i in order]
        result = self.sim.run(shuffled, shots=shots).result()
        counts_shuffled = [result.get_counts(i) for i in range(n)]
        out: List[Dict[str, int]] = [None] * n       # type: ignore
        for pos, orig in enumerate(order):
            out[orig] = counts_shuffled[pos]
        self.total_shots += n * shots
        self.total_circuits += n
        return out

    # baselines expect a plain callable ``runner(circuits, shots)``
    def __call__(self, circuits: List["object"], shots: int) -> List[Dict[str, int]]:
        return self.run(circuits, shots)


# --------------------------------------------------------------------------- #
# certificate + held-out evaluation per instance                               #
# --------------------------------------------------------------------------- #
def certificate_verdict(instance, estimator: FrozenEstimator, runner: AerRunner
                        ) -> Tuple[Dict, ResourceLedger, Dict[str, float]]:
    cf = CircuitFamily(instance)
    settings_circ = cf.depth2_correlators()
    circuits = [qc for _, qc in settings_circ]
    counts = runner.run(circuits, estimator.n_shots)
    correlators = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        c, _ = pair_correlator_from_counts(cnt)
        correlators[setting.label] = c
    verdict = estimator.decide(correlators)
    verdict["method"] = "certificate"
    ledger = ResourceLedger("certificate")
    ledger.add("depth2", len(settings_circ), estimator.n_shots)
    return verdict, ledger, correlators


def heldout_verdict(instance, estimator: FrozenEstimator, runner: AerRunner
                    ) -> Tuple[Dict, ResourceLedger]:
    """Held-out depth-3 validation: off-null detector on the deeper family."""
    cf = CircuitFamily(instance)
    settings_circ = cf.heldout_depth3()
    circuits = [qc for _, qc in settings_circ]
    counts = runner.run(circuits, estimator.n_shots)
    offnull = {}
    for (setting, _), cnt in zip(settings_circ, counts):
        p, q = setting.meta
        null = null_heldout_correlator(p, q)
        if abs(null) < 0.5:
            c, _ = pair_correlator_from_counts(cnt)
            se = math.sqrt(max(1e-6, 1.0 - c * c) / estimator.n_shots)
            offnull[setting.label] = c / se
    m = len(offnull)
    gate = norm_ppf(1.0 - estimator.alpha_exp / (2.0 * max(1, m) * estimator.n_inst_declared))
    if offnull:
        arg = max(offnull, key=lambda k: abs(offnull[k]))
        detected = abs(offnull[arg]) > gate
        max_z = abs(offnull[arg])
    else:
        detected, max_z = False, 0.0
    verdict = {"method": "heldout_depth3", "detected": bool(detected),
               "max_abs_z": max_z, "gate": gate}
    ledger = ResourceLedger("heldout_depth3")
    ledger.add("heldout_depth3", len(settings_circ), estimator.n_shots)
    return verdict, ledger


# --------------------------------------------------------------------------- #
# O7 drift diagnostic stub                                                      #
# --------------------------------------------------------------------------- #
def drift_diagnostic(instances, estimator: FrozenEstimator, runner: AerRunner
                     ) -> Dict:
    """First-half / second-half consistency check on the certificate probe.

    For each dark instance, run the argmax off-null correlator twice with
    independent half-budgets (order randomized by the runner) and compare.  On
    the i.i.d. simulator this passes trivially; on hardware it is the O7 guard
    against calibration drift / queue nonstationarity.  A real run would use a
    martingale / block-bootstrap test -- this is the harness stub.
    """
    half = max(256, estimator.n_shots // 2)
    records = []
    worst = 0.0
    for inst in instances:
        if not inst.dark:
            continue
        cf = CircuitFamily(inst)
        # probe both YX and ZX (the axis-invariant signal subspace)
        probes = [cf.comb_correlator_circuit("|+>", "|+>", p, q)
                  for (p, q) in (("Y", "X"), ("Z", "X"))]
        c_first = runner.run(probes, half)
        c_second = runner.run(probes, half)
        for idx, lab in enumerate(("YX", "ZX")):
            a, _ = pair_correlator_from_counts(c_first[idx])
            b, _ = pair_correlator_from_counts(c_second[idx])
            se = math.sqrt(max(1e-6, 1.0 - a * a) / half) + \
                math.sqrt(max(1e-6, 1.0 - b * b) / half)
            z = abs(a - b) / se if se > 0 else 0.0
            worst = max(worst, z)
            records.append({"instance": inst.instance_id, "pair": lab,
                            "c_first": a, "c_second": b, "drift_z": z})
    # consistency gate: |C1 - C2| within 3 combined sigma
    return {
        "n_probes": len(records),
        "worst_drift_z": worst,
        "drift_gate_z": 3.0,
        "consistent": worst < 3.0,
        "half_shots": half,
        "note": "i.i.d. simulator -> trivially consistent; hardware needs a "
                "martingale/block-bootstrap test (O7).",
        "records": records,
    }


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def score_method(per_instance: List[Dict], instances) -> Dict:
    n_dark = sum(1 for i in instances if i.dark)
    n_null = sum(1 for i in instances if not i.dark)
    tp = sum(1 for v, inst in zip(per_instance, instances) if inst.dark and v["detected"])
    fp = sum(1 for v, inst in zip(per_instance, instances) if (not inst.dark) and v["detected"])
    return {
        "n_dark": n_dark,
        "n_null": n_null,
        "true_positives": tp,
        "false_positives": fp,
        "power": (tp / n_dark) if n_dark else float("nan"),
        "fp_rate": (fp / n_null) if n_null else float("nan"),
    }


# --------------------------------------------------------------------------- #
# main driver                                                                  #
# --------------------------------------------------------------------------- #
def run_condition(condition: str, instances, estimator: FrozenEstimator,
                  runner: AerRunner) -> Dict:
    """Run all methods on all instances for one noise condition."""
    methods = ["certificate", "marginal_qpt", "restricted_ptt", "full_ptt",
               "structured_lowmem", "heldout_depth3"]
    verdicts: Dict[str, List[Dict]] = {m: [] for m in methods}
    ledgers: Dict[str, ResourceLedger] = {}

    for inst in instances:
        cv, cl, _ = certificate_verdict(inst, estimator, runner)
        verdicts["certificate"].append(cv)
        ledgers.setdefault("certificate", cl)

        mv, ml = marginal_qpt_baseline(inst, estimator, runner)
        verdicts["marginal_qpt"].append(mv)
        ledgers.setdefault("marginal_qpt", ml)

        rv, rl = restricted_ptt_baseline(inst, estimator, runner)
        verdicts["restricted_ptt"].append(rv)
        ledgers.setdefault("restricted_ptt", rl)

        fv, fl = full_ptt_baseline(inst, estimator, runner)
        verdicts["full_ptt"].append(fv)
        ledgers.setdefault("full_ptt", fl)

        sv, sl = structured_lowmem_baseline(inst, estimator, runner)
        verdicts["structured_lowmem"].append(sv)
        ledgers.setdefault("structured_lowmem", sl)

        hv, hl = heldout_verdict(inst, estimator, runner)
        verdicts["heldout_depth3"].append(hv)
        ledgers.setdefault("heldout_depth3", hl)

    scores = {m: score_method(verdicts[m], instances) for m in methods}
    return {
        "condition": condition,
        "scores": scores,
        "ledgers": {m: ledgers[m].as_dict() for m in methods},
        "verdicts": {m: verdicts[m] for m in methods},
    }


def build_markdown(report: Dict) -> str:
    md: List[str] = []
    md.append("# C7 Step-2 blind benchmark -- simulator dry run")
    md.append("")
    meta = report["metadata"]
    md.append(f"- Date: {meta['date']}")
    md.append(f"- Master seed: {meta['master_seed']} (deterministic)")
    md.append(f"- Instances: {meta['n_inst']}  "
              f"(dark {meta['n_dark']} / null {meta['n_null']}, "
              f"delta in {meta['delta_range']}, {meta['n_slots']}-slot comb)")
    md.append(f"- Backend: qiskit {meta['qiskit_version']}, "
              f"qiskit-aer {meta['aer_version']}")
    md.append(f"- IBM Heron approximation: 1Q depol {HERON_1Q}, 2Q depol "
              f"{HERON_2Q}, readout {HERON_RO} (DEVICE_NOISE_PARAMETERS.md)")
    md.append("")
    md.append("## Sealed manifest")
    md.append("")
    md.append(f"- sealed_key_sha256: `{meta['sealed_key_sha256']}`")
    md.append(f"- unblind sha256 verification: **{meta['unblind_verified']}**")
    md.append(f"- frozen estimator config_sha256: `{meta['estimator_config_sha256']}`")
    md.append(f"- frozen z_gate: {meta['z_gate']:.4f}  |  n_shots/setting: "
              f"{meta['n_shots']}  |  alpha_exp: {meta['alpha_exp']}")
    md.append(f"- Step-1 artifact sha256: `{meta['step1_artifact_sha256']}`  "
              f"(n_ref={meta['n_ref_step1']}, power_factor={meta['power_factor']:.3f})")
    md.append("")

    order = ["certificate", "structured_lowmem", "marginal_qpt",
             "restricted_ptt", "full_ptt", "heldout_depth3"]
    for cond in report["conditions"]:
        md.append(f"## Score table -- {cond['condition']}")
        md.append("")
        md.append("| method | power | FP rate | TP/dark | FP/null | "
                  "settings | shots/inst | op-basis |")
        md.append("|---|---|---|---|---|---|---|---|")
        for m in order:
            s = cond["scores"][m]
            l = cond["ledgers"][m]
            opb = l["extra"].get("operation_basis_size",
                                 l["extra"].get("operation_basis", "-"))
            md.append(
                f"| {m} | {s['power']:.2f} | {s['fp_rate']:.2f} | "
                f"{s['true_positives']}/{s['n_dark']} | "
                f"{s['false_positives']}/{s['n_null']} | "
                f"{l['distinct_settings']} | {l['total_shots']} | {opb} |")
        md.append("")

    md.append("## O7 drift diagnostic (circuit-order randomization)")
    md.append("")
    for cond in report["conditions"]:
        d = cond["drift"]
        md.append(f"- {cond['condition']}: {d['n_probes']} probes, worst "
                  f"drift-z {d['worst_drift_z']:.2f} (gate {d['drift_gate_z']}), "
                  f"consistent: **{d['consistent']}**")
    md.append("")
    md.append("## Headlines")
    md.append("")
    for h in report["headlines"]:
        md.append(f"- {h}")
    md.append("")
    return "\n".join(md)


def main() -> None:
    import qiskit
    import qiskit_aer
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. generate blind challenges + seal
    gen = ChallengeGenerator(MASTER_SEED, N_INST, DELTA_RANGE, N_SLOTS)
    instances = gen.instances
    sealed_path = os.path.join(OUT_DIR, "sealed_key.json")
    sealed_sha = gen.write_sealed_key(sealed_path)
    manifest = gen.public_manifest()
    with open(os.path.join(OUT_DIR, "public_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # 2. freeze the estimator (thresholds from Step-1 artifacts) BEFORE any data
    estimator = FrozenEstimator(target_platform=TARGET_PLATFORM,
                                min_detectable_delta=DELTA_RANGE[0],
                                n_inst_declared=N_INST, alpha_exp=ALPHA_EXP)
    est_config = estimator.config
    est_config_sha = estimator.config_sha256()
    with open(os.path.join(OUT_DIR, "frozen_estimator_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": est_config, "config_sha256": est_config_sha}, fh, indent=2)

    # 3. run both noise conditions
    conditions = []
    for cond_name, nm in [("noiseless", None),
                          ("ibm_heron_approx", heron_noise_model())]:
        runner = AerRunner(noise_model=nm, seed=MASTER_SEED)
        res = run_condition(cond_name, instances, estimator, runner)
        res["drift"] = drift_diagnostic(instances, estimator, runner)
        res["total_circuits_executed"] = runner.total_circuits
        res["total_shots_executed"] = runner.total_shots
        conditions.append(res)

    # 4. UNBLIND: verify the sealed key against the manifest sha256
    ok, sealed_payload = ChallengeGenerator.verify_sealed_key(
        sealed_path, manifest["sealed_key_sha256"])

    n_dark = sum(1 for i in instances if i.dark)
    n_null = N_INST - n_dark

    # 5. headlines
    headlines = []
    for cond in conditions:
        sc = cond["scores"]
        headlines.append(
            f"[{cond['condition']}] certificate power {sc['certificate']['power']:.2f}"
            f" / FP {sc['certificate']['fp_rate']:.2f} at "
            f"{cond['ledgers']['certificate']['distinct_settings']} settings; "
            f"marginal-QPT power {sc['marginal_qpt']['power']:.2f} "
            f"(blind by Theorem 1); full-PTT power {sc['full_ptt']['power']:.2f} "
            f"at {cond['ledgers']['full_ptt']['extra'].get('operation_basis_size')} "
            f"operation-basis ops.")
    headlines.append(
        f"Resource separation: certificate "
        f"{conditions[0]['ledgers']['certificate']['distinct_settings']} settings vs "
        f"full-PTT {conditions[0]['ledgers']['full_ptt']['extra']['operation_basis_size']}"
        f" (=d^4k) vs restricted-PTT "
        f"{conditions[0]['ledgers']['restricted_ptt']['distinct_settings']}; the "
        f"structured O6 comparator matches the certificate at "
        f"{conditions[0]['ledgers']['structured_lowmem']['distinct_settings']} settings.")

    report = {
        "metadata": {
            "date": str(date.today()),
            "master_seed": MASTER_SEED,
            "n_inst": N_INST,
            "n_dark": n_dark,
            "n_null": n_null,
            "delta_range": list(DELTA_RANGE),
            "n_slots": N_SLOTS,
            "alpha_exp": ALPHA_EXP,
            "target_platform": TARGET_PLATFORM,
            "qiskit_version": qiskit.__version__,
            "aer_version": qiskit_aer.__version__,
            "heron_params": {"err_1q": HERON_1Q, "err_2q": HERON_2Q,
                             "readout": HERON_RO},
            "sealed_key_sha256": sealed_sha,
            "manifest_sealed_key_sha256": manifest["sealed_key_sha256"],
            "unblind_verified": bool(ok),
            "estimator_config_sha256": est_config_sha,
            "z_gate": estimator.z_gate,
            "n_shots": estimator.n_shots,
            "step1_artifact_sha256": est_config["step1_artifact_sha256"],
            "n_ref_step1": est_config["n_ref_step1"],
            "power_factor": est_config["power_factor"],
            "runtime_seconds": None,     # filled below
        },
        "frozen_estimator_config": est_config,
        "public_manifest": manifest,
        "answer_key": [inst.answer_key() for inst in instances],
        "conditions": conditions,
        "headlines": headlines,
    }
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT_DIR, "blind_benchmark_dryrun.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = build_markdown(report)
    with open(os.path.join(OUT_DIR, "blind_benchmark_dryrun.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print(md)
    print("=" * 78)
    print(f"sealed key      : {sealed_path}")
    print(f"unblind verified: {ok}")
    print(f"JSON artifact   : {os.path.join(OUT_DIR, 'blind_benchmark_dryrun.json')}")
    print(f"runtime         : {report['metadata']['runtime_seconds']}s")


if __name__ == "__main__":
    main()
