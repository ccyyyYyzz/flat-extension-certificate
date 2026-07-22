"""IBM-hardware runner for the C7 Step-2 blind benchmark.

This is the hardware counterpart of ``run_blind_benchmark_dryrun.py``.  It runs
the identical blind-discipline pipeline (deterministic sealed challenges, frozen
flat-extension certificate, resource ledger, same scoring) through a
``qiskit-ibm-runtime`` SamplerV2 in **job mode** against IBM ``ibm_marrakesh``.

SAFETY / how to run
-------------------
* Default (NO ``--arm``): a full DRESS REHEARSAL.  The exact same PUB/transpile/
  submit code path runs against ``AerSimulator.from_backend(backend)`` (a
  noise-model-from-device local simulator; qiskit-ibm-runtime local testing
  mode).  NOTHING is submitted to real hardware.  Outputs are written under
  ``research/results/blind_benchmark_hardware_rehearsal/`` with a ``rehearsal_``
  prefix.  If the saved IBM account / backend cannot be fetched, it falls back
  to a generic Heron noise model and says so.

* ``--arm``: the ONLY switch that submits to real hardware.  Without it the
  runner refuses to touch the QPU.  (This script is delivered pre-armed=False;
  an operator must pass ``--arm`` deliberately.)

Run (rehearsal):
    PYTHONPATH=src python experiments/run_blind_benchmark_hardware.py
Run (REAL hardware, operator-initiated only):
    PYTHONPATH=src python experiments/run_blind_benchmark_hardware.py --arm
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from cyz_m.blind_benchmark import ChallengeGenerator, CircuitFamily
from cyz_m.hardware_runner import (
    HardwareRunPlan, HW_MASTER_SEED, N_INST_HW, TARGET_BACKEND,
    calibration_snapshot, build_job_pubs, counts_from_pub_result,
    certificate_correlators, score_certificate, score_structured,
    score_marginal, score_heldout, score_method_power,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARDWARE_DIR = os.path.join(_REPO, "research", "results", "blind_benchmark_hardware")
REHEARSAL_DIR = os.path.join(_REPO, "research", "results",
                             "blind_benchmark_hardware_rehearsal")


# --------------------------------------------------------------------------- #
# backend acquisition (read-only) + rehearsal simulator                        #
# --------------------------------------------------------------------------- #
def fetch_real_backend(target_backend: str):
    """Read-only fetch of the target IBM backend from the saved account."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.backend(target_backend)
    return backend, service


def make_sampler(mode):
    """Construct a SamplerV2 bound to ``mode`` (real backend -> submits; local
    AerSimulator -> local testing mode, no submission).  Module-level so tests
    can patch it to assert the no-``--arm`` refusal path never binds a real
    backend."""
    from qiskit_ibm_runtime import SamplerV2
    return SamplerV2(mode=mode)


def rehearsal_backend(target_backend: str) -> Tuple[object, object, str]:
    """Build the local dress-rehearsal simulator.

    Returns ``(transpile_backend, exec_backend, source)`` where
    ``transpile_backend`` provides the ``.target`` for the preset pass manager and
    ``exec_backend`` is the (noisy) AerSimulator used by the local SamplerV2.
    """
    from qiskit_aer import AerSimulator
    try:
        backend, _ = fetch_real_backend(target_backend)
        sim = AerSimulator.from_backend(backend)
        return backend, sim, f"AerSimulator.from_backend({backend.name})"
    except Exception as exc:                                  # pragma: no cover
        # fall back to a generic Heron noise model (real fetch failed)
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
        nm = NoiseModel()
        one_q = ["sx", "x", "rz", "id", "h", "s", "sdg", "r", "u"]
        nm.add_all_qubit_quantum_error(depolarizing_error(2.9e-4, 1), one_q)
        nm.add_all_qubit_quantum_error(depolarizing_error(2.334e-3, 2), ["cz", "cx"])
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - 1.23e-3, 1.23e-3], [1.23e-3, 1 - 1.23e-3]]))
        sim = AerSimulator(noise_model=nm)
        return sim, sim, (f"generic Heron noise model (real backend fetch "
                          f"failed: {type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# execution (job mode: one SamplerV2 job per instance)                         #
# --------------------------------------------------------------------------- #
def execute_plan(plan: HardwareRunPlan, transpile_backend, sampler,
                 armed: bool) -> Tuple[Dict, List[Dict]]:
    """Submit one job per instance and collect counts.

    ``sampler`` is a ``qiskit_ibm_runtime.SamplerV2`` bound either to the real
    backend (armed) or to a local AerSimulator (rehearsal).  Returns
    ``(counts, job_records)`` with ``counts[(inst, arm, label)] -> counts_dict``.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1,
                                      target=transpile_backend.target)

    counts: Dict[Tuple[int, str, str], Dict[str, int]] = {}
    job_records: List[Dict] = []
    for job in plan.jobs:
        pubs, meta = build_job_pubs(plan, job, pm)
        submit_ts = datetime.now(timezone.utc).isoformat()
        job_obj = sampler.run(pubs)                # SUBMIT (armed) or local run
        job_id = job_obj.job_id()
        result = job_obj.result()
        recv_ts = datetime.now(timezone.utc).isoformat()
        try:
            usage = job_obj.usage()
        except Exception:                          # pragma: no cover
            usage = None
        for i, pi in enumerate(meta):
            counts[(job["instance_id"], pi.arm, pi.setting_label)] = \
                counts_from_pub_result(result[i], pi.creg_name)
        job_records.append({
            "instance_id": job["instance_id"],
            "job_id": job_id,
            "armed": armed,
            "n_pubs": len(pubs),
            "shuffle_seed": job["shuffle_seed"],
            "pub_order": job["pub_order"],
            "submit_ts_utc": submit_ts,
            "result_ts_utc": recv_ts,
            "usage_qpu_s": usage,
        })
    return counts, job_records


# --------------------------------------------------------------------------- #
# scoring across all instances                                                 #
# --------------------------------------------------------------------------- #
def score_run(plan: HardwareRunPlan, counts: Dict) -> Dict:
    est = plan.estimator
    instances = plan.instances

    cert_v, struct_v, marg_v = [], [], []
    heldout_v, heldout_instances = [], []

    for inst in instances:
        # certificate depth-2 correlators
        cert_counts = {lab: counts[(inst.instance_id, "certificate", lab)]
                       for (iid, arm, lab) in counts
                       if iid == inst.instance_id and arm == "certificate"}
        corr = certificate_correlators(cert_counts)
        cert_v.append(score_certificate(est, corr))
        struct_v.append(score_structured(est, corr))

        # marginal-QPT atomic off-null
        cf = CircuitFamily(inst)
        atomic_meta = {s.label: s.meta for s, _ in cf.atomic_testers()}
        atomic_counts = {}
        for (iid, arm, lab), cnt in counts.items():
            if iid == inst.instance_id and arm == "marginal_qpt":
                prep, slot_open, pauli = atomic_meta[lab]
                atomic_counts[lab] = (prep, slot_open, pauli, cnt)
        marg_v.append(score_marginal(est, atomic_counts, plan.arms[1].shots))

        # held-out depth-3 (only planned instances)
        held_labels = [(iid, arm, lab) for (iid, arm, lab) in counts
                       if iid == inst.instance_id and arm == "heldout_depth3"]
        if held_labels:
            held_meta = {s.label: s.meta for s, _ in cf.heldout_depth3()}
            pair_counts = {}
            for (iid, arm, lab) in held_labels:
                p, q = held_meta[lab]
                pair_counts[lab] = (p, q, counts[(iid, arm, lab)])
            heldout_v.append(score_heldout(est, pair_counts, plan.arms[2].shots))
            heldout_instances.append(inst)

    scores = {
        "certificate": score_method_power(cert_v, instances),
        "structured_lowmem": score_method_power(struct_v, instances),
        "marginal_qpt": score_method_power(marg_v, instances),
    }
    if heldout_instances:
        scores["heldout_depth3"] = score_method_power(heldout_v, heldout_instances)

    return {
        "scores": scores,
        "verdicts": {
            "certificate": cert_v,
            "structured_lowmem": struct_v,
            "marginal_qpt": marg_v,
            "heldout_depth3": heldout_v,
        },
    }


# --------------------------------------------------------------------------- #
# resource ledger for the whole run                                            #
# --------------------------------------------------------------------------- #
def resource_ledger(plan: HardwareRunPlan) -> Dict:
    t = plan.totals()
    return {
        "target_backend": plan.target_backend,
        "job_count": t["job_count"],
        "total_pubs": t["total_pubs"],
        "total_shots": t["total_shots"],
        "by_arm": t["by_arm"],
        "structured_lowmem": {"extra_shots": 0,
                              "note": "analysis-only on certificate counts"},
        "full_ptt": {"extra_shots": 0, "note": "simulator-only; excluded"},
        "qpu_seconds_estimate": plan.qpu_seconds_estimate(),
    }


# --------------------------------------------------------------------------- #
# markdown report                                                              #
# --------------------------------------------------------------------------- #
def build_markdown(meta: Dict, plan: HardwareRunPlan, scored: Dict,
                   job_records: List[Dict]) -> str:
    md: List[str] = []
    tag = "REHEARSAL" if not meta["armed"] else "HARDWARE"
    md.append(f"# C7 Step-2 blind benchmark -- {tag} run")
    md.append("")
    md.append(f"- Mode: **{tag}** ({meta['exec_source']})")
    md.append(f"- Date: {meta['date']}  |  target backend: {plan.target_backend}")
    md.append(f"- Master seed: {plan.master_seed} (fresh; dry-run used 20260721)")
    md.append(f"- Instances: {plan.n_inst} "
              f"(dark {meta['n_dark']} / null {meta['n_null']})")
    md.append(f"- qiskit {meta['qiskit_version']}, "
              f"qiskit-ibm-runtime {meta['ibm_runtime_version']}, "
              f"qiskit-aer {meta['aer_version']}")
    md.append("")
    md.append("## Sealed manifest (sealed BEFORE execution)")
    md.append("")
    md.append(f"- sealed_key_sha256: `{meta['sealed_key_sha256']}`")
    md.append(f"- unblind sha256 verification: **{meta['unblind_verified']}**")
    md.append(f"- plan_sha256: `{meta['plan_sha256']}`")
    md.append(f"- frozen estimator config_sha256: `{meta['estimator_config_sha256']}`")
    _cfg = plan.estimator.config
    md.append(f"- frozen z_gate: {plan.estimator.z_gate:.4f}  |  "
              f"cert shots/setting: {plan.cert_shots}  |  alpha_exp: "
              f"{plan.estimator.alpha_exp}")
    if _cfg.get("config_version"):
        md.append(f"- estimator config v{_cfg['config_version']} "
                  f"(decision family: {_cfg['decision_family']}, "
                  f"pairs {_cfg.get('signal_subspace_pairs')}); "
                  f"z_gate_shot={_cfg['z_gate_shot']:.4f} inflated by "
                  f"1/lambda={1.0/_cfg['readout_attenuation_lambda']:.4f} "
                  f"for readout eps={_cfg['readout_assignment_error']:.4f} "
                  f"-> device-aware z_gate={_cfg['z_gate']:.4f}")
    md.append("")
    md.append(plan.preflight_markdown())
    md.append("## Score table")
    md.append("")
    md.append("| method | power | FP rate | TP/dark | FP/null | note |")
    md.append("|---|---|---|---|---|---|")
    order = [("certificate", "9 settings x 2048 (decision family)"),
             ("structured_lowmem", "analysis-only on cert counts (O6)"),
             ("marginal_qpt", "36 settings x 512 (blindness null; must be 0)"),
             ("heldout_depth3", "9 settings x 2048 on 2 instances (confirmation)")]
    for m, note in order:
        s = scored["scores"].get(m)
        if s is None:
            continue
        md.append(f"| {m} | {s['power']:.2f} | {s['fp_rate']:.2f} | "
                  f"{s['true_positives']}/{s['n_dark']} | "
                  f"{s['false_positives']}/{s['n_null']} | {note} |")
    md.append("")
    md.append("## O7 drift discipline")
    md.append("")
    md.append(f"- Deterministic per-job PUB interleave (shuffle seeds recorded); "
              f"{len(job_records)} jobs, submission + result timestamps recorded.")
    # Label the snapshot with the ACTUAL backend it was taken from (may be a
    # fallback generic simulator, not plan.target_backend, if the real fetch
    # failed) rather than hard-coding the target name.
    cal_backend = meta.get("calibration_backend") or plan.target_backend
    md.append(f"- Calibration snapshot ({cal_backend}.target) recorded "
              f"pre-submission.")
    md.append("")
    md.append("## Resource ledger")
    md.append("")
    led = meta["resource_ledger"]
    md.append(f"- Jobs: {led['job_count']}  |  PUBs: {led['total_pubs']}  |  "
              f"total shots: {led['total_shots']}")
    for arm, b in led["by_arm"].items():
        md.append(f"  - {arm}: {b['instances']} inst x {b['settings']} settings "
                  f"x {b['shots_per_setting']} shots = {b['shots']} shots")
    md.append(f"- Structured comparator: 0 extra shots (analysis-only).  "
              f"Full PTT: simulator-only (excluded).")
    md.append("")
    md.append("## Jobs")
    md.append("")
    md.append("| instance | job_id | PUBs | usage QPU-s | submit (UTC) |")
    md.append("|---|---|---|---|---|")
    for jr in job_records:
        md.append(f"| {jr['instance_id']} | `{jr['job_id']}` | {jr['n_pubs']} | "
                  f"{jr['usage_qpu_s']} | {jr['submit_ts_utc']} |")
    md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# CLI / safety-gate seams (kept small + pure for testing)                       #
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="store_true",
                    help="REQUIRED to submit to real IBM hardware.  Without it "
                         "the full pipeline runs against a local "
                         "AerSimulator.from_backend dress rehearsal.")
    ap.add_argument("--master-seed", type=int, default=HW_MASTER_SEED)
    ap.add_argument("--n-inst", type=int, default=N_INST_HW)
    ap.add_argument("--target-backend", default=TARGET_BACKEND)
    return ap


def will_submit_to_hardware(arm: bool) -> bool:
    """The sole hardware gate: submission happens iff ``--arm`` was supplied."""
    return bool(arm)


def resolve_output(armed: bool):
    """(out_dir, prefix) for the armed vs rehearsal path."""
    if armed:
        return HARDWARE_DIR, ""
    return REHEARSAL_DIR, "rehearsal_"


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    t0 = time.time()
    import qiskit
    import qiskit_aer
    import qiskit_ibm_runtime

    plan = HardwareRunPlan(master_seed=args.master_seed, n_inst=args.n_inst,
                           target_backend=args.target_backend)

    # ---- pre-flight (always print; assert under the 7-min reserve) --------- #
    print(plan.preflight_markdown())
    q = plan.qpu_seconds_estimate()
    assert q["under_budget"], (
        f"estimated QPU seconds {q['high_s']:.1f}s exceeds the "
        f"{q['budget_s']}s (7-min) Open-plan reserve")

    # ---- choose path (SAFETY: --arm is the ONLY hardware gate) ------------- #
    armed = will_submit_to_hardware(args.arm)
    out_dir, prefix = resolve_output(armed)
    if armed:
        # === REAL HARDWARE SUBMISSION (operator-initiated) ================= #
        print("\n[ARMED] --arm supplied: submitting to REAL hardware "
              f"{args.target_backend}.")
        backend, _service = fetch_real_backend(args.target_backend)
        transpile_backend = backend
        sampler = make_sampler(mode=backend)
        exec_source = f"IBM hardware {backend.name}"
    else:
        # === DRESS REHEARSAL (no submission) =============================== #
        print("\n[REHEARSAL] no --arm: running the full pipeline against a local "
              "AerSimulator.from_backend (no job submitted).")
        transpile_backend, exec_backend, exec_source = \
            rehearsal_backend(args.target_backend)
        sampler = make_sampler(mode=exec_backend)  # local testing mode -> no submit

    os.makedirs(out_dir, exist_ok=True)
    print(f"[exec source] {exec_source}")

    # ---- SEAL + write manifest / config / plan / calibration BEFORE exec --- #
    sealed_path = os.path.join(out_dir, prefix + "sealed_key.json")
    sealed_sha = plan.gen.write_sealed_key(sealed_path)
    manifest = plan.gen.public_manifest()
    manifest["target_backend"] = args.target_backend
    manifest["plan_sha256"] = plan.plan_sha256()
    manifest["armed"] = armed
    with open(os.path.join(out_dir, prefix + "public_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(out_dir, prefix + "frozen_estimator_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"config": plan.estimator.config,
                   "config_sha256": plan.estimator.config_sha256()}, fh, indent=2)
    with open(os.path.join(out_dir, prefix + "run_plan.json"), "w",
              encoding="utf-8") as fh:
        json.dump(plan.as_dict(), fh, indent=2)

    # calibration snapshot from the transpile backend's target (pre-submission)
    try:
        cal = calibration_snapshot(transpile_backend)
    except Exception as exc:                       # pragma: no cover
        cal = {"error": f"{type(exc).__name__}: {exc}"}
    with open(os.path.join(out_dir, prefix + "calibration_snapshot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)

    print(f"[sealed] {sealed_path}  sha256={sealed_sha[:16]}...  "
          f"(sealed BEFORE execution)")

    # ---- execute (job mode) ----------------------------------------------- #
    counts, job_records = execute_plan(plan, transpile_backend, sampler, armed)

    # ---- UNBLIND + score --------------------------------------------------- #
    ok, _payload = ChallengeGenerator.verify_sealed_key(
        sealed_path, manifest["sealed_key_sha256"])
    scored = score_run(plan, counts)

    n_dark = sum(1 for i in plan.instances if i.dark)
    n_null = plan.n_inst - n_dark

    meta = {
        "armed": armed,
        "exec_source": exec_source,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "target_backend": args.target_backend,
        "calibration_backend": (cal.get("backend_name")
                                if isinstance(cal, dict) else None),
        "master_seed": plan.master_seed,
        "n_inst": plan.n_inst,
        "n_dark": n_dark,
        "n_null": n_null,
        "qiskit_version": qiskit.__version__,
        "aer_version": qiskit_aer.__version__,
        "ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "sealed_key_sha256": sealed_sha,
        "unblind_verified": bool(ok),
        "plan_sha256": plan.plan_sha256(),
        "estimator_config_sha256": plan.estimator.config_sha256(),
        "resource_ledger": resource_ledger(plan),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    report = {
        "metadata": meta,
        "run_plan": plan.as_dict(),
        "public_manifest": manifest,
        "answer_key": [inst.answer_key() for inst in plan.instances],
        "scores": scored["scores"],
        "verdicts": scored["verdicts"],
        "job_records": job_records,
        "calibration_snapshot": cal,
    }
    with open(os.path.join(out_dir, prefix + "blind_benchmark_hardware.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = build_markdown(meta, plan, scored, job_records)
    with open(os.path.join(out_dir, prefix + "blind_benchmark_hardware.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print("=" * 78)
    print(f"mode            : {'ARMED (hardware)' if armed else 'REHEARSAL (local)'}")
    print(f"exec source     : {exec_source}")
    print(f"unblind verified: {ok}")
    print(f"out dir         : {out_dir}")
    print(f"runtime         : {meta['runtime_seconds']}s")


if __name__ == "__main__":
    main()
