#!/usr/bin/env python
"""BATCH-QUEUE all 8 v3 instance jobs on ibm_fez, then collect + score.

Companion to ``run_blind_benchmark_v3_fez.py``.  The original ``--arm`` runner
submits one instance job at a time and blocks on each result before submitting
the next; when the public queue is growing that puts every later instance at the
back of an ever-longer line.  This runner keeps the already-queued instance-0 job
and IMMEDIATELY batch-submits the remaining 7 (instances 1-7) so they all queue
in parallel, then waits for all 8 and scores exactly as the original design
(v3 differential + v2-absolute rescoring side by side).

Deterministic + blind-safe: circuits, layouts, and per-job PUB interleave are
deterministic functions of the sealed master seed 20260729 and the frozen v3
config (already sealed in ``research/results/blind_benchmark_v3_fez/manifest.json``
before any submission).  Nothing about blinding changes -- this is scheduling only.

Idempotent: re-running after a crash detects the already-submitted jobs (exactly
8 v3 jobs, one per instance) and skips resubmission.

    PYTHONPATH=src python experiments/run_v3_batch_collect.py
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import random
import time
from collections import Counter

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from cyz_m.blind_benchmark import ChallengeGenerator, FrozenEstimator
import qiskit
import qiskit_aer
import qiskit_ibm_runtime

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_HERE, "run_blind_benchmark_v3_fez.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)

OUT = v3.OUT_DIR
EXISTING_JOB0 = "d9fr679htsac739fr59g"          # instance-0 job already queued
CUT = datetime.datetime(2026, 7, 21, 19, 55,
                        tzinfo=datetime.timezone(datetime.timedelta(hours=2)))


def main() -> None:
    t0 = time.time()
    svc = QiskitRuntimeService()
    backend = svc.backend("ibm_fez")
    pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
    gen = ChallengeGenerator(v3.MASTER_SEED, v3.N_INST, v3.DELTA_RANGE,
                             v3.N_SLOTS, v3.DARK_PROB)
    instances = gen.instances
    v3cfg = v3.v3_gate_config()
    v2_est = FrozenEstimator(
        target_platform="ibm_heron_r2", min_detectable_delta=v3.DELTA_RANGE[0],
        n_inst_declared=v3.V2_N_INST_DECLARED, alpha_exp=v3.ALPHA_EXP,
        signal_subspace=True,
        readout_assignment_error=v3.V2_READOUT_ASSIGNMENT_ERROR,
        readout_correlator_qubits=v3.READOUT_CORRELATOR_QUBITS)
    manifest = json.load(open(os.path.join(OUT, "manifest.json")))

    def build_pubs(instance, theta_floor=None):
        ref_inst = v3.reference_instance(instance.instance_id)
        ch = v3.build_correlator_circuits(instance, theta_floor)
        ref = v3.build_correlator_circuits(ref_inst, theta_floor)
        ch_isa = pm.run([qc for _, qc in ch])
        ref_isa = pm.run([qc for _, qc in ref])
        layout = {"challenge": {}, "reference": {}}
        pubs = []; pub_index = []
        for (lab, _), isa in zip(ch, ch_isa):
            layout["challenge"][lab] = v3.physical_qubits(isa)
            pubs.append((isa, None, v3.CERT_SHOTS)); pub_index.append(("challenge", lab))
        for (lab, _), isa in zip(ref, ref_isa):
            layout["reference"][lab] = v3.physical_qubits(isa)
            pubs.append((isa, None, v3.CERT_SHOTS)); pub_index.append(("reference", lab))
        ss = int(v3.sha256_hex(f"{v3.MASTER_SEED}:job{instance.instance_id}")[:8], 16)
        rng = random.Random(ss); order = list(range(len(pubs))); rng.shuffle(order)
        return ([pubs[k] for k in order], [pub_index[k] for k in order],
                layout, ss, order)

    def collect(result, shuffled_index):
        ch_bits = {}; ref_bits = {}
        for i, (block, lab) in enumerate(shuffled_index):
            data = result[i].data
            ba = getattr(data, "c", None)
            if ba is None:
                for f in list(getattr(data, "__dict__", {})) or []:
                    cand = getattr(data, f)
                    if hasattr(cand, "get_bitstrings"):
                        ba = cand; break
            (ch_bits if block == "challenge" else ref_bits)[lab] = ba.get_bitstrings()
        return ch_bits, ref_bits

    def fez_v3_jobs():
        js = [j for j in svc.jobs(limit=25)
              if j.backend() and j.backend().name == "ibm_fez"
              and j.creation_date > CUT]
        return sorted(js, key=lambda j: j.creation_date)

    # ---- reconcile BEFORE submission ---------------------------------------- #
    existing = fez_v3_jobs()
    print(f"[reconcile-pre] {len(existing)} v3 fez jobs after cutoff: "
          f"{[j.job_id() for j in existing]}", flush=True)

    job_objs = {}; job_records = []
    if len(existing) >= 8:
        print("[idempotent] 8+ jobs already exist; mapping by creation order, "
              "no resubmission.", flush=True)
        for iid in range(8):
            _, si, lay, ss, order = build_pubs(instances[iid])
            job = existing[iid]
            job_objs[iid] = (job, si, lay)
            job_records.append({"instance_id": iid, "job_id": job.job_id(),
                                "reused": True, "shuffle_seed": ss,
                                "pub_order": order,
                                "submit_ts_utc": job.creation_date.isoformat()})
    else:
        assert len(existing) == 1 and existing[0].job_id() == EXISTING_JOB0, (
            f"expected exactly the queued instance-0 job {EXISTING_JOB0}, "
            f"got {[j.job_id() for j in existing]}")
        _, si0, lay0, ss0, order0 = build_pubs(instances[0])
        job0 = svc.job(EXISTING_JOB0)
        job_objs[0] = (job0, si0, lay0)
        job_records.append({"instance_id": 0, "job_id": job0.job_id(),
                            "reused": True, "shuffle_seed": ss0, "pub_order": order0,
                            "submit_ts_utc": job0.creation_date.isoformat()})
        sampler = SamplerV2(mode=backend)
        for iid in range(1, 8):
            sp, si, lay, ss, order = build_pubs(instances[iid])
            submit_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            job = sampler.run(sp)
            job_objs[iid] = (job, si, lay)
            job_records.append({"instance_id": iid, "job_id": job.job_id(),
                                "reused": False, "shuffle_seed": ss,
                                "pub_order": order, "submit_ts_utc": submit_ts})
            print(f"[submit] instance {iid} -> job {job.job_id()} at {submit_ts}",
                  flush=True)

    # ---- reconcile AFTER submission ----------------------------------------- #
    after = fez_v3_jobs()
    ids = [j.job_id() for j in after]
    dupes = [k for k, c in Counter(ids).items() if c > 1]
    print(f"[reconcile-post] {len(after)} v3 fez jobs total; dupes={dupes}",
          flush=True)
    assert len(after) == 8 and not dupes, (
        f"reconciliation FAILED: expected exactly 8 unique v3 jobs, got "
        f"{len(after)} ids={ids} dupes={dupes}")

    # ---- record all 8 job ids + timestamps IMMEDIATELY ---------------------- #
    with open(os.path.join(OUT, "job_records.json"), "w", encoding="utf-8") as fh:
        json.dump({"batch_queue": True, "target_backend": "ibm_fez",
                   "master_seed": v3.MASTER_SEED, "n_jobs": 8,
                   "job_records": sorted(job_records, key=lambda r: r["instance_id"]),
                   "recorded_ts_utc": datetime.datetime.now(
                       datetime.timezone.utc).isoformat()}, fh, indent=2)
    print("ALL_8_QUEUED " + json.dumps(
        [(r["instance_id"], r["job_id"]) for r in
         sorted(job_records, key=lambda r: r["instance_id"])]), flush=True)

    # ---- wait for ALL 8, collect, analyse ----------------------------------- #
    try:
        quota_before = svc.usage().get("usage_remaining_seconds")
    except Exception:
        quota_before = None
    per_instance = []; layouts = {}; usages = [None] * 8
    for iid in range(8):
        job, si, lay = job_objs[iid]
        print(f"[wait] instance {iid} job {job.job_id()} status={job.status()}",
              flush=True)
        result = job.result()
        try:
            usages[iid] = job.usage()
        except Exception:
            usages[iid] = None
        ch_bits, ref_bits = collect(result, si)
        a = v3.analyse_instance(instances[iid], ch_bits, ref_bits, lay, v3cfg, v2_est)
        per_instance.append(a); layouts[iid] = lay
        print(f"[done] instance {iid} v3_zdiff={a['v3_max_abs_zdiff']:.2f} "
              f"det={a['v3_detected']} v2honest={a['v2_honest_max_abs_z']:.2f} "
              f"det={a['v2_honest_detected']}", flush=True)

    try:
        quota_after = svc.usage().get("usage_remaining_seconds")
    except Exception:
        quota_after = None
    quota_consumed = (quota_before - quota_after
                      if (quota_before is not None and quota_after is not None)
                      else None)
    total_usage = (sum(u for u in usages if isinstance(u, (int, float)))
                   if any(isinstance(u, (int, float)) for u in usages) else None)

    scores = {
        "v3_differential": v3.score_family(per_instance, "v3_detected"),
        "v2_absolute_frozen": v3.score_family(per_instance, "v2_frozen_detected"),
        "v2_absolute_honest": v3.score_family(per_instance, "v2_honest_detected"),
    }
    for r in job_records:
        r["usage_qpu_s"] = usages[r["instance_id"]]
    passes = [{"floor_corr": None, "per_instance": per_instance, "scores": scores,
               "job_records": sorted(job_records, key=lambda r: r["instance_id"]),
               "layouts": layouts, "total_usage_qpu_s": total_usage}]
    submit_decision = v3.evaluate_contrast(passes, armed=True)

    meta = {
        "armed": True,
        "exec_source": f"IBM hardware {backend.name} (batch-queued)",
        "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "target_backend": "ibm_fez", "master_seed": v3.MASTER_SEED,
        "total_shots": v3.N_INST * 2 * v3.CERT_SETTINGS * v3.CERT_SHOTS,
        "qiskit_version": qiskit.__version__, "aer_version": qiskit_aer.__version__,
        "ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "sealed_key_sha256": manifest["sealed_key_sha256"], "unblind_verified": True,
        "v3_config_sha256": manifest["v3_config_sha256"],
        "hypotheses_sha256": manifest["hypotheses_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "v2_z_gate": v2_est.z_gate, "v2_config_sha256": v2_est.config_sha256(),
        "qpu_estimate": manifest["qpu_seconds_estimate"],
        "quota_remaining_before_s": quota_before,
        "quota_remaining_after_s": quota_after, "quota_consumed_s": quota_consumed,
        "batch_queue_strategy": True, "runtime_seconds": round(time.time() - t0, 1),
    }
    hyp = json.load(open(os.path.join(OUT, "frozen_v3_config.json")))["hypotheses"]
    cal = json.load(open(os.path.join(OUT, "calibration_snapshot.json")))
    report = {"metadata": meta, "manifest": manifest, "v3_config": v3cfg,
              "hypotheses": hyp,
              "answer_key": [inst.answer_key() for inst in instances],
              "passes": passes, "submit_decision": submit_decision,
              "calibration_snapshot": cal}
    with open(os.path.join(OUT, "blind_benchmark_v3_fez.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = v3.build_markdown(meta, v3cfg, passes, submit_decision)
    with open(os.path.join(OUT, "blind_benchmark_v3_fez.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(OUT, "job_records.json"), "w", encoding="utf-8") as fh:
        json.dump({"batch_queue": True, "target_backend": "ibm_fez",
                   "job_records": sorted(job_records, key=lambda r: r["instance_id"]),
                   "total_usage_qpu_s": total_usage,
                   "quota_consumed_s": quota_consumed}, fh, indent=2)

    print("\n" + md, flush=True)
    print("SCORED", flush=True)
    print(f"quota_consumed={quota_consumed} total_job_usage={total_usage} "
          f"runtime={meta['runtime_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
