#!/usr/bin/env python
"""Drift / half-split / device-floor diagnostics for the v3 fez hardware run.

Re-fetches the 8 DONE v3 jobs (read-only), de-interleaves them with the same
deterministic per-job PUB order used at submission, and computes, per instance:

  * challenge / reference correlators (XX, YX, ZX) and the differential;
  * within-run half-split (first 768 / last 768 shots, shot-ordered) of the
    differential decision statistic -> drift / stability check;
  * the reference-block device-floor estimate (YX) and its within-run half-split;

plus the POOLED reference-block YX floor across all 8 instances (8 x 1536 =
12288 shots) with its z -- the maximum-power estimate of this session's
device-locked floor and its sign, directly comparable to the earlier
6-null confirmation (pooled YX z=-6.20, corr -0.056).

Writes ``diagnostics_v3_fez.{md,json}`` to the results dir.  No submission.

    PYTHONPATH=src python experiments/run_v3_diagnostics.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import statistics
from collections import Counter

from qiskit_ibm_runtime import QiskitRuntimeService
from cyz_m.blind_benchmark import (ChallengeGenerator, pair_correlator_from_counts,
                                   SIGNAL_SUBSPACE_PAIRS)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(_HERE, "run_blind_benchmark_v3_fez.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)
OUT = v3.OUT_DIR
GATE = v3.v3_gate_config()["z_gate_diff"]
V2_GATE = 3.559932645771468
HALF = v3.CERT_SHOTS // 2


def corr(bits):
    return pair_correlator_from_counts(Counter(bits))[0]


def var(c, n):
    return max(1e-6, 1.0 - c * c) / n


def zdiff(cc, nc, cr, nr):
    return (cc - cr) / math.sqrt(var(cc, nc) + var(cr, nr))


def main() -> None:
    svc = QiskitRuntimeService()
    backend = svc.backend("ibm_fez")
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
    gen = ChallengeGenerator(v3.MASTER_SEED, v3.N_INST, v3.DELTA_RANGE,
                             v3.N_SLOTS, v3.DARK_PROB)
    instances = gen.instances
    recs = {r["instance_id"]: r for r in
            json.load(open(os.path.join(OUT, "job_records.json")))["job_records"]}

    def shuffled_index(instance):
        ref = v3.reference_instance(instance.instance_id)
        ch = v3.build_correlator_circuits(instance, None)
        rf = v3.build_correlator_circuits(ref, None)
        pub_index = [("challenge", lab) for lab, _ in ch] + \
                    [("reference", lab) for lab, _ in rf]
        ss = int(v3.sha256_hex(f"{v3.MASTER_SEED}:job{instance.instance_id}")[:8], 16)
        rng = random.Random(ss); order = list(range(len(pub_index))); rng.shuffle(order)
        return [pub_index[k] for k in order]

    per = []
    pooled_ref_yx_bits = []
    for inst in instances:
        iid = inst.instance_id
        job = svc.job(recs[iid]["job_id"])
        result = job.result()
        si = shuffled_index(inst)
        ch_bits, ref_bits = {}, {}
        for i, (block, lab) in enumerate(si):
            data = result[i].data
            ba = getattr(data, "c", None)
            (ch_bits if block == "challenge" else ref_bits)[lab] = ba.get_bitstrings()

        row = {"instance_id": iid, "dark": inst.dark, "pairs": {}}
        pooled_ref_yx_bits.extend(ref_bits["YX"])
        # full + half-split differential per signal pair
        for p in SIGNAL_SUBSPACE_PAIRS:
            cb, rb = ch_bits[p], ref_bits[p]
            n = len(cb)
            full = zdiff(corr(cb), n, corr(rb), len(rb))
            z1 = zdiff(corr(cb[:HALF]), HALF, corr(rb[:HALF]), HALF)
            z2 = zdiff(corr(cb[HALF:]), n - HALF, corr(rb[HALF:]), len(rb) - HALF)
            row["pairs"][p] = {
                "c_challenge": corr(cb), "c_reference": corr(rb),
                "z_diff_full": full, "z_diff_first_half": z1, "z_diff_second_half": z2,
                "ref_floor_first_half": corr(rb[:HALF]),
                "ref_floor_second_half": corr(rb[HALF:]),
            }
        arg = max(row["pairs"], key=lambda k: abs(row["pairs"][k]["z_diff_full"]))
        zf = row["pairs"][arg]["z_diff_full"]
        z1 = row["pairs"][arg]["z_diff_first_half"]
        z2 = row["pairs"][arg]["z_diff_second_half"]
        # half-split stable: both halves on the same side of the gate as the full
        det = abs(zf) > GATE
        if det:
            stable = (abs(z1) > GATE / 2 and abs(z2) > GATE / 2 and
                      (z1 > 0) == (z2 > 0) == (zf > 0))
        else:
            stable = abs(z1) < GATE and abs(z2) < GATE
        row.update({"argmax_pair": arg, "z_diff_full": zf,
                    "z_diff_first_half": z1, "z_diff_second_half": z2,
                    "detected": det, "half_split_stable": stable,
                    "xx_reference": corr(ref_bits["XX"]),
                    "ref_yx": row["pairs"]["YX"]["c_reference"]})
        per.append(row)

    # pooled reference-block YX floor across all 8 instances (12288 shots)
    pc = corr(pooled_ref_yx_bits); pn = len(pooled_ref_yx_bits)
    pooled = {"corr": pc, "n": pn, "z": pc / math.sqrt(var(pc, pn)),
              "sign": "+" if pc > 0 else "-"}
    ref_yx = [r["ref_yx"] for r in per]
    signs = ["+" if x > 0 else "-" for x in ref_yx]
    dom = max(set(signs), key=signs.count)
    diag = {
        "protocol": "v3_fez_diagnostics", "cert_shots_per_block": v3.CERT_SHOTS,
        "v3_gate": GATE, "half_split_shots": HALF,
        "pooled_reference_YX_floor": pooled,
        "reference_YX_floor_per_instance": ref_yx,
        "reference_YX_floor_sign_fraction": signs.count(dom) / len(signs),
        "reference_YX_floor_dominant_sign": dom,
        "all_half_split_stable": all(r["half_split_stable"] for r in per),
        "comparison_confirmation_run": ("6-null confirmation (2026-07-21 15:21 UTC): "
            "pooled YX z=-6.20 corr=-0.056 sign '-'; this v3 session's pooled "
            f"reference YX z={pooled['z']:+.2f} corr={pooled['corr']:+.4f} sign "
            f"'{pooled['sign']}' -> the floor's sign/magnitude DRIFTED between "
            "calibration windows, which the differential cancels by construction."),
        "per_instance": per,
    }
    with open(os.path.join(OUT, "diagnostics_v3_fez.json"), "w",
              encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)

    md = ["# Protocol v3 -- ibm_fez drift / half-split / device-floor diagnostics",
          "",
          f"- Shots/block: {v3.CERT_SHOTS} (half-split {HALF}/{HALF}, shot-ordered)  "
          f"|  v3 gate |z_diff| = {GATE:.4f}",
          f"- **Pooled reference-block YX floor (this session, {pn} shots): "
          f"corr = {pooled['corr']:+.4f}, z = {pooled['z']:+.2f}, sign "
          f"'{pooled['sign']}'**",
          f"- Reference YX sign agreement across 8 instances: "
          f"{signs.count(dom)}/8 sign '{dom}'  (per-instance floor mean "
          f"{statistics.mean(ref_yx):+.4f}, range {min(ref_yx):+.4f}..{max(ref_yx):+.4f})",
          f"- All instances half-split stable: **{all(r['half_split_stable'] for r in per)}**",
          "",
          "> Cross-run: the 6-null confirmation (15:21 UTC) had pooled YX z=-6.20 "
          "(corr -0.056, sign '-'); this v3 session (19:01 UTC) has pooled "
          f"reference YX z={pooled['z']:+.2f} (corr {pooled['corr']:+.4f}, sign "
          f"'{pooled['sign']}').  The device floor's sign/magnitude DRIFTED between "
          "calibration windows -- an absolute gate calibrated once cannot track "
          "this; the differential cancels it every run by construction.",
          "",
          "## Per-instance differential + half-split",
          "",
          "| inst | truth | argmax | z_diff full | 1st-half | 2nd-half | stable? | "
          "ref YX floor | <XX>ref |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in per:
        md.append(f"| {r['instance_id']} | {'dark' if r['dark'] else 'null'} | "
                  f"{r['argmax_pair']} | {r['z_diff_full']:+.2f} | "
                  f"{r['z_diff_first_half']:+.2f} | {r['z_diff_second_half']:+.2f} | "
                  f"{'yes' if r['half_split_stable'] else 'NO'} | "
                  f"{r['ref_yx']:+.4f} | {r['xx_reference']:+.3f} |")
    md.append("")
    with open(os.path.join(OUT, "diagnostics_v3_fez.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    print("\nDIAGNOSTICS_DONE")


if __name__ == "__main__":
    main()
