"""Tests for the C7 blind-benchmark IBM hardware runner.

Covers the four required properties:
    * run-plan determinism (deterministic-from-master-seed plan + interleave),
    * the ``--arm`` refusal path (no submission without ``--arm``),
    * PUB construction counts (per-arm circuit + shot counts), and
    * frozen estimator-config hash stability.

Repo convention: unittest-style ``TestCase``.  qiskit-dependent tests are guarded
by ``skipUnless`` so the plan/estimator/CLI-gate tests still run without qiskit.
The refusal test proves -- via a spy sampler and a ``fetch_real_backend`` that
raises -- that the no-``--arm`` path never constructs a real-backend submission.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import unittest

# make the experiment script importable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_REPO, "experiments")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from cyz_m.hardware_runner import (
    HardwareRunPlan, HW_MASTER_SEED, MARGINAL_SHOTS, QPU_BUDGET_S,
    DEVICE_READOUT_ASSIGNMENT_ERROR, READOUT_CORRELATOR_QUBITS,
    ESTIMATOR_PLATFORM, DELTA_RANGE, ALPHA_EXP, N_INST_HW,
    calibration_snapshot, build_job_pubs, counts_from_pub_result,
)
from cyz_m.blind_benchmark import (
    FrozenEstimator, SIGNAL_SUBSPACE_PAIRS, OFFNULL_PAIRS, norm_ppf,
)

try:
    import qiskit  # noqa: F401
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    HAS_QISKIT = True
except Exception:                              # pragma: no cover
    HAS_QISKIT = False


# --------------------------------------------------------------------------- #
# run-plan determinism (no qiskit needed)                                       #
# --------------------------------------------------------------------------- #
class PlanDeterminismTests(unittest.TestCase):
    def test_plan_sha_is_deterministic_for_seed(self):
        a = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        b = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        self.assertEqual(a.plan_sha256(), b.plan_sha256())
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_plan_sha_changes_with_seed(self):
        a = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        b = HardwareRunPlan(master_seed=HW_MASTER_SEED + 1)
        self.assertNotEqual(a.plan_sha256(), b.plan_sha256())

    def test_totals_and_job_count(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        t = p.totals()
        # 8 instances, one job each
        self.assertEqual(t["job_count"], 8)
        # certificate: 8 inst x 9 settings x 2048
        self.assertEqual(t["by_arm"]["certificate"]["pubs"], 72)
        self.assertEqual(t["by_arm"]["certificate"]["shots"], 72 * 2048)
        # marginal: 8 inst x 36 settings x 512
        self.assertEqual(t["by_arm"]["marginal_qpt"]["pubs"], 288)
        self.assertEqual(t["by_arm"]["marginal_qpt"]["shots"], 288 * 512)
        # heldout: 2 inst x 9 settings x 2048
        self.assertEqual(t["by_arm"]["heldout_depth3"]["pubs"], 18)
        self.assertEqual(t["by_arm"]["heldout_depth3"]["shots"], 18 * 2048)
        self.assertEqual(t["total_shots"], 72 * 2048 + 288 * 512 + 18 * 2048)
        self.assertEqual(t["total_shots"], 331776)

    def test_qpu_estimate_under_budget(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        q = p.qpu_seconds_estimate()
        self.assertTrue(q["under_budget"])
        self.assertLess(q["high_s"], QPU_BUDGET_S)
        self.assertLess(q["low_s"], q["high_s"])

    def test_marginal_ci_halfwidth(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        ci = p.marginal_ci()
        self.assertEqual(ci["marginal_shots"], MARGINAL_SHOTS)
        self.assertAlmostEqual(ci["se_null"], 1.0 / math.sqrt(512), places=9)
        self.assertAlmostEqual(ci["ci_halfwidth_95"],
                               1.959963984540054 / math.sqrt(512), places=6)

    def test_pub_order_is_deterministic_permutation(self):
        a = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        b = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        for ja, jb in zip(a.jobs, b.jobs):
            self.assertEqual(ja["pub_order"], jb["pub_order"])
            self.assertEqual(sorted(ja["pub_order"]), list(range(ja["n_pubs"])))

    def test_heldout_only_first_two_instances(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        heldout_arm = next(a for a in p.arms if a.name == "heldout_depth3")
        self.assertEqual(heldout_arm.instance_ids, (0, 1))


# --------------------------------------------------------------------------- #
# frozen estimator-config hash stability                                        #
# --------------------------------------------------------------------------- #
class EstimatorConfigHashTests(unittest.TestCase):
    def test_config_hash_stable_across_plans(self):
        a = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        b = HardwareRunPlan(master_seed=HW_MASTER_SEED + 5)  # seed changes challenges only
        self.assertEqual(a.estimator.config_sha256(),
                         b.estimator.config_sha256())

    def test_cert_shots_match_frozen_n_shots(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        self.assertEqual(p.cert_shots, p.estimator.n_shots)
        self.assertEqual(p.cert_shots, 2048)

    def test_estimator_config_matches_independent_freeze(self):
        p = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        # the plan freezes the v2 hardened certificate: signal-subspace decision
        # + device-aware gate; an independent freeze with the same declared
        # params must reproduce the identical config hash.
        indep = FrozenEstimator(
            target_platform=ESTIMATOR_PLATFORM,
            min_detectable_delta=DELTA_RANGE[0],
            n_inst_declared=N_INST_HW, alpha_exp=ALPHA_EXP,
            signal_subspace=True,
            readout_assignment_error=DEVICE_READOUT_ASSIGNMENT_ERROR,
            readout_correlator_qubits=READOUT_CORRELATOR_QUBITS)
        self.assertEqual(p.estimator.config_sha256(), indep.config_sha256())


# --------------------------------------------------------------------------- #
# v2 hardening: signal-subspace decision                                        #
# --------------------------------------------------------------------------- #
class SignalSubspaceTests(unittest.TestCase):
    """The plan's certificate restricts its decision to the {YX, ZX} subspace."""

    def setUp(self):
        self.plan = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        self.est = self.plan.estimator

    def test_decision_family_is_signal_subspace(self):
        self.assertEqual(list(self.est.decision_pairs), ["YX", "ZX"])
        self.assertEqual(SIGNAL_SUBSPACE_PAIRS, ["YX", "ZX"])
        c = self.est.config
        self.assertEqual(c["config_version"], 2)
        self.assertEqual(c["decision_family"], "depth2_signal_subspace_YX_ZX")
        self.assertEqual(c["signal_subspace_pairs"], ["YX", "ZX"])
        self.assertEqual(c["m_signal"], 2)
        self.assertEqual(c["bonferroni_tests"], 2 * self.est.n_inst_declared)
        self.assertIn("changelog", c)
        self.assertIn("rehearsal", c["changelog"].lower())

    def test_offsignal_correlator_never_detects(self):
        """The rehearsal FP mechanism: a large off-signal (e.g. XY) correlator
        must NOT trip the hardened certificate, at any magnitude."""
        for bad in ("XY", "XZ", "YY", "YZ", "ZY", "ZZ"):
            self.assertNotIn(bad, self.est.decision_pairs)
            vec = {p: 0.0 for p in OFFNULL_PAIRS}
            vec[bad] = 0.30                         # z ~ 14, would trip v1
            v = self.est.decide(vec)
            self.assertFalse(v["detected"],
                             f"off-signal {bad} tripped the hardened certificate")

    def test_signal_correlator_still_detects(self):
        for good in ("YX", "ZX"):
            vec = {p: 0.0 for p in OFFNULL_PAIRS}
            vec[good] = 0.12                         # a delta0~0.2 first-order signal
            v = self.est.decide(vec)
            self.assertTrue(v["detected"], f"signal {good} missed")
            self.assertEqual(v["argmax_pair"], good)

    def test_legacy_estimator_unchanged(self):
        """v1 defaults stay byte-identical (dry-run + test_blind_benchmark)."""
        legacy = FrozenEstimator(target_platform=ESTIMATOR_PLATFORM,
                                 min_detectable_delta=DELTA_RANGE[0],
                                 n_inst_declared=N_INST_HW, alpha_exp=ALPHA_EXP)
        self.assertEqual(list(legacy.decision_pairs), list(OFFNULL_PAIRS))
        self.assertNotIn("config_version", legacy.config)
        self.assertNotEqual(legacy.config_sha256(), self.est.config_sha256())


# --------------------------------------------------------------------------- #
# v2 hardening: device-aware gate recalibration                                 #
# --------------------------------------------------------------------------- #
class GateRecalibrationTests(unittest.TestCase):
    """z_gate is recalibrated from the device readout assignment error."""

    def setUp(self):
        self.est = HardwareRunPlan(master_seed=HW_MASTER_SEED).estimator
        self.c = self.est.config

    def test_shot_gate_is_two_pair_bonferroni(self):
        expected = norm_ppf(1.0 - ALPHA_EXP / (2.0 * 2 * N_INST_HW))
        self.assertAlmostEqual(self.c["z_gate_shot"], expected, places=9)

    def test_readout_inflation_factor(self):
        eps = DEVICE_READOUT_ASSIGNMENT_ERROR
        k = READOUT_CORRELATOR_QUBITS
        lam = (1.0 - 2.0 * eps) ** k
        self.assertAlmostEqual(self.c["readout_assignment_error"], eps, places=12)
        self.assertAlmostEqual(self.c["readout_attenuation_lambda"], lam, places=12)
        # device-aware gate = shot gate inflated by 1/lambda
        self.assertAlmostEqual(self.est.z_gate, self.c["z_gate_shot"] / lam,
                               places=9)

    def test_device_gate_between_shot_gate_and_legacy(self):
        legacy = norm_ppf(1.0 - ALPHA_EXP / (2.0 * len(OFFNULL_PAIRS) * N_INST_HW))
        # readout inflation raises the gate above the pure 2-pair shot gate ...
        self.assertGreater(self.est.z_gate, self.c["z_gate_shot"])
        # ... but the 8->2 family reduction keeps it below the v1 8-pair gate.
        self.assertLess(self.est.z_gate, legacy)

    def test_power_target_preserved_shots_floor(self):
        # the Step-1 matched-power budget still floors at 2048 shots.
        self.assertEqual(self.est.n_shots, 2048)
        self.assertGreater(self.c["power_factor"], 1.0)


# --------------------------------------------------------------------------- #
# --arm refusal path                                                            #
# --------------------------------------------------------------------------- #
class ArmRefusalTests(unittest.TestCase):
    def _exp(self):
        return importlib.import_module("run_blind_benchmark_hardware")

    def test_default_refuses_hardware(self):
        exp = self._exp()
        args = exp.build_arg_parser().parse_args([])
        self.assertFalse(args.arm)
        self.assertFalse(exp.will_submit_to_hardware(args.arm))

    def test_arm_flag_enables_submission_gate(self):
        exp = self._exp()
        args = exp.build_arg_parser().parse_args(["--arm"])
        self.assertTrue(args.arm)
        self.assertTrue(exp.will_submit_to_hardware(args.arm))

    def test_output_routing(self):
        exp = self._exp()
        out_r, pre_r = exp.resolve_output(False)
        out_a, pre_a = exp.resolve_output(True)
        self.assertTrue(out_r.endswith("blind_benchmark_hardware_rehearsal"))
        self.assertEqual(pre_r, "rehearsal_")
        self.assertTrue(out_a.endswith("blind_benchmark_hardware"))
        self.assertEqual(pre_a, "")

    @unittest.skipUnless(HAS_QISKIT, "qiskit / qiskit-aer not installed")
    def test_no_arm_never_constructs_real_backend_sampler(self):
        """End-to-end proof: with no --arm and a real-backend fetch that raises,
        main() still completes locally and never builds a real-backend sampler."""
        exp = self._exp()
        from qiskit_ibm_runtime import SamplerV2 as RealSamplerV2

        seen_modes = []

        def spy_make_sampler(mode):
            seen_modes.append(type(mode).__name__)
            # a real IBM backend would not be an AerSimulator
            assert isinstance(mode, AerSimulator), \
                f"refusal breached: sampler bound to {type(mode).__name__}"
            return RealSamplerV2(mode=mode)

        def _raise_fetch(_name):
            raise AssertionError("no --arm: fetch_real_backend must not gate "
                                 "submission (fallback to generic Heron)")

        orig_sampler = exp.make_sampler
        orig_fetch = exp.fetch_real_backend
        try:
            exp.make_sampler = spy_make_sampler
            exp.fetch_real_backend = _raise_fetch
            # tiny run: 1 instance keeps it fast; still exercises all arms
            exp.main(["--n-inst", "1"])
        finally:
            exp.make_sampler = orig_sampler
            exp.fetch_real_backend = orig_fetch

        # rehearsal artifacts exist; every sampler mode was a local AerSimulator
        self.assertTrue(seen_modes)
        self.assertTrue(all(m == "AerSimulator" for m in seen_modes), seen_modes)
        md = os.path.join(exp.REHEARSAL_DIR,
                          "rehearsal_blind_benchmark_hardware.md")
        self.assertTrue(os.path.isfile(md))


# --------------------------------------------------------------------------- #
# PUB construction counts                                                       #
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_QISKIT, "qiskit / qiskit-aer not installed")
class PubConstructionTests(unittest.TestCase):
    def setUp(self):
        self.plan = HardwareRunPlan(master_seed=HW_MASTER_SEED)
        self.pm = generate_preset_pass_manager(optimization_level=1,
                                               backend=AerSimulator())

    def _job(self, instance_id):
        return next(j for j in self.plan.jobs if j["instance_id"] == instance_id)

    def test_heldout_instance_pub_counts(self):
        # instance 0 gets certificate(9) + marginal(36) + heldout(9) = 54
        pubs, meta = build_job_pubs(self.plan, self._job(0), self.pm)
        self.assertEqual(len(pubs), 54)
        self.assertEqual(len(meta), 54)
        arms = [m.arm for m in meta]
        self.assertEqual(arms.count("certificate"), 9)
        self.assertEqual(arms.count("marginal_qpt"), 36)
        self.assertEqual(arms.count("heldout_depth3"), 9)

    def test_non_heldout_instance_pub_counts(self):
        # instance 2 gets certificate(9) + marginal(36) = 45 (no heldout)
        pubs, meta = build_job_pubs(self.plan, self._job(2), self.pm)
        self.assertEqual(len(pubs), 45)
        arms = [m.arm for m in meta]
        self.assertEqual(arms.count("heldout_depth3"), 0)
        self.assertEqual(arms.count("certificate"), 9)
        self.assertEqual(arms.count("marginal_qpt"), 36)

    def test_per_pub_shots(self):
        pubs, meta = build_job_pubs(self.plan, self._job(0), self.pm)
        for (_, _, shots), m in zip(pubs, meta):
            self.assertEqual(shots, m.shots)
            if m.arm == "marginal_qpt":
                self.assertEqual(shots, 512)
            else:
                self.assertEqual(shots, 2048)

    def test_pub_order_matches_plan_shuffle(self):
        job = self._job(0)
        pubs, meta = build_job_pubs(self.plan, job, self.pm)
        # the arm sequence in meta must match the planned (shuffled) pub refs
        planned_arms = [ref[0] for ref in job["pubs"]]
        self.assertEqual([m.arm for m in meta], planned_arms)


# --------------------------------------------------------------------------- #
# calibration snapshot (light; uses a small fake backend if available)          #
# --------------------------------------------------------------------------- #
class CalibrationSnapshotTests(unittest.TestCase):
    def test_snapshot_from_fake_backend(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        except Exception:                          # pragma: no cover
            self.skipTest("fake_provider unavailable")
        snap = calibration_snapshot(FakeManilaV2())
        self.assertIn("qubit_properties", snap)
        self.assertIn("operation_error_stats", snap)
        self.assertGreater(snap["num_qubits"], 0)
        self.assertEqual(len(snap["qubit_properties"]), snap["num_qubits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
