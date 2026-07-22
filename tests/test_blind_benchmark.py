"""Tests for the C7 Step-2 blind benchmark harness (cyz_m.blind_benchmark,
cyz_m.benchmark_baselines).

Repo convention: unittest-style ``TestCase`` classes.  The pure-Python blind
discipline (sealed key, spec-leak, frozen config, ledger arithmetic) is tested
without qiskit; the simulator dry-run tests are guarded by ``skipUnless`` so the
suite still runs where qiskit-aer is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

from cyz_m.blind_benchmark import (
    ChallengeGenerator, ChallengeInstance, FrozenEstimator, CircuitFamily,
    ResourceLedger, canonical_json, sha256_of_obj, norm_ppf,
    CORRELATOR_NULL, OFFNULL_PAIRS, SLOT_INTER, PAULI_PAIRS, PREP_ORDER,
)

try:
    import qiskit  # noqa: F401
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except Exception:                          # pragma: no cover
    HAS_QISKIT = False

_SIM_SEED = 20260721


# --------------------------------------------------------------------------- #
# small deterministic Aer runner for the guarded tests                          #
# --------------------------------------------------------------------------- #
def _noiseless_runner():
    sim = AerSimulator()

    def run(circuits, shots):
        res = sim.run(circuits, shots=shots, seed_simulator=_SIM_SEED).result()
        return [res.get_counts(i) for i in range(len(circuits))]
    return run


# --------------------------------------------------------------------------- #
# blind discipline (no qiskit)                                                 #
# --------------------------------------------------------------------------- #
class SealedKeyTests(unittest.TestCase):
    def test_sealed_key_sha256_roundtrip(self):
        gen = ChallengeGenerator(master_seed=123, n_inst=10, delta_range=(0.2, 0.4))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sealed_key.json")
            written_sha = gen.write_sealed_key(path)
            manifest = gen.public_manifest()
            # the manifest records the same sha as writing produced
            self.assertEqual(written_sha, manifest["sealed_key_sha256"])
            # re-opening and recomputing verifies
            ok, payload = ChallengeGenerator.verify_sealed_key(
                path, manifest["sealed_key_sha256"])
            self.assertTrue(ok)
            self.assertEqual(len(payload["answer_key"]), 10)

    def test_tamper_breaks_sha256(self):
        gen = ChallengeGenerator(master_seed=7, n_inst=5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sealed_key.json")
            sha = gen.write_sealed_key(path)
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["answer_key"][0]["delta"] = 0.999   # tamper
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(canonical_json(payload))
            ok, _ = ChallengeGenerator.verify_sealed_key(path, sha)
            self.assertFalse(ok)

    def test_sha_is_deterministic_for_seed(self):
        a = ChallengeGenerator(master_seed=42, n_inst=20).sealed_key_sha256()
        b = ChallengeGenerator(master_seed=42, n_inst=20).sealed_key_sha256()
        self.assertEqual(a, b)


class SpecLeakTests(unittest.TestCase):
    def test_public_spec_hides_hidden_fields(self):
        gen = ChallengeGenerator(master_seed=99, n_inst=20)
        manifest = gen.public_manifest()
        forbidden = {"dark", "axis", "delta", "slot"}
        for inst in manifest["instances"]:
            self.assertEqual(set(inst.keys()) & forbidden, set(),
                             f"public spec leaked hidden fields: {inst}")
        # the serialized manifest must not contain the words as keys either
        text = json.dumps(manifest)
        # instances carry only id/n_slots/families
        for inst in manifest["instances"]:
            self.assertEqual(set(inst.keys()), {"instance_id", "n_slots", "families"})

    def test_manifest_has_sealed_hash_but_no_answers(self):
        gen = ChallengeGenerator(master_seed=1, n_inst=8)
        manifest = gen.public_manifest()
        self.assertIn("sealed_key_sha256", manifest)
        self.assertNotIn("answer_key", manifest)

    def test_instance_public_spec_matches(self):
        inst = ChallengeInstance(3, 2, dark=True, axis=1.1, delta=0.3, slot=SLOT_INTER)
        spec = inst.public_spec()
        self.assertNotIn("dark", spec)
        self.assertNotIn("axis", spec)
        self.assertEqual(spec["instance_id"], 3)


class FrozenConfigTests(unittest.TestCase):
    def test_config_hashes_and_is_stable(self):
        e1 = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        e2 = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        self.assertEqual(e1.config_sha256(), e2.config_sha256())

    def test_thresholds_from_step1_artifact(self):
        e = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        c = e.config
        # z_gate is the Bonferroni gate over M_offnull x n_inst
        expected = norm_ppf(1.0 - 0.01 / (2.0 * len(OFFNULL_PAIRS) * 20))
        self.assertAlmostEqual(e.z_gate, expected, places=6)
        self.assertGreaterEqual(e.n_shots, 2048)
        # if the artifact is present it must be the source and hashed
        artifact = c["step1_artifact_sha256"]
        if c["step1_source"] == "artifact":
            self.assertIsNotNone(artifact)
            self.assertEqual(len(artifact), 64)

    def test_offnull_family_excludes_xx(self):
        self.assertEqual(len(OFFNULL_PAIRS), 8)
        self.assertNotIn("XX", OFFNULL_PAIRS)
        self.assertEqual(CORRELATOR_NULL["XX"], 1.0)

    def test_decision_rule_threshold(self):
        e = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        # a strong YX signal must be detected; an all-zero vector must not
        strong = {p: 0.0 for p in OFFNULL_PAIRS}
        strong["YX"] = 0.3
        self.assertTrue(e.decide(strong)["detected"])
        self.assertFalse(e.decide({p: 0.0 for p in OFFNULL_PAIRS})["detected"])


class LedgerTests(unittest.TestCase):
    def test_ledger_arithmetic(self):
        led = ResourceLedger("m")
        led.add("depth2", 9, 2048)
        led.add("atomic", 36, 2048)
        self.assertEqual(led.settings, 45)
        self.assertEqual(led.shots, 45 * 2048)
        self.assertEqual(led.by_family, {"depth2": 9, "atomic": 36})


# --------------------------------------------------------------------------- #
# circuit families + simulator dry run (qiskit)                                #
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_QISKIT, "qiskit / qiskit-aer not installed")
class CircuitCountTests(unittest.TestCase):
    def setUp(self):
        self.inst = ChallengeInstance(0, 2, dark=True, axis=0.0, delta=0.3, slot=SLOT_INTER)
        self.cf = CircuitFamily(self.inst)

    def test_family_circuit_counts(self):
        self.assertEqual(len(self.cf.depth2_correlators()), 9)
        self.assertEqual(len(self.cf.atomic_testers()), 36)     # 6 preps x 2 slots x 3 Paulis
        self.assertEqual(len(self.cf.heldout_depth3()), 9)

    def test_three_slot_is_stubbed(self):
        inst3 = ChallengeInstance(0, 3, dark=False, axis=0.0, delta=0.0, slot=None)
        with self.assertRaises(NotImplementedError):
            CircuitFamily(inst3)

    def test_certificate_and_baseline_ledgers_match_circuit_counts(self):
        from cyz_m.blind_benchmark import pair_correlator_from_counts
        from cyz_m.benchmark_baselines import (
            marginal_qpt_baseline, restricted_ptt_baseline, full_ptt_baseline,
            structured_lowmem_baseline,
        )
        run = _noiseless_runner()
        est = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)

        # certificate ledger == depth2 circuit count
        d2 = self.cf.depth2_correlators()
        led = ResourceLedger("certificate")
        led.add("depth2", len(d2), est.n_shots)
        self.assertEqual(led.settings, 9)
        self.assertEqual(led.shots, 9 * est.n_shots)

        _, ml = marginal_qpt_baseline(self.inst, est, run)
        self.assertEqual(ml.settings, 36)
        _, rl = restricted_ptt_baseline(self.inst, est, run)
        self.assertEqual(rl.settings, len(PREP_ORDER) ** 2 * len(PAULI_PAIRS))  # 324
        _, fl = full_ptt_baseline(self.inst, est, run)
        self.assertEqual(fl.settings, 144)
        self.assertEqual(fl.extra["operation_basis_size"], 256)  # d^{4k}
        _, sl = structured_lowmem_baseline(self.inst, est, run)
        self.assertEqual(sl.settings, 9)


@unittest.skipUnless(HAS_QISKIT, "qiskit / qiskit-aer not installed")
class NoiselessDryRunTests(unittest.TestCase):
    """The core claim: noiseless certificate detects every dark, zero false
    positives; the marginal-QPT baseline detects none (Theorem 1)."""

    def _instances(self):
        # controlled mix spanning the x-y axis, plus nulls
        darks = [
            ChallengeInstance(0, 2, True, 0.0, 0.25, SLOT_INTER),           # x axis
            ChallengeInstance(1, 2, True, math.pi / 4, 0.25, SLOT_INTER),   # 45 deg
            ChallengeInstance(2, 2, True, math.pi / 2, 0.25, SLOT_INTER),   # y axis
            ChallengeInstance(3, 2, True, 3 * math.pi / 4, 0.30, SLOT_INTER),
        ]
        nulls = [
            ChallengeInstance(4, 2, False, 0.0, 0.0, None),
            ChallengeInstance(5, 2, False, 0.0, 0.0, None),
            ChallengeInstance(6, 2, False, 0.0, 0.0, None),
        ]
        return darks, nulls

    def test_certificate_detects_all_darks_zero_fp(self):
        from cyz_m.blind_benchmark import pair_correlator_from_counts
        run = _noiseless_runner()
        est = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        darks, nulls = self._instances()

        def cert(inst):
            sc = CircuitFamily(inst).depth2_correlators()
            counts = run([qc for _, qc in sc], est.n_shots)
            corr = {s.label: pair_correlator_from_counts(c)[0]
                    for (s, _), c in zip(sc, counts)}
            return est.decide(corr)

        for inst in darks:
            self.assertTrue(cert(inst)["detected"],
                            f"certificate missed dark axis={inst.axis:.2f}")
        for inst in nulls:
            self.assertFalse(cert(inst)["detected"],
                             "certificate false-positive on a null instance")

    def test_marginal_qpt_detects_no_darks(self):
        from cyz_m.benchmark_baselines import marginal_qpt_baseline
        run = _noiseless_runner()
        est = FrozenEstimator(n_inst_declared=20, alpha_exp=0.01)
        darks, nulls = self._instances()
        for inst in darks + nulls:
            verdict, _ = marginal_qpt_baseline(inst, est, run)
            self.assertFalse(verdict["detected"],
                             "marginal QPT should be blind to the dark link "
                             "(Theorem 1)")

    def test_atomic_family_is_dark_invariant(self):
        from cyz_m.blind_benchmark import single_expectation_from_counts
        run = _noiseless_runner()
        est = FrozenEstimator()
        dark = ChallengeInstance(0, 2, True, 1.0, 0.35, SLOT_INTER)
        null = ChallengeInstance(0, 2, False, 0.0, 0.0, None)
        sc_d = CircuitFamily(dark).atomic_testers()
        sc_0 = CircuitFamily(null).atomic_testers()
        cd = run([qc for _, qc in sc_d], 4096)
        c0 = run([qc for _, qc in sc_0], 4096)
        worst = 0.0
        for (s, _), a, b in zip(sc_d, cd, c0):
            va, _ = single_expectation_from_counts(a)
            vb, _ = single_expectation_from_counts(b)
            worst = max(worst, abs(va - vb))
        # atomic circuits are structurally identical for dark/null -> only shot
        # noise separates them
        self.assertLess(worst, 0.1, f"atomic family not dark-invariant ({worst})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
