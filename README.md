# cyz_m — Order-Aware Certification of Multitime Response Completeness

Code, proofs, and hardware provenance for the manuscript:

> **Finite Atomic Experiments Cannot Certify Their Own Operational Dimension: An Order-Aware Flat-Extension Certificate**
> (Letter: `paper/prl/main.tex`, 4 pp; Supplemental Material: `paper/prl/supplement.tex`)

## What this repository establishes

1. **Atomic non-self-certification (exact counterexample).** A two-qubit collision model `CZ–R_x(δ)–CZ` whose complete single-time marginal-channel IC tester statistics are exactly independent of a smooth, non-gauge memory rotation that one retained two-time correlator detects at first order (`examples/run_c7_prototype.py`).
2. **Delayed-chain necessity.** Finite Hankel plateaus never certify all-depth closure without order provenance (`examples/run_c7_jet_checks.py`).
3. **Saturated first-jet flat-extension certificate.** For a declared response order `n_J`, a rank-saturated two-sided flat Hankel core determines every registered response at arbitrary depth, with mandatory failure semantics (`src/cyz_m/jet_hankel.py`; proofs in the Supplemental Material and the archived theorem record, issue #5).
4. **Failure-aware finite-error layer and resource separation** (`src/cyz_m/spacetime_witness.py`; Theorems P and R).
5. **Preregistered sealed blind benchmarks on IBM hardware** — `ibm_marrakesh` and `ibm_fez`, with sealed challenge keys (SHA-256 published pre-submission), frozen estimators, unblinding verification, and a preregistered null-floor localization experiment (`research/results/blind_benchmark_hardware*`, `research/results/fez_null_confirmation/`).

## Reproduction

```bash
pip install numpy               # core numerics (qiskit only needed for hardware/benchmark code)
PYTHONPATH=src python -m unittest discover -s tests          # full test suite
PYTHONPATH=src python examples/run_c7_prototype.py           # counterexample certificate numbers
PYTHONPATH=src python examples/run_c7_jet_checks.py          # jet / flat-extension checks
PYTHONPATH=src python experiments/run_step1_separator_power.py
PYTHONPATH=src python experiments/run_blind_benchmark_dryrun.py
```

Hardware artifacts (job IDs, calibration snapshots, sealed manifests, per-run reports) are under `research/results/`. The external adversarial-review record (rounds R1–R9: novelty adjudication, theorem proofs, referee simulations, independent review) lives in the development record.
