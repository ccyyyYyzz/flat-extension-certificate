# Blind benchmark PROTOCOL v3 (device-referenced differential) -- REHEARSAL run

- Mode: **REHEARSAL** (AerSimulator.from_backend(ibm_fez) seed=20260729)
- Date: 2026-07-21  |  target: ibm_fez
- Master seed: 20260729  |  qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, aer 0.17.2
- Blind set: 8 instances (dark_prob 0.5), certificate arm only, 8x(9 challenge + 9 reference)x1536 = 221184 shots, 1 job/instance (8 jobs x 18 PUBs)
- sealed_key_sha256: `2b9325a556cfbd2f286c17b15bb1d5c8d5dcc45fad3dad0cdb61fc34274a9395`  |  unblind verified: **True**
- v3 estimator config_sha256: `253ece5e799edf8935bc33e8d552daa0806a70cbea5baf074077e8005e891b9e`  |  hypotheses_sha256: `79de57ab9517357297abc31dad74519de4735380d620c4c68813a0c956634f5e`  |  manifest_sha256: `921a4d4df90c5c75e454c08fc13a00ff6c85a5b48962b2157367de610a420494` (all sealed pre-exec)

## Frozen v3 differential gate

- z_gate_shot (Bonferroni over 2x8 = 16 tests, alpha_exp 0.01): **3.4205**
- fez readout eps (median measure, from fez calibration snapshot): 0.00872802734375  -> lambda_fez = (1-2eps)^2 = 0.965393
- **differential gate z_gate_diff = z_gate_shot/lambda_fez = 3.5431** (z_diff units); equivalently sqrt(2)*z_gate_shot/lambda_fez = 5.0108 in single-block-sigma units (identical test)
- component variance: var_diff_p = (1-(C_ch_p)^2)/N_ch + (1-(C_ref_p)^2)/N_ref  (paper Eq. differential-variance)
- INCONCLUSIVE: reference-drift guard: |<XX>_ref| < 0.5 -> INCONCLUSIVE
- v2-absolute comparison gate (frozen fez v2 classifier): z_gate = 3.5599

## Rehearsal condition -- injected YX floor corr = -0.056 (confirmation-day POOLED floor)

- QPU-s this pass (sum job.usage()): **0**

| classifier | power | FP rate | TP/dark | FP/null | inconclusive |
|---|---|---|---|---|---|
| v3 device-referenced differential | 1.00 | 0.00 | 5/5 | 0/3 | 0 |
| v2 absolute gate (honest N rescore) | 1.00 | 0.00 | 5/5 | 0/3 | 0 |
| v2 absolute gate (frozen v2 estimator) | 1.00 | 0.00 | 5/5 | 0/3 | 0 |

| inst | truth | phys | v3 z_diff(YX,ZX) | max|z_diff| | v3? | v2 |z|(honest) | v2? | <XX>ref |
|---|---|---|---|---|---|---|---|---|
| 0 | null | (0, 1, 2) | (+1.26,-0.94) | 1.26 | - | 0.82 | - | +0.939 |
| 1 | null | (0, 1, 2) | (-0.94,+0.04) | 0.94 | - | 2.15 | - | +0.958 |
| 2 | dark | (0, 1, 2) | (+3.83,+5.87) | 5.87 | DET | 9.95 | DET | +0.944 |
| 3 | dark | (0, 1, 2) | (+1.37,-9.41) | 9.41 | DET | 13.37 | DET | +0.948 |
| 4 | null | (0, 1, 2) | (-0.40,+1.77) | 1.77 | - | 2.35 | - | +0.935 |
| 5 | dark | (0, 1, 2) | (-6.91,+2.54) | 6.91 | DET | 12.24 | DET | +0.958 |
| 6 | dark | (0, 1, 2) | (+1.55,-7.09) | 7.09 | DET | 8.95 | DET | +0.953 |
| 7 | dark | (0, 1, 2) | (+8.84,-2.35) | 8.84 | DET | 9.73 | DET | +0.944 |

## Rehearsal condition -- injected YX floor corr = -0.120 (worst-case STRONG-floor day (v2-breaking; the contrast))

- QPU-s this pass (sum job.usage()): **0**

| classifier | power | FP rate | TP/dark | FP/null | inconclusive |
|---|---|---|---|---|---|
| v3 device-referenced differential | 1.00 | 0.00 | 5/5 | 0/3 | 0 |
| v2 absolute gate (honest N rescore) | 1.00 | 0.67 | 5/5 | 2/3 | 0 |
| v2 absolute gate (frozen v2 estimator) | 1.00 | 1.00 | 5/5 | 3/3 | 0 |

| inst | truth | phys | v3 z_diff(YX,ZX) | max|z_diff| | v3? | v2 |z|(honest) | v2? | <XX>ref |
|---|---|---|---|---|---|---|---|---|
| 0 | null | (0, 1, 2) | (+1.16,-0.97) | 1.16 | - | 3.43 | - | +0.932 |
| 1 | null | (0, 1, 2) | (-0.94,+0.04) | 0.94 | - | 4.78 | DET | +0.953 |
| 2 | dark | (0, 1, 2) | (+3.69,+5.87) | 5.87 | DET | 9.95 | DET | +0.938 |
| 3 | dark | (0, 1, 2) | (+1.52,-9.38) | 9.38 | DET | 13.31 | DET | +0.940 |
| 4 | null | (0, 1, 2) | (-0.25,+1.77) | 1.77 | - | 3.79 | DET | +0.926 |
| 5 | dark | (0, 1, 2) | (-6.66,+2.50) | 6.66 | DET | 14.78 | DET | +0.953 |
| 6 | dark | (0, 1, 2) | (+1.57,-7.02) | 7.02 | DET | 8.84 | DET | +0.947 |
| 7 | dark | (0, 1, 2) | (+8.96,-2.35) | 8.96 | DET | 7.48 | DET | +0.938 |

## Submit decision

- Rehearsal: v3 power=1.00/FP=0.00 in all floor conditions = True; v2-absolute FP>=1 null in strong-floor condition = True. PROCEED to --arm.
- contrast satisfied: **True**

## QPU usage / budget

- preflight estimate: model 19.1-57.2 s, empirical-fit **77.6 s** (budget 80 s, under_budget True)

## One-sentence result (paper's marked slot)

> Under an injected device-locked YX null floor, protocol-v3 differential decision held power 1.00 / false-positive rate 0.00 while the unreferenced absolute gate false-positived on 2/3 nulls, confirming the differential cancels the fixed floor the absolute gate cannot.
