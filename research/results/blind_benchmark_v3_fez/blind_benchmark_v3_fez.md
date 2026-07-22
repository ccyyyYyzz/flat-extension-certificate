# Blind benchmark PROTOCOL v3 (device-referenced differential) -- HARDWARE run

- Mode: **HARDWARE** (IBM hardware ibm_fez (batch-queued))
- Date: 2026-07-21  |  target: ibm_fez
- Master seed: 20260729  |  qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, aer 0.17.2
- Blind set: 8 instances (dark_prob 0.5), certificate arm only, 8x(9 challenge + 9 reference)x1536 = 221184 shots, 1 job/instance (8 jobs x 18 PUBs)
- sealed_key_sha256: `2b9325a556cfbd2f286c17b15bb1d5c8d5dcc45fad3dad0cdb61fc34274a9395`  |  unblind verified: **True**
- v3 estimator config_sha256: `253ece5e799edf8935bc33e8d552daa0806a70cbea5baf074077e8005e891b9e`  |  hypotheses_sha256: `79de57ab9517357297abc31dad74519de4735380d620c4c68813a0c956634f5e`  |  manifest_sha256: `ac9c9b818f3b574765d971a9c74f4ea176609bd71eb17f599467a67845c92d91` (all sealed pre-exec)

## Frozen v3 differential gate

- z_gate_shot (Bonferroni over 2x8 = 16 tests, alpha_exp 0.01): **3.4205**
- fez readout eps (median measure, from fez calibration snapshot): 0.00872802734375  -> lambda_fez = (1-2eps)^2 = 0.965393
- **differential gate z_gate_diff = z_gate_shot/lambda_fez = 3.5431** (z_diff units); equivalently sqrt(2)*z_gate_shot/lambda_fez = 5.0108 in single-block-sigma units (identical test)
- component variance: var_diff_p = (1-(C_ch_p)^2)/N_ch + (1-(C_ref_p)^2)/N_ref  (paper Eq. differential-variance)
- INCONCLUSIVE: reference-drift guard: |<XX>_ref| < 0.5 -> INCONCLUSIVE
- v2-absolute comparison gate (frozen fez v2 classifier): z_gate = 3.5599

## Results -- REAL device floor (no injection)

- QPU-s this pass (sum job.usage()): **72**

| classifier | power | FP rate | TP/dark | FP/null | inconclusive |
|---|---|---|---|---|---|
| v3 device-referenced differential | 1.00 | 0.00 | 5/5 | 0/3 | 0 |
| v2 absolute gate (honest N rescore) | 1.00 | 0.00 | 5/5 | 0/3 | 0 |
| v2 absolute gate (frozen v2 estimator) | 1.00 | 0.00 | 5/5 | 0/3 | 0 |

| inst | truth | phys | v3 z_diff(YX,ZX) | max|z_diff| | v3? | v2 |z|(honest) | v2? | <XX>ref |
|---|---|---|---|---|---|---|---|---|
| 0 | null | (0, 1, 2) | (-0.54,+1.05) | 1.05 | - | 1.02 | - | +0.895 |
| 1 | null | (0, 1, 2) | (-0.43,+1.26) | 1.26 | - | 2.61 | - | +0.906 |
| 2 | dark | (0, 1, 2) | (+2.17,+6.04) | 6.04 | DET | 9.39 | DET | +0.892 |
| 3 | dark | (0, 1, 2) | (+6.93,-8.30) | 8.30 | DET | 12.95 | DET | +0.900 |
| 4 | null | (0, 1, 2) | (-0.33,-1.62) | 1.62 | - | 2.40 | - | +0.887 |
| 5 | dark | (0, 1, 2) | (-8.26,+1.08) | 8.26 | DET | 10.23 | DET | +0.876 |
| 6 | dark | (0, 1, 2) | (+2.10,-4.21) | 4.21 | DET | 6.57 | DET | +0.898 |
| 7 | dark | (0, 1, 2) | (+10.19,+0.97) | 10.19 | DET | 14.16 | DET | +0.896 |

## Submit decision

- REAL fez run: v3 differential scored power=1.00 FP=0.00; v2-absolute FP=0/3.
- contrast satisfied: **True**

## QPU usage / budget

- preflight estimate: model 19.1-57.2 s, empirical-fit **77.6 s** (budget 80 s, under_budget True)
- quota consumed (svc.usage delta): **0 s**  |  remaining after: 285 s

## One-sentence result (paper's marked slot)

> On ibm_fez, device-referenced differential decision (protocol v3) recovered full specificity -- power 1.00 on 5 planted-dark instances and false-positive rate 0.00 on 3 sealed nulls -- while the unreferenced absolute gate false-positived on 0/3 of the same nulls.
