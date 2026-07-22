# C7 Step-2 blind benchmark -- HARDWARE run

- Mode: **HARDWARE** (IBM hardware ibm_fez)
- Date: 2026-07-21  |  target backend: ibm_fez
- Master seed: 20260727 (fresh; dry-run used 20260721)
- Instances: 8 (dark 5 / null 3)
- qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, qiskit-aer 0.17.2

## Sealed manifest (sealed BEFORE execution)

- sealed_key_sha256: `c0dcae2602da1873cb31106780808750dec16030224c2e614b123e39df283918`
- unblind sha256 verification: **True**
- plan_sha256: `f9c14c2b65be3412a1f65ae415b42521fe4a2a94619e0dae0dbdbd8a49d01f14`
- frozen estimator config_sha256: `76296041f509bda06a9d0e7433ee6312f1ba769bd70e39bbfb1956eada7e8543`
- frozen z_gate: 3.5599  |  cert shots/setting: 2048  |  alpha_exp: 0.01
- estimator config v2 (decision family: depth2_signal_subspace_YX_ZX, pairs ['YX', 'ZX']); z_gate_shot=3.4205 inflated by 1/lambda=1.0408 for readout eps=0.0099 -> device-aware z_gate=3.5599

### Pre-flight run plan (target ibm_fez, seed 20260727)

| arm | instances | settings | shots/setting | PUBs | shots |
|---|---|---|---|---|---|
| certificate | 8 | 9 | 2048 | 72 | 147456 |
| marginal_qpt | 8 | 36 | 512 | 288 | 147456 |
| heldout_depth3 | 2 | 9 | 2048 | 18 | 36864 |
| **total** | - | - | - | **378** | **331776** |

- Jobs (job mode, 1 per instance): **8**
- QPU-seconds estimate: **24.6-73.8 s** (per-shot 50-150 us + per-job 1-3 s)
- Under 7-min Open-plan reserve (420 s): **True**
- Structured comparator: analysis-only on certificate counts (0 extra shots).  Full PTT: simulator-only (never hardware).
- Marginal-QPT blindness null @ 512 shots: se=0.0442, 95% CI half-width=**+/-0.0866** (99%: +/-0.1138)

## Score table

| method | power | FP rate | TP/dark | FP/null | note |
|---|---|---|---|---|---|
| certificate | 1.00 | 0.67 | 5/5 | 2/3 | 9 settings x 2048 (decision family) |
| structured_lowmem | 1.00 | 0.67 | 5/5 | 2/3 | analysis-only on cert counts (O6) |
| marginal_qpt | 0.00 | 0.00 | 0/5 | 0/3 | 36 settings x 512 (blindness null; must be 0) |
| heldout_depth3 | 1.00 | 0.00 | 1/1 | 0/1 | 9 settings x 2048 on 2 instances (confirmation) |

## O7 drift discipline

- Deterministic per-job PUB interleave (shuffle seeds recorded); 8 jobs, submission + result timestamps recorded.
- Calibration snapshot (ibm_fez.target) recorded pre-submission.

## Resource ledger

- Jobs: 8  |  PUBs: 378  |  total shots: 331776
  - certificate: 8 inst x 9 settings x 2048 shots = 147456 shots
  - marginal_qpt: 8 inst x 36 settings x 512 shots = 147456 shots
  - heldout_depth3: 2 inst x 9 settings x 2048 shots = 36864 shots
- Structured comparator: 0 extra shots (analysis-only).  Full PTT: simulator-only (excluded).

## Jobs

| instance | job_id | PUBs | usage QPU-s | submit (UTC) |
|---|---|---|---|---|
| 0 | `d9fojtkinv1c73arl660` | 54 | 17 | 2026-07-21T15:02:56.373111+00:00 |
| 1 | `d9fok3ineu4c739psm30` | 54 | 17 | 2026-07-21T15:03:20.644529+00:00 |
| 2 | `d9fok9kjeosc73fk8qqg` | 45 | 12 | 2026-07-21T15:03:44.588625+00:00 |
| 3 | `d9fokh4jeosc73fk8rb0` | 45 | 12 | 2026-07-21T15:04:14.656989+00:00 |
| 4 | `d9fokq1htsac739fnvt0` | 45 | 12 | 2026-07-21T15:04:50.532413+00:00 |
| 5 | `d9fol0sjeosc73fk8s70` | 45 | 12 | 2026-07-21T15:05:17.554755+00:00 |
| 6 | `d9fol6aneu4c739psnug` | 45 | 12 | 2026-07-21T15:05:40.432889+00:00 |
| 7 | `d9folccjeosc73fk8sv0` | 45 | 12 | 2026-07-21T15:06:04.236561+00:00 |
