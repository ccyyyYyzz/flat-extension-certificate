# C7 Step-2 blind benchmark -- HARDWARE run

- Mode: **HARDWARE** (IBM hardware ibm_marrakesh)
- Date: 2026-07-21  |  target backend: ibm_marrakesh
- Master seed: 20260725 (fresh; dry-run used 20260721)
- Instances: 8 (dark 3 / null 5)
- qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, qiskit-aer 0.17.2

## Sealed manifest (sealed BEFORE execution)

- sealed_key_sha256: `99baf933bd533f84469c2e8a9d049a8f365294fc8e04de5b79db026070bb1274`
- unblind sha256 verification: **True**
- plan_sha256: `fe6f3adcd2452c6ec1a0e6e90550b44b18724699010ad545086a3059edc7855c`
- frozen estimator config_sha256: `76296041f509bda06a9d0e7433ee6312f1ba769bd70e39bbfb1956eada7e8543`
- frozen z_gate: 3.5599  |  cert shots/setting: 2048  |  alpha_exp: 0.01
- estimator config v2 (decision family: depth2_signal_subspace_YX_ZX, pairs ['YX', 'ZX']); z_gate_shot=3.4205 inflated by 1/lambda=1.0408 for readout eps=0.0099 -> device-aware z_gate=3.5599

### Pre-flight run plan (target ibm_marrakesh, seed 20260725)

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
| certificate | 1.00 | 0.00 | 3/3 | 0/5 | 9 settings x 2048 (decision family) |
| structured_lowmem | 1.00 | 0.00 | 3/3 | 0/5 | analysis-only on cert counts (O6) |
| marginal_qpt | 0.00 | 0.00 | 0/3 | 0/5 | 36 settings x 512 (blindness null; must be 0) |
| heldout_depth3 | 1.00 | 0.00 | 1/1 | 0/1 | 9 settings x 2048 on 2 instances (confirmation) |

## O7 drift discipline

- Deterministic per-job PUB interleave (shuffle seeds recorded); 8 jobs, submission + result timestamps recorded.
- Calibration snapshot (ibm_marrakesh.target) recorded pre-submission.

## Resource ledger

- Jobs: 8  |  PUBs: 378  |  total shots: 331776
  - certificate: 8 inst x 9 settings x 2048 shots = 147456 shots
  - marginal_qpt: 8 inst x 36 settings x 512 shots = 147456 shots
  - heldout_depth3: 2 inst x 9 settings x 2048 shots = 36864 shots
- Structured comparator: 0 extra shots (analysis-only).  Full PTT: simulator-only (excluded).

## Jobs

| instance | job_id | PUBs | usage QPU-s | submit (UTC) |
|---|---|---|---|---|
| 0 | `d9fnvbqneu4c739prti0` | 54 | 17 | 2026-07-21T14:19:05.084223+00:00 |
| 1 | `d9foa2cinv1c73arkq70` | 54 | 17 | 2026-07-21T14:41:55.686731+00:00 |
| 2 | `d9foad1htsac739fnhk0` | 45 | 12 | 2026-07-21T14:42:38.673013+00:00 |
| 3 | `d9fobeineu4c739psbeg` | 45 | 12 | 2026-07-21T14:44:52.956075+00:00 |
| 4 | `d9foblaneu4c739psbmg` | 45 | 12 | 2026-07-21T14:45:19.970457+00:00 |
| 5 | `d9fodahhtsac739fnlqg` | 45 | 12 | 2026-07-21T14:48:52.594197+00:00 |
| 6 | `d9fodrkjeosc73fk8j0g` | 45 | 12 | 2026-07-21T14:50:00.508499+00:00 |
| 7 | `d9foftsjeosc73fk8lig` | 45 | 12 | 2026-07-21T14:54:25.928160+00:00 |
