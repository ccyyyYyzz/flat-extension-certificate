# C7 Step-2 blind benchmark -- REHEARSAL run

- Mode: **REHEARSAL** (AerSimulator.from_backend(ibm_marrakesh))
- Date: 2026-07-21  |  target backend: ibm_marrakesh
- Master seed: 20260723 (fresh; dry-run used 20260721)
- Instances: 8 (dark 5 / null 3)
- qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, qiskit-aer 0.17.2

## Sealed manifest (sealed BEFORE execution)

- sealed_key_sha256: `eca687ff1bbd313ec6c4b187af5c9d2d76bd97aad5b7f5ff05fbdc3d96ca52b3`
- unblind sha256 verification: **True**
- plan_sha256: `422e1ec76949be738016397a8420ee7e234ee77af4b327600777eff424b84c8d`
- frozen estimator config_sha256: `76296041f509bda06a9d0e7433ee6312f1ba769bd70e39bbfb1956eada7e8543`
- frozen z_gate: 3.5599  |  cert shots/setting: 2048  |  alpha_exp: 0.01
- estimator config v2 (decision family: depth2_signal_subspace_YX_ZX, pairs ['YX', 'ZX']); z_gate_shot=3.4205 inflated by 1/lambda=1.0408 for readout eps=0.0099 -> device-aware z_gate=3.5599

### Pre-flight run plan (target ibm_marrakesh, seed 20260723)

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
| certificate | 1.00 | 0.00 | 5/5 | 0/3 | 9 settings x 2048 (decision family) |
| structured_lowmem | 1.00 | 0.00 | 5/5 | 0/3 | analysis-only on cert counts (O6) |
| marginal_qpt | 0.00 | 0.00 | 0/5 | 0/3 | 36 settings x 512 (blindness null; must be 0) |
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
| 0 | `4df03378-6c15-457a-8c61-5e11cd8a25a5` | 54 | 0 | 2026-07-21T14:05:35.694985+00:00 |
| 1 | `ce2fbb31-22e7-461d-8738-9c016f51be7b` | 54 | 0 | 2026-07-21T14:05:46.360929+00:00 |
| 2 | `0456595d-cd65-4707-bcff-dbb32a1a7212` | 45 | 0 | 2026-07-21T14:05:57.002092+00:00 |
| 3 | `36837ed9-8338-4aff-a3b7-eb76114a6bf0` | 45 | 0 | 2026-07-21T14:06:07.943647+00:00 |
| 4 | `446c28ce-77e6-40b3-8aaa-f3de0cb542d8` | 45 | 0 | 2026-07-21T14:06:19.220561+00:00 |
| 5 | `1b983f6c-1ad8-4c2a-8360-a6d0a260aef6` | 45 | 0 | 2026-07-21T14:06:30.199949+00:00 |
| 6 | `4e8190e7-6040-4f9e-a512-a21c600c2850` | 45 | 0 | 2026-07-21T14:06:41.375180+00:00 |
| 7 | `dde273ef-408d-41c1-9346-954cc86bb85a` | 45 | 0 | 2026-07-21T14:06:53.221970+00:00 |
