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
| 0 | `824d2358-90ac-4a6b-9b18-17ba4618fa89` | 54 | 0 | 2026-07-21T14:12:07.100641+00:00 |
| 1 | `bebd8d76-7c47-4c68-9051-a459d3a8d6ac` | 54 | 0 | 2026-07-21T14:12:18.015095+00:00 |
| 2 | `37e55a24-8b75-423e-a458-e748b6c9a242` | 45 | 0 | 2026-07-21T14:12:28.822747+00:00 |
| 3 | `555be40d-7509-420a-ab3f-ed163b9e69e0` | 45 | 0 | 2026-07-21T14:12:39.273636+00:00 |
| 4 | `a087bbf6-0aac-4379-b666-f1b606645b81` | 45 | 0 | 2026-07-21T14:12:50.755711+00:00 |
| 5 | `b6f19d40-80c6-42ca-ac6d-f284e6bfcd83` | 45 | 0 | 2026-07-21T14:13:01.273295+00:00 |
| 6 | `c4f2ce4e-7238-4a2f-812e-6e84a5d3aff9` | 45 | 0 | 2026-07-21T14:13:13.017782+00:00 |
| 7 | `2616a0dd-a601-4c13-aceb-1e02a03e3755` | 45 | 0 | 2026-07-21T14:13:23.894077+00:00 |
