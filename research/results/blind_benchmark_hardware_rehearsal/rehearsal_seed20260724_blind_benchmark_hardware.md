# C7 Step-2 blind benchmark -- REHEARSAL run

- Mode: **REHEARSAL** (AerSimulator.from_backend(ibm_marrakesh))
- Date: 2026-07-21  |  target backend: ibm_marrakesh
- Master seed: 20260724 (fresh; dry-run used 20260721)
- Instances: 8 (dark 3 / null 5)
- qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, qiskit-aer 0.17.2

## Sealed manifest (sealed BEFORE execution)

- sealed_key_sha256: `01a88bf888d4d9d1f09bb1cbfa0b013503bc14dd46aacf5a11d4eea36428d845`
- unblind sha256 verification: **True**
- plan_sha256: `3169b1f73208768a65dfef9661dcee884487d0a908dd40c1f13c2c798525acb7`
- frozen estimator config_sha256: `76296041f509bda06a9d0e7433ee6312f1ba769bd70e39bbfb1956eada7e8543`
- frozen z_gate: 3.5599  |  cert shots/setting: 2048  |  alpha_exp: 0.01
- estimator config v2 (decision family: depth2_signal_subspace_YX_ZX, pairs ['YX', 'ZX']); z_gate_shot=3.4205 inflated by 1/lambda=1.0408 for readout eps=0.0099 -> device-aware z_gate=3.5599

### Pre-flight run plan (target ibm_marrakesh, seed 20260724)

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
| heldout_depth3 | nan | 0.00 | 0/0 | 0/2 | 9 settings x 2048 on 2 instances (confirmation) |

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
| 0 | `703bca49-43dd-40d2-98dd-a6bc87f4822c` | 54 | 0 | 2026-07-21T14:07:46.875040+00:00 |
| 1 | `eb8fb124-96f5-4c69-b012-346ec753df05` | 54 | 0 | 2026-07-21T14:07:57.731623+00:00 |
| 2 | `028565cf-448e-4f6a-b293-e9d08254bb6c` | 45 | 0 | 2026-07-21T14:08:08.436929+00:00 |
| 3 | `6997a104-685d-4c4f-a176-0fefceb760cc` | 45 | 0 | 2026-07-21T14:08:18.908631+00:00 |
| 4 | `69aa280e-258b-41d4-8178-0bcb54702ffa` | 45 | 0 | 2026-07-21T14:08:30.075530+00:00 |
| 5 | `7be86d52-b36e-4de0-91f6-771212eae6dc` | 45 | 0 | 2026-07-21T14:08:41.301118+00:00 |
| 6 | `0649f1eb-7af9-410b-b198-5b2f00304d4d` | 45 | 0 | 2026-07-21T14:08:52.610694+00:00 |
| 7 | `19377e42-3931-4038-9cde-85d884921685` | 45 | 0 | 2026-07-21T14:09:03.335525+00:00 |
