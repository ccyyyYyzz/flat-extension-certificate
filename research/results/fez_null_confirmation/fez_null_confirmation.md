# ibm_fez null-floor confirmation -- HARDWARE run

- Mode: **HARDWARE** (IBM hardware ibm_fez)
- Date: 2026-07-21  |  target: ibm_fez
- Master seed: 20260728  |  qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, aer 0.17.2
- Null instances: 6 x certificate 9 settings x 2048 shots = 110592 shots, 1 job of 54 PUBs
- manifest_sha256: `fc4ea351282cb323a69ff02df5b78afa5d9bee5fdbe3e9425ebde014a050f878`  |  hypotheses_sha256: `c3c61c14f98c92f88fd7e22d56cf3363f9862c9f2029668453bab17fdca34e7a` (both sealed pre-exec)
- Reference fez frozen gate z_gate = 3.5599; decision threshold |z| = 3.6

## VERDICT: **H-intrinsic** -- pooled/consistent-sign (per-instance magnitude below fez gate)

the DEVICE-INTRINSIC reading is confirmed, NOT a drift fluke: all 6/6 null YX are sign - and the pooled 12288-shot YX z=-6.20 decisively rejects a zero floor (p~1e-9), with stable within-run half-splits. BUT the per-instance amplitude is below the fez run: only 1/6 individually clear |z|>3.6 (fez nulls hit 4.89/4.40), so the coherent floor's MAGNITUDE is smaller/varies day-to-day while its STRUCTURE (sign, locus) is fixed.

- reproducible fixed floor (pooled test): **True**
- YX |z|>3.6 on **1/6** instances; pooled 12288-shot YX z = **-6.20** (corr -0.06)
- global sign agreement: **1.00** (dominant sign -)
- within-run half-split stable on all instances: **True**
- distinct physical triples used: ['(0, 1, 2)']

## Per-instance certificate statistics

| inst | phys (S1,E,S2) | YX corr | YX z | 1st-half z | 2nd-half z | ZX z | XX corr | max|z| YX/ZX | >3.6? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (0, 1, 2) | -0.0156 | -0.71 | -1.06 | +0.06 | +0.66 | +0.9004 | 0.71 | no |
| 1 | (0, 1, 2) | -0.1025 | -4.66 | -4.67 | -1.94 | -0.71 | +0.9141 | 4.66 | YES |
| 2 | (0, 1, 2) | -0.0391 | -1.77 | -1.82 | -0.69 | -1.37 | +0.8877 | 1.77 | no |
| 3 | (0, 1, 2) | -0.0615 | -2.79 | -3.52 | -0.44 | +0.53 | +0.9043 | 2.79 | no |
| 4 | (0, 1, 2) | -0.0449 | -2.03 | -1.94 | -0.94 | -1.19 | +0.8965 | 2.03 | no |
| 5 | (0, 1, 2) | -0.0713 | -3.23 | -1.75 | -2.82 | +0.40 | +0.9062 | 3.23 | no |

## Physical qubit readout errors (from backend.target)

- triple (0, 1, 2) readout error: {'0': 0.0118408203125, '1': 0.017822265625, '2': 0.0052490234375}

## Job

- job_id: `d9fosr9htsac739foad0`  |  PUBs: 54  |  usage QPU-s (job.usage()): 31
- submit: 2026-07-21T15:21:59.376097+00:00  |  result: 2026-07-21T15:22:54.797365+00:00
- quota consumed this run (svc.usage delta): **31 s**  |  remaining after: 357 s
- QPU-seconds preflight estimate: 6.5-19.6 s (budget 60 s)
