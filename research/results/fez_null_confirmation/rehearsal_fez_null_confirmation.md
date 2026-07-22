# ibm_fez null-floor confirmation -- REHEARSAL run

- Mode: **REHEARSAL** (AerSimulator.from_backend(ibm_fez))
- Date: 2026-07-21  |  target: ibm_fez
- Master seed: 20260728  |  qiskit 2.5.0, qiskit-ibm-runtime 0.48.0, aer 0.17.2
- Null instances: 6 x certificate 9 settings x 2048 shots = 110592 shots, 1 job of 54 PUBs
- manifest_sha256: `c2adc0d9c84cb9dee64349534ef9c5c05580ec0e07a1608614f267b3b630dc19`  |  hypotheses_sha256: `c3c61c14f98c92f88fd7e22d56cf3363f9862c9f2029668453bab17fdca34e7a` (both sealed pre-exec)
- Reference fez frozen gate z_gate = 3.5599; decision threshold |z| = 3.6

## VERDICT: **H-drift**

YX |z|>3.6 on only 0/6 instances (excess did not reproduce); pooled YX z=-1.44

- YX |z|>3.6 on **0/6** instances; pooled 12288-shot YX z = **-1.44** (corr -0.01)
- sign consistent per qubit-triple: **True** (distinct signs over |z|>2: [])
- within-run half-split stable on all instances: **True**
- distinct physical triples used: ['(0, 1, 2)']

## Per-instance certificate statistics

| inst | phys (S1,E,S2) | YX corr | YX z | 1st-half z | 2nd-half z | ZX z | XX corr | max|z| YX/ZX | >3.6? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | (0, 1, 2) | -0.0371 | -1.68 | -2.57 | +0.19 | -0.62 | +0.9287 | 1.68 | no |
| 1 | (0, 1, 2) | +0.0137 | +0.62 | -0.31 | +1.19 | -0.53 | +0.9473 | 0.62 | no |
| 2 | (0, 1, 2) | -0.0059 | -0.27 | +0.31 | -0.69 | +0.13 | +0.9453 | 0.27 | no |
| 3 | (0, 1, 2) | +0.0186 | +0.84 | +1.13 | +0.06 | -1.41 | +0.9541 | 1.41 | no |
| 4 | (0, 1, 2) | -0.0322 | -1.46 | -0.56 | -1.50 | -0.13 | +0.9365 | 1.46 | no |
| 5 | (0, 1, 2) | -0.0352 | -1.59 | -2.00 | -0.25 | +0.84 | +0.9541 | 1.59 | no |

## Physical qubit readout errors (from backend.target)

- triple (0, 1, 2) readout error: {'0': 0.0118408203125, '1': 0.017822265625, '2': 0.0052490234375}

## Job

- job_id: `d76d002d-fdd1-44e8-8da6-7cb7cde0c87d`  |  PUBs: 54  |  usage QPU-s (job.usage()): 0
- submit: 2026-07-21T15:21:18.055228+00:00  |  result: 2026-07-21T15:21:24.337003+00:00
- QPU-seconds preflight estimate: 6.5-19.6 s (budget 60 s)
