# C7 Step-2 blind benchmark -- simulator dry run

- Date: 2026-07-21
- Master seed: 20260721 (deterministic)
- Instances: 20  (dark 7 / null 13, delta in [0.2, 0.4], 2-slot comb)
- Backend: qiskit 2.5.0, qiskit-aer 0.17.2
- IBM Heron approximation: 1Q depol 0.00029, 2Q depol 0.002334, readout 0.00123 (DEVICE_NOISE_PARAMETERS.md)

## Sealed manifest

- sealed_key_sha256: `5d9f37d9e6837640f70e7157bd213165d8626c06d9c13874959b91ad114a7599`
- unblind sha256 verification: **True**
- frozen estimator config_sha256: `87a1da0d424211ac18644432930b848df704f3dcd9733b801bdfdd557b9378c4`
- frozen z_gate: 4.0032  |  n_shots/setting: 2048  |  alpha_exp: 0.01
- Step-1 artifact sha256: `f1bff524ce759df8e30131f3071e8d785fb34e4f2a41bff83a3cd6fd24f88e1c`  (n_ref=341, power_factor=2.455)

## Score table -- noiseless

| method | power | FP rate | TP/dark | FP/null | settings | shots/inst | op-basis |
|---|---|---|---|---|---|---|---|
| certificate | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |
| structured_lowmem | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |
| marginal_qpt | 0.00 | 0.00 | 0/7 | 0/13 | 36 | 73728 | - |
| restricted_ptt | 1.00 | 0.00 | 7/7 | 0/13 | 324 | 663552 | unitary-restricted (6x6 IC preps x 9 pairs) |
| full_ptt | 1.00 | 0.00 | 7/7 | 0/13 | 144 | 294912 | 256 |
| heldout_depth3 | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |

## Score table -- ibm_heron_approx

| method | power | FP rate | TP/dark | FP/null | settings | shots/inst | op-basis |
|---|---|---|---|---|---|---|---|
| certificate | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |
| structured_lowmem | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |
| marginal_qpt | 0.00 | 0.00 | 0/7 | 0/13 | 36 | 73728 | - |
| restricted_ptt | 1.00 | 0.00 | 7/7 | 0/13 | 324 | 663552 | unitary-restricted (6x6 IC preps x 9 pairs) |
| full_ptt | 1.00 | 0.00 | 7/7 | 0/13 | 144 | 294912 | 256 |
| heldout_depth3 | 1.00 | 0.00 | 7/7 | 0/13 | 9 | 18432 | - |

## O7 drift diagnostic (circuit-order randomization)

- noiseless: 14 probes, worst drift-z 2.30 (gate 3.0), consistent: **True**
- ibm_heron_approx: 14 probes, worst drift-z 1.43 (gate 3.0), consistent: **True**

## Headlines

- [noiseless] certificate power 1.00 / FP 0.00 at 9 settings; marginal-QPT power 0.00 (blind by Theorem 1); full-PTT power 1.00 at 256 operation-basis ops.
- [ibm_heron_approx] certificate power 1.00 / FP 0.00 at 9 settings; marginal-QPT power 0.00 (blind by Theorem 1); full-PTT power 1.00 at 256 operation-basis ops.
- Resource separation: certificate 9 settings vs full-PTT 256 (=d^4k) vs restricted-PTT 324; the structured O6 comparator matches the certificate at 9 settings.
