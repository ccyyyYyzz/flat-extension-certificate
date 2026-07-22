# Step-1 separator noise-power simulation

- Date: 2026-07-21
- Test: dark direction present (delta=delta_0) vs absent (delta=0), two-sided, alpha=0.05, power=0.95
- Correlator: <Y_S1 X_S2> (S1=S2=|+>), ideal E(delta)=sin(delta), ideal slope=1
- z_(alpha/2)=1.959964, z_beta=1.644854
- N_exact uses exact binomial variance 1-C^2; N_gauss is the variance-1 Gaussian shortcut
- N_total = N_exact x (1 + 48): correlator plus the 48 atomic-baseline statistics at matched shot budget
- MC empirical power: 40000 binomial-sampled trials, seed=20260721

## Quantinuum H2 (trapped ion, QCCD)

Citation: Moses PRX 13 041052; DeCross PRX 15 021052  |  mid-circuit model: `ion`

- 1Q 2.5e-5, 2Q 1.83e-3, SPAM 1.6e-3 (Moses, verified)
- SPAM split prep 6.0e-4 + readout 1.0e-3 (total verified)
- S1 dephasing 3.0e-4 = midpoint of verified [2.2e-4,4.0e-4]
- crosstalk 5.0e-6 (verified meas crosstalk 4.5e-6)
- phi_kick inert for the ion model

CP-TP validation: all 6 channels pass (Choi min-eig >= -1e-12, TP defect <= 1e-10): **True**

| phi_kick | slope@0 | contrast loss | delta0 | C1(noisy) | N_exact | N_gauss | N_total | MC power |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.949 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.948 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.947 |
| 0.005 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.952 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.948 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.949 |
| 0.010 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.952 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.950 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.950 |
| 0.020 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.950 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.949 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.949 |
| 0.050 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.952 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.949 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.948 |
| 0.100 | 0.98990 | 1.010% | 0.05 | 0.04947 | 5,303 | 5,309 | 259,847 | 0.953 |
|  |  |  | 0.1 | 0.09883 | 1,325 | 1,331 | 64,925 | 0.948 |
|  |  |  | 0.2 | 0.19666 | 331 | 336 | 16,219 | 0.947 |

## IBM Heron r2 (transmon, heavy-hex)

Citation: ibm_marrakesh arXiv:2605.24252; AbuGhanem J Supercomput 81 687  |  mid-circuit model: `ibm`

- 2Q 2.334e-3, readout 1.23e-3, T2 96.89us (marrakesh, verified)
- 1Q 2.9e-4 (Heron ~3-4e-4; verified Eagle anchor 2.411e-4)
- MCM window 1.5us UNVERIFIED lower bound
- ZZ phase-kick phi_kick UNVERIFIED (Govia mechanism)
- prep_error 0 (no verified IBM prep number; readout carries SPAM)

CP-TP validation: all 6 channels pass (Choi min-eig >= -1e-12, TP defect <= 1e-10): **True**

| phi_kick | slope@0 | contrast loss | delta0 | C1(noisy) | N_exact | N_gauss | N_total | MC power |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.97438 | 2.562% | 0.05 | 0.04870 | 5,474 | 5,480 | 268,226 | 0.948 |
|  |  |  | 0.1 | 0.09728 | 1,368 | 1,374 | 67,032 | 0.949 |
|  |  |  | 0.2 | 0.19358 | 341 | 347 | 16,709 | 0.950 |
| 0.005 | 0.97437 | 2.563% | 0.05 | 0.04870 | 5,474 | 5,480 | 268,226 | 0.951 |
|  |  |  | 0.1 | 0.09727 | 1,368 | 1,374 | 67,032 | 0.948 |
|  |  |  | 0.2 | 0.19358 | 341 | 347 | 16,709 | 0.950 |
| 0.010 | 0.97433 | 2.567% | 0.05 | 0.04870 | 5,474 | 5,480 | 268,226 | 0.948 |
|  |  |  | 0.1 | 0.09727 | 1,368 | 1,374 | 67,032 | 0.949 |
|  |  |  | 0.2 | 0.19357 | 341 | 347 | 16,709 | 0.950 |
| 0.020 | 0.97419 | 2.581% | 0.05 | 0.04869 | 5,476 | 5,482 | 268,324 | 0.950 |
|  |  |  | 0.1 | 0.09726 | 1,368 | 1,374 | 67,032 | 0.948 |
|  |  |  | 0.2 | 0.19354 | 341 | 347 | 16,709 | 0.950 |
| 0.050 | 0.97316 | 2.684% | 0.05 | 0.04864 | 5,488 | 5,494 | 268,912 | 0.951 |
|  |  |  | 0.1 | 0.09715 | 1,371 | 1,377 | 67,179 | 0.951 |
|  |  |  | 0.2 | 0.19334 | 342 | 348 | 16,758 | 0.945 |
| 0.100 | 0.96951 | 3.049% | 0.05 | 0.04846 | 5,529 | 5,535 | 270,921 | 0.950 |
|  |  |  | 0.1 | 0.09679 | 1,382 | 1,388 | 67,718 | 0.947 |
|  |  |  | 0.2 | 0.19261 | 345 | 351 | 16,905 | 0.951 |

## Headlines

- Under Quantinuum H2 (trapped ion, QCCD) parameters [Moses PRX 13 041052; DeCross PRX 15 021052], N = 64,925 total shots certify the planted dark direction delta_0 = 0.1 at 95% power / alpha = 0.05 (correlator 1,325 + baseline 48x).
- Under IBM Heron r2 (transmon, heavy-hex) parameters [ibm_marrakesh arXiv:2605.24252; AbuGhanem J Supercomput 81 687] with ZZ phase-kick phi=0.0 rad, N = 67,032 total shots certify the planted dark direction delta_0 = 0.1 at 95% power / alpha = 0.05 (correlator 1,368 + baseline 48x).
- Under IBM Heron r2 (transmon, heavy-hex) parameters [ibm_marrakesh arXiv:2605.24252; AbuGhanem J Supercomput 81 687] with ZZ phase-kick phi=0.1 rad, N = 67,718 total shots certify the planted dark direction delta_0 = 0.1 at 95% power / alpha = 0.05 (correlator 1,382 + baseline 48x).

## Ion-vs-IBM comparison across the phi_kick sweep

- delta0=0.05: ion N_total=259,847 (phi-independent); IBM N_total ranges 268,226 (phi=0) .. 270,921 (phi=0.1); ion favored across the whole sweep: True.
- delta0=0.1: ion N_total=64,925 (phi-independent); IBM N_total ranges 67,032 (phi=0) .. 67,718 (phi=0.1); ion favored across the whole sweep: True.
- delta0=0.2: ion N_total=16,219 (phi-independent); IBM N_total ranges 16,709 (phi=0) .. 16,905 (phi=0.1); ion favored across the whole sweep: True.
- The phi_kick sweep does NOT flip the ion-vs-IBM ordering: ion needs fewer certification shots at every swept phi_kick and every delta0.
- phi_kick only degrades IBM (monotone slope drop / N_total rise); the ion model is phi_kick-independent by construction (no ZZ-kick term). The ion advantage widens as phi_kick grows.
