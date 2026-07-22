# Protocol v3 -- ibm_fez drift / half-split / device-floor diagnostics

- Shots/block: 1536 (half-split 768/768, shot-ordered)  |  v3 gate |z_diff| = 3.5431
- **Pooled reference-block YX floor (this session, 12288 shots): corr = +0.0430, z = +4.77, sign '+'**
- Reference YX sign agreement across 8 instances: 7/8 sign '+'  (per-instance floor mean +0.0430, range -0.0169..+0.0781)
- All instances half-split stable: **True**

> Cross-run: the 6-null confirmation (15:21 UTC) had pooled YX z=-6.20 (corr -0.056, sign '-'); this v3 session (19:01 UTC) has pooled reference YX z=+4.77 (corr +0.0430, sign '+').  The device floor's sign/magnitude DRIFTED between calibration windows -- an absolute gate calibrated once cannot track this; the differential cancels it every run by construction.

## Per-instance differential + half-split

| inst | truth | argmax | z_diff full | 1st-half | 2nd-half | stable? | ref YX floor | <XX>ref |
|---|---|---|---|---|---|---|---|---|
| 0 | null | ZX | +1.05 | +0.66 | +0.82 | yes | +0.0143 | +0.895 |
| 1 | null | ZX | +1.26 | -0.87 | +2.66 | yes | +0.0781 | +0.906 |
| 2 | dark | ZX | +6.04 | +3.47 | +5.07 | yes | +0.0443 | +0.892 |
| 3 | dark | ZX | -8.30 | -6.04 | -5.70 | yes | +0.0703 | +0.900 |
| 4 | null | ZX | -1.62 | -1.17 | -1.12 | yes | +0.0729 | +0.887 |
| 5 | dark | YX | -8.26 | -6.29 | -5.38 | yes | +0.0404 | +0.876 |
| 6 | dark | ZX | -4.21 | -2.16 | -3.81 | yes | +0.0404 | +0.898 |
| 7 | dark | YX | +10.19 | +7.55 | +6.86 | yes | -0.0169 | +0.896 |
