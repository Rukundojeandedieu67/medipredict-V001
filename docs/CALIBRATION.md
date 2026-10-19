# Simulator calibration report

All values are computed from simulated data and are not observed Rwanda inventory outcomes.

- Seed runs: `20` sequential seeds beginning at `2026`.
- Rare-group threshold: node/group mean daily demand below 0.5 units.
- Selected policy: rare local target `2` units, processing delay `6` hours; common groups retain cover-day ordering.
- National unmet demand: `0.107%` (target 0.18%-0.38%).
- Combined expiry: `1.202%`; per-group limit is strictly below 10%.

## Recalibration search

| Rare local target | Processing delay (h) | Unmet demand % | O+ expiry | O- expiry | A+ expiry | A- expiry | B+ expiry | B- expiry | AB+ expiry | AB- expiry | Min episodes/group | Meets all targets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 6 | 4.342 | 0.00% | 0.15% | 0.00% | 5.76% | 0.00% | 10.18% | 0.14% | 49.05% | 36 | no |
| 0 | 12 | 4.342 | 0.00% | 0.15% | 0.00% | 5.76% | 0.00% | 10.18% | 0.14% | 49.05% | 36 | no |
| 0 | 24 | 4.902 | 0.00% | 0.15% | 0.00% | 5.94% | 0.00% | 10.33% | 0.13% | 48.98% | 1047 | no |
| 0 | 48 | 6.483 | 0.00% | 0.15% | 0.00% | 6.34% | 0.00% | 10.76% | 0.14% | 49.17% | 4510 | no |
| 1 | 6 | 0.902 | 0.00% | 0.21% | 0.00% | 6.45% | 0.00% | 11.29% | 0.22% | 56.33% | 46 | no |
| 1 | 12 | 0.902 | 0.00% | 0.21% | 0.00% | 6.45% | 0.00% | 11.29% | 0.22% | 56.33% | 46 | no |
| 1 | 24 | 2.103 | 0.00% | 0.21% | 0.00% | 6.69% | 0.00% | 11.56% | 0.22% | 56.33% | 390 | no |
| 1 | 48 | 4.216 | 0.00% | 0.22% | 0.00% | 7.10% | 0.00% | 12.10% | 0.23% | 56.76% | 677 | no |
| 2 | 6 | 0.107 | 0.00% | 0.79% | 0.00% | 15.44% | 0.00% | 24.02% | 0.89% | 71.57% | 17 | no |
| 2 | 12 | 0.107 | 0.00% | 0.79% | 0.00% | 15.44% | 0.00% | 24.02% | 0.89% | 71.57% | 17 | no |
| 2 | 24 | 0.648 | 0.00% | 0.73% | 0.00% | 15.08% | 0.00% | 23.53% | 0.83% | 71.13% | 38 | no |
| 2 | 48 | 2.343 | 0.00% | 0.69% | 0.00% | 14.93% | 0.00% | 23.23% | 0.80% | 70.76% | 87 | no |

## Processing-delay sensitivity at rare local target 2 units

All rows aggregate the same 20 seeds; only processing delay varies. In this daily simulation, 6-hour and 12-hour delays yield identical results because total route lead time is rounded up to daily steps.

| Processing delay (h) | Unmet demand | O+ expiry | O- expiry | A+ expiry | A- expiry | B+ expiry | B- expiry | AB+ expiry | AB- expiry | Minimum episodes/group |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.107% | 0.00% | 0.79% | 0.00% | 15.44% | 0.00% | 24.02% | 0.89% | 71.57% | 17 |
| 12 | 0.107% | 0.00% | 0.79% | 0.00% | 15.44% | 0.00% | 24.02% | 0.89% | 71.57% | 17 |
| 24 | 0.648% | 0.00% | 0.73% | 0.00% | 15.08% | 0.00% | 23.53% | 0.83% | 71.13% | 38 |
| 48 | 2.343% | 0.00% | 0.69% | 0.00% | 14.93% | 0.00% | 23.23% | 0.80% | 70.76% | 87 |

## Selected policy results by blood group

| Group | Stockout episodes | Stockout rate | Mean closing stock | Supply | Expired | Expiry / supply |
|---|---:|---:|---:|---:|---:|---:|
| O+ | 43 | 0.010% | 19.950 | 1,622,835 | 0 | 0.000% |
| O- | 752 | 0.172% | 2.323 | 139,752 | 1,108 | 0.793% |
| A+ | 451 | 0.104% | 10.029 | 804,903 | 0 | 0.000% |
| A- | 424 | 0.097% | 1.879 | 54,186 | 8,365 | 15.438% |
| B+ | 614 | 0.141% | 7.736 | 610,851 | 0 | 0.000% |
| B- | 171 | 0.039% | 1.900 | 44,830 | 10,767 | 24.017% |
| AB+ | 750 | 0.172% | 2.323 | 137,541 | 1,224 | 0.890% |
| AB- | 17 | 0.004% | 1.939 | 27,847 | 19,929 | 71.566% |

## Acceptance

- National unmet demand within 0.28% ±0.1 percentage point: **FAIL**.
- Every group's expiry below 10%: **FAIL**.
- At least 30 stockout episodes per group across these seeds: **FAIL**.

No candidate in this bounded search satisfies all three constraints simultaneously. The selected candidate minimizes the documented normalized unmet-demand, expiry-excess, and episode-deficit score; failed targets are reported rather than hidden.
