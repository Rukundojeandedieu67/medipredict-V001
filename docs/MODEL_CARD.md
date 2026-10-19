# Phase 2 model card

## Intended use

Prototype early warning of simulated unmet blood demand for decision support. It is not validated on operational Rwanda stock readings and must not drive automatic dispatch.

## Data and split

All model observations are `simulated`. A pooled model uses blood group as a one-hot feature. Seed directories are separated 60%/20%/20% into train/validation/test, and each seed uses chronological 60%/20%/20% windows with a 3-day purge between windows. Validation data alone calibrates probabilities using Platt scaling.

- Training seeds: seed_00, seed_01, seed_02, seed_03, seed_04, seed_05, seed_06, seed_07, seed_08, seed_09, seed_10, seed_11
- Validation seeds: seed_12, seed_13, seed_14, seed_15
- Test seeds: seed_16, seed_17, seed_18, seed_19

| Split | Rows | 24h positives | 48h positives |
|---|---:|---:|---:|
| train | 1,224,000 | 1,121 | 2,240 |
| validation | 137,280 | 138 | 277 |
| test | 135,360 | 116 | 231 |

## Held-out results

Metrics are measured on the held-out simulated test seeds only. PR-AUC is average precision; bootstrap intervals use 200 resamples of seed-day clusters. Alert budgets are per seed-day, where a seed-day contains 30 district nodes × 8 blood groups. The days-of-cover baseline ranks the lowest-cover cells under the same budget.
Alerts per true event is the selected alert count divided by true-positive alerts on positive-label node/group-days (labels may overlap in the 48-hour horizon); it is not a count of deduplicated clinical episodes.

| Horizon | Budget | Model | PR-AUC | Brier | Precision | Recall | Alerts / true event |
|---|---:|---|---:|---:|---:|---:|---:|
| 24h | 1 | days_cover_rule | 0.0023 | 0.00086 | 0.0071 | 0.0345 | 141.00 |
| 24h | 1 | logistic_regression | 0.0056 | 0.00085 | 0.0071 | 0.0345 | 141.00 |
| 24h | 1 | lightgbm | 0.0076 | 0.00085 | 0.0053 | 0.0259 | 188.00 |
| 24h | 3 | days_cover_rule | 0.0023 | 0.00086 | 0.0053 | 0.0776 | 188.00 |
| 24h | 3 | logistic_regression | 0.0056 | 0.00085 | 0.0071 | 0.1034 | 141.00 |
| 24h | 3 | lightgbm | 0.0076 | 0.00085 | 0.0112 | 0.1638 | 89.05 |
| 24h | 5 | days_cover_rule | 0.0023 | 0.00086 | 0.0043 | 0.1034 | 235.00 |
| 24h | 5 | logistic_regression | 0.0056 | 0.00085 | 0.0071 | 0.1724 | 141.00 |
| 24h | 5 | lightgbm | 0.0076 | 0.00085 | 0.0103 | 0.2500 | 97.24 |
| 24h | 10 | days_cover_rule | 0.0023 | 0.00086 | 0.0037 | 0.1810 | 268.57 |
| 24h | 10 | logistic_regression | 0.0056 | 0.00085 | 0.0062 | 0.3017 | 161.14 |
| 24h | 10 | lightgbm | 0.0076 | 0.00085 | 0.0076 | 0.3707 | 131.16 |

24h LightGBM PR-AUC: 0.0076 (95% CI [0.0048, 0.0163]).

| Budget | LightGBM recall (95% CI) | Days-of-cover recall | Recall difference (LightGBM - baseline, paired 95% CI) |
|---:|---:|---:|---:|
| 1 | 0.0259 ([0.0000, 0.0604]) | 0.0345 | -0.0086 ([-0.0439, +0.0361]) |
| 3 | 0.1638 ([0.1055, 0.2316]) | 0.0776 | +0.0862 ([+0.0177, +0.1656]) |
| 5 | 0.2500 ([0.1809, 0.3306]) | 0.1034 | +0.1466 ([+0.0628, +0.2329]) |
| 10 | 0.3707 ([0.2636, 0.4579]) | 0.1810 | +0.1897 ([+0.0909, +0.2966]) |

24h LightGBM calibration curve (held-out test rows):

| Rows | Mean predicted risk | Observed positive rate |
|---:|---:|---:|
| 135,360 | 0.00104 | 0.00086 |

24h LightGBM breakdown at 10 alerts per seed-day:

| Blood group | Positives | PR-AUC | Recall |
|---|---:|---:|---:|
| O+ | 1 | 0.0044 | 0.0000 |
| O- | 32 | 0.0070 | 0.2812 |
| A+ | 15 | 0.0057 | 0.4667 |
| A- | 15 | 0.0268 | 0.8667 |
| B+ | 25 | 0.0089 | 0.4400 |
| B- | 5 | 0.0422 | 0.4000 |
| AB+ | 23 | 0.0021 | 0.0435 |
| AB- | 0 | n/a | n/a |
| 48h | 1 | days_cover_rule | 0.0038 | 0.00170 | 0.0089 | 0.0216 | 112.80 |
| 48h | 1 | logistic_regression | 0.0097 | 0.00170 | 0.0089 | 0.0216 | 112.80 |
| 48h | 1 | lightgbm | 0.0121 | 0.00169 | 0.0177 | 0.0433 | 56.40 |
| 48h | 3 | days_cover_rule | 0.0038 | 0.00170 | 0.0077 | 0.0563 | 130.15 |
| 48h | 3 | logistic_regression | 0.0097 | 0.00170 | 0.0142 | 0.1039 | 70.50 |
| 48h | 3 | lightgbm | 0.0121 | 0.00169 | 0.0171 | 0.1255 | 58.34 |
| 48h | 5 | days_cover_rule | 0.0038 | 0.00170 | 0.0064 | 0.0779 | 156.67 |
| 48h | 5 | logistic_regression | 0.0097 | 0.00170 | 0.0135 | 0.1645 | 74.21 |
| 48h | 5 | lightgbm | 0.0121 | 0.00169 | 0.0163 | 0.1991 | 61.30 |
| 48h | 10 | days_cover_rule | 0.0038 | 0.00170 | 0.0057 | 0.1385 | 176.25 |
| 48h | 10 | logistic_regression | 0.0097 | 0.00170 | 0.0128 | 0.3117 | 78.33 |
| 48h | 10 | lightgbm | 0.0121 | 0.00169 | 0.0144 | 0.3506 | 69.63 |

48h LightGBM PR-AUC: 0.0121 (95% CI [0.0099, 0.0176]).

| Budget | LightGBM recall (95% CI) | Days-of-cover recall | Recall difference (LightGBM - baseline, paired 95% CI) |
|---:|---:|---:|---:|
| 1 | 0.0433 ([0.0187, 0.0709]) | 0.0216 | +0.0216 ([-0.0073, +0.0609]) |
| 3 | 0.1255 ([0.0806, 0.1661]) | 0.0563 | +0.0693 ([+0.0249, +0.1217]) |
| 5 | 0.1991 ([0.1514, 0.2478]) | 0.0779 | +0.1212 ([+0.0669, +0.1715]) |
| 10 | 0.3506 ([0.2845, 0.4019]) | 0.1385 | +0.2121 ([+0.1357, +0.2693]) |

48h LightGBM calibration curve (held-out test rows):

| Rows | Mean predicted risk | Observed positive rate |
|---:|---:|---:|
| 135,360 | 0.00206 | 0.00171 |

48h LightGBM breakdown at 10 alerts per seed-day:

| Blood group | Positives | PR-AUC | Recall |
|---|---:|---:|---:|
| O+ | 2 | 0.0058 | 0.0000 |
| O- | 64 | 0.0151 | 0.1562 |
| A+ | 29 | 0.0122 | 0.6552 |
| A- | 30 | 0.0245 | 0.8667 |
| B+ | 50 | 0.0205 | 0.4000 |
| B- | 10 | 0.0150 | 0.2000 |
| AB+ | 46 | 0.0061 | 0.0870 |
| AB- | 0 | n/a | n/a |

## Rolling-origin evaluation

Expanding-window LightGBM folds are evaluated chronologically within the training-seed cohort; the outer validation and test seed cohorts remain untouched. These are simulated-data diagnostics, not the primary held-out test metrics.

| Fold | Train end (day) | Evaluation window | Rows | Positives | PR-AUC | Recall at budget | Precision at budget |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 255 | 258-328 | 201,600 | 196 | 0.0067 | 0.3367 | 0.0079 |
| 2 | 328 | 331-423 | 264,960 | 257 | 0.0103 | 0.4047 | 0.0094 |

## Leakage checks

- Labels use only observed unmet demand in rows t+1 through t+h; current-row unmet demand is not a feature. Label-shuffle 24h PR-AUC: `0.0008` versus test prevalence `0.0009`.
- Shuffled-label collapse within 0.02 AP of prevalence: **yes**.
- Target/observed unmet-demand columns in the feature list: `[]`.
- Univariate held-out proxy screen (AP >= 0.90): `[]`; overall screen: **PASS**.
- Median-replacement ablation was measured for every feature on held-out test rows; the detailed deltas are in `metrics.json`. It is an influence diagnostic, not causal evidence.
- 24h LightGBM recall at 1 alerts/day does not beat the days-of-cover baseline.
- 24h LightGBM recall at 3 alerts/day beats the days-of-cover baseline.
- 24h LightGBM recall at 5 alerts/day beats the days-of-cover baseline.
- 24h LightGBM recall at 10 alerts/day beats the days-of-cover baseline.
- 48h LightGBM recall at 1 alerts/day beats the days-of-cover baseline.
- 48h LightGBM recall at 3 alerts/day beats the days-of-cover baseline.
- 48h LightGBM recall at 5 alerts/day beats the days-of-cover baseline.
- 48h LightGBM recall at 10 alerts/day beats the days-of-cover baseline.

## Limitations

The simulations share one policy and demand generator; results may reflect simulator artifacts. Seed-held-out evaluation tests robustness across simulator randomness, not generalization to real hospitals. Daily timestamps support 24h and 48h labels but do not provide intraday warning timing, so no warning lead-time metric is reported. Alert budgets are evaluation scenarios, not operational alert policies. No real-world performance claim is supported.

## Artifacts

[`../models/phase2_model.joblib`](../models/phase2_model.joblib) stores LightGBM estimators, Platt calibrators, feature names, labels and validation cutoffs by alert budget. [`metrics.json`](./metrics.json) contains measured counts, metrics, bootstrap intervals and leakage checks.
