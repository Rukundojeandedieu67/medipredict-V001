"""Train and evaluate the pooled, seed-held-out Phase 2 early-warning models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medipredict.features import FEATURE_COLUMNS, build_features
from medipredict.modeling import (
    bootstrap_recall_intervals,
    bootstrap_metric_intervals,
    paired_bootstrap_recall_difference,
    partition_seed_directories,
    time_partition_masks,
)
from medipredict.simulator import DEFAULT_OUTPUT, DEFAULT_REFERENCE, GROUPS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "docs" / "metrics.json"
DEFAULT_MODEL = ROOT / "models" / "phase2_model.joblib"
DEFAULT_MODEL_CARD = ROOT / "docs" / "MODEL_CARD.md"
ALERT_BUDGETS = (1, 3, 5, 10)
BOOTSTRAP_REPLICATES = 200


def _reference_metadata(reference_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    districts = pd.read_csv(reference_dir / "dim_district.csv")
    admissions = districts.set_index("district")["hospital_admissions"].astype(float)
    admissions_weights = (admissions / admissions.sum()).to_dict()
    collection_rows = pd.read_csv(reference_dir / "rbc_2024_collected_by_group_rcbt.csv")
    totals = collection_rows.groupby("blood_group")["units_collected"].sum()
    total = float(totals.sum())
    group_shares = {group: float(totals[group] / total) for group in GROUPS}
    return admissions_weights, group_shares


def _load_examples(
    generated_dir: Path, reference_dir: Path, gap_days: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    seed_dirs = sorted(path for path in generated_dir.glob("seed_*") if path.is_dir())
    partitions = partition_seed_directories(path.name for path in seed_dirs)
    memberships = {
        "train": set(partitions.train),
        "validation": set(partitions.validation),
        "test": set(partitions.test),
    }
    admissions_weights, group_shares = _reference_metadata(reference_dir)
    accumulators: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "x": [],
            "y_24h": [],
            "y_48h": [],
            "clusters": [],
            "groups": [],
            "period_indices": [],
        }
        for name in ("train", "validation", "test")
    }
    seed_periods: dict[str, int] = {}
    source_seeds: dict[str, list[str]] = {name: [] for name in accumulators}
    for seed_dir in seed_dirs:
        summary = json.loads((seed_dir / "simulation_summary.json").read_text(encoding="utf-8"))
        if summary["frequency"] not in {"daily", "hourly"}:
            raise ValueError(f"Unsupported simulation frequency in {seed_dir}")
        periods_per_day = 1 if summary["frequency"] == "daily" else 24
        if summary["frequency"] == "hourly":
            raise ValueError("Phase 2 seed generation currently requires daily readings")
        readings_path = seed_dir / summary["stock_readings_file"]
        readings = pd.read_csv(readings_path)
        feature_frame = build_features(
            readings,
            admissions_weights,
            group_shares,
            periods_per_day=periods_per_day,
        )
        period_count = int(summary["period_days"]) * periods_per_day
        seed_periods[seed_dir.name] = period_count
        period_indices = feature_frame["_period_index"].to_numpy(dtype=np.int32)
        time_masks = time_partition_masks(
            period_indices, period_count, periods_per_day, gap_days
        )
        destination = next(name for name, names in memberships.items() if seed_dir.name in names)
        if seed_dir.name not in source_seeds[destination]:
            source_seeds[destination].append(seed_dir.name)
        selection = {
            "train": time_masks[0],
            "validation": time_masks[1],
            "test": time_masks[2],
        }[destination]
        selected = feature_frame.loc[selection]
        if selected.empty:
            raise ValueError(
                f"No {destination} rows for {seed_dir.name}; increase period_days or reduce gaps"
            )
        accumulator = accumulators[destination]
        accumulator["x"].append(selected.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        accumulator["y_24h"].append(selected["y_24h"].to_numpy(dtype=np.int8))
        accumulator["y_48h"].append(selected["y_48h"].to_numpy(dtype=np.int8))
        accumulator["clusters"].append(
            (
                str(summary["seed"])
                + ":"
                + selected["timestamp"].dt.strftime("%Y-%m-%d")
            ).to_numpy(dtype=str)
        )
        accumulator["groups"].append(selected["blood_group"].to_numpy(dtype=str))
        accumulator["period_indices"].append(
            selected["_period_index"].to_numpy(dtype=np.int32)
        )

    datasets: dict[str, dict[str, Any]] = {}
    positive_counts: dict[str, dict[str, int]] = {}
    for split, values in accumulators.items():
        datasets[split] = {
            "x": np.concatenate(values["x"]),
            "y_24h": np.concatenate(values["y_24h"]),
            "y_48h": np.concatenate(values["y_48h"]),
            "clusters": np.concatenate(values["clusters"]),
            "groups": np.concatenate(values["groups"]),
            "period_indices": np.concatenate(values["period_indices"]),
        }
        positive_counts[split] = {
            "rows": int(len(datasets[split]["x"])),
            "y_24h": int(datasets[split]["y_24h"].sum()),
            "y_48h": int(datasets[split]["y_48h"].sum()),
            "prevalence_24h": float(datasets[split]["y_24h"].mean()),
            "prevalence_48h": float(datasets[split]["y_48h"].mean()),
            "by_blood_group": {
                group: {
                    "y_24h": int(
                        datasets[split]["y_24h"][datasets[split]["groups"] == group].sum()
                    ),
                    "y_48h": int(
                        datasets[split]["y_48h"][datasets[split]["groups"] == group].sum()
                    ),
                }
                for group in GROUPS
            },
        }
    return datasets, source_seeds, {
        "partitions": partitions,
        "source_seeds": source_seeds,
        "positive_counts": positive_counts,
        "seed_periods": seed_periods,
        "group_shares": group_shares,
    }


def _lightgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=240,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _platt_calibrator(validation_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    if len(np.unique(labels)) != 2:
        raise ValueError("Validation labels must contain both classes to calibrate probabilities")
    logits = np.log(np.clip(validation_scores, 1e-6, 1 - 1e-6) / np.clip(1 - validation_scores, 1e-6, 1))
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logits.reshape(-1, 1), labels)
    return calibrator


def _calibrated_scores(
    classifier: Any, calibrator: LogisticRegression, features: np.ndarray
) -> np.ndarray:
    scores = classifier.predict_proba(features)[:, 1]
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def _budget_cutoff(scores: np.ndarray, clusters: np.ndarray, budget: int) -> float:
    chosen = []
    for cluster in np.unique(clusters):
        cluster_scores = scores[clusters == cluster]
        count = min(budget, len(cluster_scores))
        chosen.extend(np.sort(cluster_scores)[-count:].tolist())
    return float(min(chosen))


def _selected_alert_indices(
    scores: np.ndarray, clusters: np.ndarray, budget: int
) -> np.ndarray:
    selected = []
    for cluster in np.unique(clusters):
        indices = np.flatnonzero(clusters == cluster)
        selected.extend(
            indices[np.argsort(-scores[indices], kind="stable")[: min(budget, len(indices))]]
        )
    return np.asarray(selected, dtype=np.int64)


def _calibration_bins(
    labels: np.ndarray, scores: np.ndarray, bin_count: int = 10
) -> list[dict[str, float | int]]:
    bins = []
    boundaries = np.linspace(0, 1, bin_count + 1)
    for index in range(bin_count):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (scores >= lower) & (
            scores <= upper if index == bin_count - 1 else scores < upper
        )
        if np.any(mask):
            bins.append(
                {
                    "count": int(mask.sum()),
                    "mean_predicted_probability": float(scores[mask].mean()),
                    "observed_positive_rate": float(labels[mask].mean()),
                }
            )
    return bins


def _fit_horizon(
    name: str,
    label_key: str,
    datasets: dict[str, dict[str, Any]],
    bootstrap_seed: int,
    saved_lightgbm: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train, validation, test = (datasets[split] for split in ("train", "validation", "test"))
    y_train, y_validation, y_test = (
        train[label_key],
        validation[label_key],
        test[label_key],
    )
    if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
        raise ValueError(f"{name}: training and validation splits each need positive and negative labels")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=bootstrap_seed),
    )
    logistic.fit(train["x"], y_train)
    logistic_calibrator = _platt_calibrator(
        logistic.predict_proba(validation["x"])[:, 1], y_validation
    )
    logistic_validation_scores = _calibrated_scores(
        logistic, logistic_calibrator, validation["x"]
    )
    logistic_scores = _calibrated_scores(logistic, logistic_calibrator, test["x"])

    if saved_lightgbm is None:
        lightgbm = _lightgbm(bootstrap_seed)
        lightgbm.fit(train["x"], y_train)
        lgbm_calibrator = _platt_calibrator(
            lightgbm.predict_proba(validation["x"])[:, 1], y_validation
        )
    else:
        if (
            saved_lightgbm.get("label") != label_key
            or saved_lightgbm.get("feature_columns") != list(FEATURE_COLUMNS)
        ):
            raise ValueError(f"Saved model for {name} does not match the current feature/label schema")
        lightgbm = saved_lightgbm["classifier"]
        lgbm_calibrator = saved_lightgbm["calibrator"]
    lgbm_validation_scores = _calibrated_scores(
        lightgbm, lgbm_calibrator, validation["x"]
    )
    lgbm_scores = _calibrated_scores(lightgbm, lgbm_calibrator, test["x"])
    validation_rule_scores = -validation["x"][:, FEATURE_COLUMNS.index("days_of_cover")]
    rule_calibrator = LogisticRegression(max_iter=1000)
    rule_calibrator.fit(validation_rule_scores.reshape(-1, 1), y_validation)
    validation_rule_probabilities = rule_calibrator.predict_proba(
        validation_rule_scores.reshape(-1, 1)
    )[:, 1]
    rule_scores = rule_calibrator.predict_proba(
        (-test["x"][:, FEATURE_COLUMNS.index("days_of_cover")]).reshape(-1, 1)
    )[:, 1]
    predictions = {
        "days_cover_rule": rule_scores,
        "logistic_regression": logistic_scores,
        "lightgbm": lgbm_scores,
    }
    validation_predictions = {
        "days_cover_rule": validation_rule_probabilities,
        "logistic_regression": logistic_validation_scores,
        "lightgbm": lgbm_validation_scores,
    }
    results = {}
    for model_name, scores in predictions.items():
        model_result = {
            "pr_auc": float(average_precision_score(y_test, scores)),
            "brier_score": float(brier_score_loss(y_test, scores)),
            "calibration_curve": _calibration_bins(y_test, scores),
            "by_alert_budget": {},
            "validation_cutoffs_by_budget": {},
        }
        for budget in ALERT_BUDGETS:
            selected = _selected_alert_indices(scores, test["clusters"], budget)
            true_positives = int(y_test[selected].sum())
            model_result["by_alert_budget"][str(budget)] = {
                "alerts_per_seed_day": budget,
                "alert_count": int(len(selected)),
                "true_positive_alerts": true_positives,
                "recall": float(true_positives / y_test.sum()) if y_test.sum() else 0.0,
                "precision": float(true_positives / len(selected)) if len(selected) else 0.0,
                "alerts_per_true_event": (
                    float(len(selected) / true_positives) if true_positives else None
                ),
            }
            model_result["validation_cutoffs_by_budget"][str(budget)] = _budget_cutoff(
                validation_predictions[model_name], validation["clusters"], budget
            )
        results[model_name] = model_result
    intervals = bootstrap_metric_intervals(
        y_test, lgbm_scores, test["clusters"], max(ALERT_BUDGETS),
        replicates=BOOTSTRAP_REPLICATES, seed=bootstrap_seed,
    )
    recall_intervals = bootstrap_recall_intervals(
        y_test,
        lgbm_scores,
        test["clusters"],
        budgets=ALERT_BUDGETS,
        replicates=BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )
    paired_recall_difference_ci = paired_bootstrap_recall_difference(
        y_test,
        lgbm_scores,
        rule_scores,
        test["clusters"],
        budgets=ALERT_BUDGETS,
        replicates=BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )
    per_group = {}
    selected_lgbm_alerts = _selected_alert_indices(
        lgbm_scores, test["clusters"], max(ALERT_BUDGETS)
    )
    for group in GROUPS:
        mask = test["groups"] == group
        group_labels = y_test[mask]
        alert_mask = np.zeros(len(y_test), dtype=bool)
        alert_mask[selected_lgbm_alerts] = True
        group_alerts = alert_mask[mask]
        per_group[group] = {
            "positive_count": int(group_labels.sum()),
            "pr_auc": (
                float(average_precision_score(group_labels, lgbm_scores[mask]))
                if group_labels.sum() > 0
                else None
            ),
            "recall_at_10_alerts_per_seed_day": (
                float(group_labels[group_alerts].sum() / group_labels.sum())
                if group_labels.sum() > 0
                else None
            ),
        }

    output = {
        "models": results,
        "lightgbm_ci_95": {
            "pr_auc": intervals["pr_auc"],
            "recall_by_alert_budget": recall_intervals,
        },
        "paired_recall_difference_ci_95": paired_recall_difference_ci,
        "test_positive_count": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "lightgbm_per_blood_group": per_group,
        "lightgbm_beats_days_cover_rule_by_budget": {
            str(budget): (
                results["lightgbm"]["by_alert_budget"][str(budget)]["recall"]
                > results["days_cover_rule"]["by_alert_budget"][str(budget)]["recall"]
            )
            for budget in ALERT_BUDGETS
        },
    }
    artifact = {
        "classifier": lightgbm,
        "calibrator": lgbm_calibrator,
        "feature_columns": list(FEATURE_COLUMNS),
        "label": label_key,
        "alert_budgets_per_seed_day": list(ALERT_BUDGETS),
        "validation_cutoffs_by_budget": results["lightgbm"][
            "validation_cutoffs_by_budget"
        ],
    }
    return output, artifact


def _leakage_diagnostics(
    datasets: dict[str, dict[str, Any]],
    feature_names: tuple[str, ...],
    fitted_lightgbm: LGBMClassifier | None = None,
) -> dict[str, Any]:
    train, test = datasets["train"], datasets["test"]
    labels_train = train["y_24h"].copy()
    rng = np.random.default_rng(2026)
    rng.shuffle(labels_train)
    shuffled = _lightgbm(2026)
    shuffled.set_params(n_estimators=120)
    shuffled.fit(train["x"], labels_train)
    shuffled_scores = shuffled.predict_proba(test["x"])[:, 1]
    shuffled_pr_auc = float(average_precision_score(test["y_24h"], shuffled_scores))

    main_model = (
        fitted_lightgbm if fitted_lightgbm is not None else _lightgbm(2026)
    )
    if fitted_lightgbm is None:
        main_model.fit(train["x"], train["y_24h"])
    full_scores = main_model.predict_proba(test["x"])[:, 1]
    full_pr_auc = float(average_precision_score(test["y_24h"], full_scores))
    feature_drops = []
    univariate_scores = []
    train_medians = np.median(train["x"], axis=0)
    for index, feature in enumerate(feature_names):
        ablated = test["x"].copy()
        ablated[:, index] = train_medians[index]
        score = float(
            average_precision_score(
                test["y_24h"], main_model.predict_proba(ablated)[:, 1]
            )
        )
        feature_drops.append(
            {
                "feature": feature,
                "pr_auc_after_median_ablation": score,
                "delta_vs_full_pr_auc": score - full_pr_auc,
            }
        )
        values = test["x"][:, index]
        univariate_scores.append(
            {
                "feature": feature,
                "best_univariate_pr_auc": float(
                    max(
                        average_precision_score(test["y_24h"], values),
                        average_precision_score(test["y_24h"], -values),
                    )
                ),
            }
        )
    feature_drops.sort(key=lambda row: abs(row["delta_vs_full_pr_auc"]), reverse=True)
    univariate_scores.sort(key=lambda row: row["best_univariate_pr_auc"], reverse=True)
    target_features = sorted(
        set(feature_names).intersection({"unmet_demand_units", "y_24h", "y_48h"})
    )
    proxy_suspects = [
        row["feature"]
        for row in univariate_scores
        if row["best_univariate_pr_auc"] >= 0.9
    ]
    return {
        "shuffled_label_pr_auc": shuffled_pr_auc,
        "unshuffled_lightgbm_pr_auc": full_pr_auc,
        "test_prevalence": float(test["y_24h"].mean()),
        "shuffle_collapsed_to_prevalence_band": abs(shuffled_pr_auc - test["y_24h"].mean()) < 0.02,
        "features_in_target": target_features,
        "univariate_proxy_screen": univariate_scores,
        "potential_label_proxy_features": proxy_suspects,
        "proxy_screen_passed": not target_features and not proxy_suspects,
        "median_replacement_feature_ablations": feature_drops,
    }


def _rolling_origin_evaluation(
    train: dict[str, Any], period_count: int, periods_per_day: int = 1
) -> list[dict[str, Any]]:
    windows = ((0.35, 0.45), (0.45, 0.58))
    results = []
    indices = train["period_indices"]
    labels = train["y_24h"]
    for fold, (origin_fraction, end_fraction) in enumerate(windows, start=1):
        origin = int(period_count * origin_fraction)
        evaluation_start = origin + 3 * periods_per_day
        evaluation_end = int(period_count * end_fraction)
        fit_mask = indices < origin
        evaluation_mask = (indices >= evaluation_start) & (indices < evaluation_end)
        if len(np.unique(labels[fit_mask])) != 2 or not np.any(evaluation_mask):
            raise ValueError(f"Rolling-origin fold {fold} has insufficient data or class balance")
        classifier = _lightgbm(2026 + fold)
        classifier.set_params(n_estimators=120)
        classifier.fit(train["x"][fit_mask], labels[fit_mask])
        fold_labels = labels[evaluation_mask]
        scores = classifier.predict_proba(train["x"][evaluation_mask])[:, 1]
        clusters = train["clusters"][evaluation_mask]
        selected = _selected_alert_indices(scores, clusters, max(ALERT_BUDGETS))
        results.append(
            {
                "fold": fold,
                "training_end_day_exclusive": origin,
                "evaluation_start_day": evaluation_start,
                "evaluation_end_day_exclusive": evaluation_end,
                "rows": int(evaluation_mask.sum()),
                "positive_count": int(fold_labels.sum()),
                "pr_auc": float(average_precision_score(fold_labels, scores)),
                "recall_at_alert_budget": float(
                    fold_labels[selected].sum() / fold_labels.sum()
                    if fold_labels.sum()
                    else 0.0
                ),
                "precision_at_alert_budget": float(
                    fold_labels[selected].sum() / len(selected) if len(selected) else 0.0
                ),
            }
        )
    return results


def _json_default(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _model_card(metrics: dict[str, Any]) -> str:
    split = metrics["positive_counts"]
    lines = [
        "# Phase 2 model card",
        "",
        "## Intended use",
        "",
        "Prototype early warning of simulated unmet blood demand for decision support. It is not validated on operational Rwanda stock readings and must not drive automatic dispatch.",
        "",
        "## Data and split",
        "",
        "All model observations are `simulated`. A pooled model uses blood group as a one-hot feature. Seed directories are separated 60%/20%/20% into train/validation/test, and each seed uses chronological 60%/20%/20% windows with a 3-day purge between windows. Validation data alone calibrates probabilities using Platt scaling.",
        "",
        f"- Training seeds: {', '.join(metrics['seed_splits']['train'])}",
        f"- Validation seeds: {', '.join(metrics['seed_splits']['validation'])}",
        f"- Test seeds: {', '.join(metrics['seed_splits']['test'])}",
        "",
        "| Split | Rows | 24h positives | 48h positives |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "validation", "test"):
        count = split[name]
        lines.append(
            f"| {name} | {count['rows']:,} | {count['y_24h']:,} | {count['y_48h']:,} |"
        )
    lines.extend(
        [
            "",
            "## Held-out results",
            "",
            f"Metrics are measured on the held-out simulated test seeds only. PR-AUC is average precision; bootstrap intervals use {metrics['bootstrap']['replicates']} resamples of seed-day clusters. Alert budgets are per seed-day, where a seed-day contains 30 district nodes × 8 blood groups. The days-of-cover baseline ranks the lowest-cover cells under the same budget.",
            "Alerts per true event is the selected alert count divided by true-positive alerts on positive-label node/group-days (labels may overlap in the 48-hour horizon); it is not a count of deduplicated clinical episodes.",
            "",
            "| Horizon | Budget | Model | PR-AUC | Brier | Precision | Recall | Alerts / true event |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, result in metrics["horizons"].items():
        for budget in ALERT_BUDGETS:
            for model_name, values in result["models"].items():
                metric = values["by_alert_budget"][str(budget)]
                alerts_per_event = metric["alerts_per_true_event"]
                lines.append(
                    f"| {horizon} | {budget} | {model_name} | "
                    f"{values['pr_auc']:.4f} | {values['brier_score']:.5f} | "
                    f"{metric['precision']:.4f} | {metric['recall']:.4f} | "
                    f"{alerts_per_event:.2f} |" if alerts_per_event is not None else
                    f"| {horizon} | {budget} | {model_name} | "
                    f"{values['pr_auc']:.4f} | {values['brier_score']:.5f} | "
                    f"{metric['precision']:.4f} | {metric['recall']:.4f} | n/a |"
                )
        ci = result["lightgbm_ci_95"]
        lines.extend(
            [
                "",
                f"{horizon} LightGBM PR-AUC: {result['models']['lightgbm']['pr_auc']:.4f} "
                f"(95% CI [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}]).",
                "",
                "| Budget | LightGBM recall (95% CI) | Days-of-cover recall | Recall difference (LightGBM - baseline, paired 95% CI) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for budget in ALERT_BUDGETS:
            key = str(budget)
            recall_ci = ci["recall_by_alert_budget"][key]
            difference_ci = result["paired_recall_difference_ci_95"][key]
            lgbm_recall = result["models"]["lightgbm"]["by_alert_budget"][key]["recall"]
            baseline_recall = result["models"]["days_cover_rule"]["by_alert_budget"][key]["recall"]
            lines.append(
                f"| {budget} | {lgbm_recall:.4f} "
                f"([{recall_ci[0]:.4f}, {recall_ci[1]:.4f}]) | "
                f"{baseline_recall:.4f} | "
                f"{lgbm_recall - baseline_recall:+.4f} "
                f"([{difference_ci[0]:+.4f}, {difference_ci[1]:+.4f}]) |"
            )
        lines.extend(
            [
                "",
                f"{horizon} LightGBM calibration curve (held-out test rows):",
                "",
                "| Rows | Mean predicted risk | Observed positive rate |",
                "|---:|---:|---:|",
            ]
        )
        for calibration_bin in result["models"]["lightgbm"]["calibration_curve"]:
            lines.append(
                f"| {calibration_bin['count']:,} | "
                f"{calibration_bin['mean_predicted_probability']:.5f} | "
                f"{calibration_bin['observed_positive_rate']:.5f} |"
            )
        lines.extend(
            [
                "",
                f"{horizon} LightGBM breakdown at {max(ALERT_BUDGETS)} alerts per seed-day:",
                "",
                "| Blood group | Positives | PR-AUC | Recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for group, values in result["lightgbm_per_blood_group"].items():
            group_pr_auc = (
                f"{values['pr_auc']:.4f}" if values["pr_auc"] is not None else "n/a"
            )
            group_recall = (
                f"{values['recall_at_10_alerts_per_seed_day']:.4f}"
                if values["recall_at_10_alerts_per_seed_day"] is not None
                else "n/a"
            )
            lines.append(
                f"| {group} | {values['positive_count']:,} | "
                f"{group_pr_auc} | {group_recall} |"
            )
    leakage = metrics["leakage_checks"]
    lines.extend(
        [
            "",
            "## Rolling-origin evaluation",
            "",
            "Expanding-window LightGBM folds are evaluated chronologically within the training-seed cohort; the outer validation and test seed cohorts remain untouched. These are simulated-data diagnostics, not the primary held-out test metrics.",
            "",
            "| Fold | Train end (day) | Evaluation window | Rows | Positives | PR-AUC | Recall at budget | Precision at budget |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in metrics["rolling_origin"]:
        lines.append(
            f"| {result['fold']} | {result['training_end_day_exclusive']} | "
            f"{result['evaluation_start_day']}-{result['evaluation_end_day_exclusive']} | "
            f"{result['rows']:,} | {result['positive_count']:,} | "
            f"{result['pr_auc']:.4f} | {result['recall_at_alert_budget']:.4f} | "
            f"{result['precision_at_alert_budget']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"- Labels use only observed unmet demand in rows t+1 through t+h; current-row unmet demand is not a feature. Label-shuffle 24h PR-AUC: `{leakage['shuffled_label_pr_auc']:.4f}` versus test prevalence `{leakage['test_prevalence']:.4f}`.",
            f"- Shuffled-label collapse within 0.02 AP of prevalence: **{'yes' if leakage['shuffle_collapsed_to_prevalence_band'] else 'no'}**.",
            f"- Target/observed unmet-demand columns in the feature list: `{leakage['features_in_target']}`.",
            f"- Univariate held-out proxy screen (AP >= 0.90): `{leakage['potential_label_proxy_features']}`; overall screen: **{'PASS' if leakage['proxy_screen_passed'] else 'REVIEW'}**.",
            "- Median-replacement ablation was measured for every feature on held-out test rows; the detailed deltas are in `metrics.json`. It is an influence diagnostic, not causal evidence.",
            *[
                f"- {horizon} LightGBM recall at {budget} alerts/day "
                f"{'beats' if result['lightgbm_beats_days_cover_rule_by_budget'][str(budget)] else 'does not beat'} "
                "the days-of-cover baseline."
                for horizon, result in metrics["horizons"].items()
                for budget in ALERT_BUDGETS
            ],
            "",
            "## Limitations",
            "",
            "The simulations share one policy and demand generator; results may reflect simulator artifacts. Seed-held-out evaluation tests robustness across simulator randomness, not generalization to real hospitals. Daily timestamps support 24h and 48h labels but do not provide intraday warning timing, so no warning lead-time metric is reported. Alert budgets are evaluation scenarios, not operational alert policies. No real-world performance claim is supported.",
            "",
            "## Artifacts",
            "",
            "[`../models/phase2_model.joblib`](../models/phase2_model.joblib) stores LightGBM estimators, Platt calibrators, feature names, labels and validation cutoffs by alert budget. [`metrics.json`](./metrics.json) contains measured counts, metrics, bootstrap intervals and leakage checks.",
            "",
        ]
    )
    return "\n".join(lines)


def train(
    generated_dir: Path,
    reference_dir: Path,
    gap_days: int,
    metrics_path: Path,
    model_path: Path,
    model_card_path: Path,
    reuse_model: bool = False,
) -> dict[str, Any]:
    datasets, source_seeds, meta = _load_examples(generated_dir, reference_dir, gap_days)
    existing_model = None
    if reuse_model:
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing saved model: {model_path}")
        existing_model = joblib.load(model_path)
    horizons = {}
    artifacts = {}
    for horizon, label in (("24h", "y_24h"), ("48h", "y_48h")):
        horizons[horizon], artifacts[horizon] = _fit_horizon(
            horizon,
            label,
            datasets,
            bootstrap_seed=2026,
            saved_lightgbm=(
                existing_model["models"][horizon] if existing_model is not None else None
            ),
        )
    leakage = _leakage_diagnostics(
        datasets, FEATURE_COLUMNS, artifacts["24h"]["classifier"]
    )
    period_count = min(meta["seed_periods"].values())
    metrics = {
        "data_source": "simulated",
        "frequency": "daily",
        "seed_splits": source_seeds,
        "time_split_gap_days": gap_days,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "seed-day cluster",
            "confidence_level": 0.95,
        },
        "positive_counts": meta["positive_counts"],
        "alert_budgets_per_seed_day": list(ALERT_BUDGETS),
        "horizons": horizons,
        "rolling_origin": _rolling_origin_evaluation(
            datasets["train"], period_count
        ),
        "leakage_checks": leakage,
        "feature_columns": list(FEATURE_COLUMNS),
        "test_group_shares_reference": meta["group_shares"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": artifacts,
            "feature_columns": list(FEATURE_COLUMNS),
            "seed_splits": source_seeds,
            "time_split_gap_days": gap_days,
            "data_source": "simulated",
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text(_model_card(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Reuse the saved LightGBM estimators and calibrators instead of refitting them.",
    )
    args = parser.parse_args()
    metrics = train(
        args.generated_dir,
        args.reference_dir,
        args.gap_days,
        args.metrics,
        args.model,
        args.model_card,
        args.reuse_model,
    )
    print(json.dumps(metrics, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
