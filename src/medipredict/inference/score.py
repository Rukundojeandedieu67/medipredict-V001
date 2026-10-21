"""Rank end-of-day district/group observations with the frozen Phase 2 models."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from medipredict.features import FEATURE_COLUMNS, build_features
from medipredict.inference.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from medipredict.simulator import GROUPS


ALLOWED_SOURCES = {"real", "derived", "simulated", "entered"}
LOW_CONFIDENCE_GROUPS = {"A-", "B-", "AB-"}
REQUIRED_INPUT_COLUMNS = {"timestamp", "district", "blood_group", "stock_units"}
REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"


@lru_cache(maxsize=1)
def _load_active_bundle() -> dict[str, Any]:
    return load_bundle(DEFAULT_BUNDLE_DIR)


@lru_cache(maxsize=1)
def _reference_metadata() -> tuple[dict[str, float], dict[str, float]]:
    districts = pd.read_csv(REFERENCE_DIR / "dim_district.csv")
    admissions = districts.set_index("district")["hospital_admissions"].astype(float)
    admissions_weights = (admissions / admissions.sum()).to_dict()
    collection_rows = pd.read_csv(
        REFERENCE_DIR / "rbc_2024_collected_by_group_rcbt.csv"
    )
    totals = collection_rows.groupby("blood_group")["units_collected"].sum()
    total = float(totals.sum())
    group_shares = {group: float(totals[group] / total) for group in GROUPS}
    return admissions_weights, group_shares


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _prepare_readings(
    readings: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    missing = REQUIRED_INPUT_COLUMNS.difference(readings.columns)
    if missing:
        raise ValueError(f"Missing required inference columns: {sorted(missing)}")
    frame = readings.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.loc[frame["timestamp"] <= as_of].copy()
    if frame.empty:
        raise ValueError("No readings are available at or before as_of")
    frame["district"] = frame["district"].astype(str).str.strip()
    frame["blood_group"] = frame["blood_group"].astype(str).str.strip().str.upper()
    unknown_groups = sorted(set(frame["blood_group"]) - set(GROUPS))
    if unknown_groups:
        raise ValueError(f"Unsupported blood groups: {unknown_groups}")
    admissions_weights, _ = _reference_metadata()
    unknown_districts = sorted(set(frame["district"]) - set(admissions_weights))
    if unknown_districts:
        raise ValueError(f"Unknown district names: {unknown_districts}")

    frame["node_id"] = "district:" + frame["district"]
    numeric_columns = (
        "stock_units",
        "demand_units",
        "supply_received",
        "expired_units",
        "lead_time_hours",
        "expiring_units_7d",
        "recent_surge_flag",
    )
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["stock_units"].isna().any() or (frame["stock_units"] < 0).any():
        raise ValueError("stock_units must contain non-negative numeric values")
    for column in ("demand_units", "supply_received", "expired_units"):
        if column in frame and (frame[column].dropna() < 0).any():
            raise ValueError(f"{column} must not contain negative units")
    missing_supply = (
        "supply_received" not in frame or frame["supply_received"].isna().any()
    )
    missing_surge_flag = (
        "recent_surge_flag" not in frame or frame["recent_surge_flag"].isna().any()
    )
    demand_is_missing = (
        "demand_units" not in frame or frame["demand_units"].isna().all()
    )
    if "supply_received" not in frame:
        frame["supply_received"] = 0.0
    else:
        frame["supply_received"] = frame["supply_received"].fillna(0.0)
    if "expired_units" not in frame:
        frame["expired_units"] = 0.0
    else:
        frame["expired_units"] = frame["expired_units"].fillna(0.0)
    if "lead_time_hours" not in frame:
        frame["lead_time_hours"] = np.nan
    if "expiring_units_7d" not in frame:
        frame["expiring_units_7d"] = np.nan
    if "recent_surge_flag" not in frame:
        frame["recent_surge_flag"] = 0.0
    frame["recent_surge_flag"] = frame["recent_surge_flag"].fillna(0)
    demand_column_missing = "demand_units" not in readings.columns
    if demand_is_missing:
        frame["demand_units"] = np.nan

    if "data_source" not in frame:
        frame["data_source"] = "entered"
    else:
        frame["data_source"] = frame["data_source"].fillna("entered").astype(str)
    invalid_sources = sorted(set(frame["data_source"]) - ALLOWED_SOURCES)
    if invalid_sources:
        raise ValueError(f"Unsupported data_source values: {invalid_sources}")

    frame.sort_values(
        ["node_id", "blood_group", "timestamp"], inplace=True, kind="stable"
    )
    duplicate_mask = frame.duplicated(["node_id", "blood_group", "timestamp"], keep=False)
    if duplicate_mask.any():
        duplicate_row = int(np.flatnonzero(duplicate_mask.to_numpy())[0]) + 1
        raise ValueError(f"Duplicate node/group/timestamp reading at filtered row {duplicate_row}")
    frame["_calendar_day"] = frame["timestamp"].dt.floor("D")
    duplicate_day = frame.duplicated(
        ["node_id", "blood_group", "_calendar_day"], keep=False
    )
    if duplicate_day.any():
        duplicate_row = int(np.flatnonzero(duplicate_day.to_numpy())[0]) + 1
        raise ValueError(f"Multiple daily readings for one district/group at filtered row {duplicate_row}")

    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for key, series in frame.groupby(["node_id", "blood_group"], sort=True):
        key_tuple = (str(key[0]), str(key[1]))
        distinct_days = series["_calendar_day"].drop_duplicates().sort_values()
        calendar_span = int((distinct_days.iloc[-1] - distinct_days.iloc[0]).days) + 1
        history_gaps = len(distinct_days) < calendar_span
        estimated_demand = False
        insufficient_inputs = (
            series["demand_units"].isna().all()
            and (
                "supply_received" not in readings.columns
                or readings["supply_received"].isna().all()
            )
        )
        if insufficient_inputs:
            estimated_demand = False
        elif series["demand_units"].isna().all():
            estimated_demand = True
            previous_stock = series["stock_units"].shift(1)
            estimated = (
                previous_stock.fillna(series["stock_units"])
                + series["supply_received"]
                - series["expired_units"]
                - series["stock_units"]
            ).clip(lower=0)
            frame.loc[series.index, "demand_units"] = estimated
        else:
            missing_demand = series["demand_units"].isna()
            if missing_demand.any():
                estimated_demand = True
                previous_stock = series["stock_units"].shift(1)
                estimated = (
                    previous_stock.fillna(series["stock_units"])
                    + series["supply_received"]
                    - series["expired_units"]
                    - series["stock_units"]
                ).clip(lower=0)
                frame.loc[series.index[missing_demand], "demand_units"] = estimated.loc[
                    missing_demand
                ]
        metadata[key_tuple] = {
            "history_days": calendar_span,
            "history_gaps": history_gaps,
            "estimated_demand": estimated_demand,
            "insufficient_inputs": insufficient_inputs,
            "demand_column_missing": demand_column_missing,
            "missing_supply": (
                missing_supply
            ),
            "missing_lead_time": series["lead_time_hours"].isna().any(),
            "missing_expiry_window": series["expiring_units_7d"].isna().any(),
            "missing_surge_flag": (
                missing_surge_flag
            ),
            "data_source": str(series["data_source"].iloc[-1]),
        }

    frame["lead_time_hours"] = pd.to_numeric(
        frame["lead_time_hours"], errors="coerce"
    )
    frame["expiring_units_7d"] = pd.to_numeric(
        frame["expiring_units_7d"], errors="coerce"
    )
    return frame, metadata


def _feature_values(
    readings: pd.DataFrame,
    as_of: pd.Timestamp,
    reference_stats: dict[str, Any],
    admissions_weights: dict[str, float],
    group_shares: dict[str, float],
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    prepared, metadata = _prepare_readings(readings, as_of)
    features = build_features(
        prepared,
        admissions_weights,
        group_shares,
        periods_per_day=1,
        include_labels=False,
        drop_incomplete=False,
    )
    features["_rule_days_of_cover"] = features["days_of_cover"]
    medians = {
        name: float(reference_stats[name]["quantiles"]["50%"])
        for name in FEATURE_COLUMNS
    }
    # These fields have predictable neutral defaults when not supplied by the feed.
    for column in FEATURE_COLUMNS:
        if features[column].isna().any():
            features[column] = features[column].fillna(medians[column])
    for (node_id, group), _ in prepared.groupby(
        ["node_id", "blood_group"], sort=True
    ):
        key = (str(node_id), str(group))
        history = metadata[key]["history_days"]
        flags: list[str] = []
        if metadata[key]["demand_column_missing"]:
            flags.append("demand_column_estimated")
        if metadata[key]["estimated_demand"]:
            flags.append("demand_estimated")
        if metadata[key]["insufficient_inputs"]:
            flags.append("insufficient_inputs")
        if metadata[key]["missing_supply"]:
            flags.append("supply_missing_assumed_zero")
        if metadata[key]["missing_lead_time"]:
            flags.append("lead_time_defaulted")
        if metadata[key]["missing_expiry_window"]:
            flags.append("expiring_units_defaulted")
        if metadata[key]["missing_surge_flag"]:
            flags.append("surge_flag_defaulted")
        if metadata[key]["history_gaps"]:
            flags.append("missing_daily_observations")
        if history < 14:
            flags.append("insufficient_history")
        elif history < 28:
            flags.append("history_under_28_days")
        metadata[key]["quality_flags"] = flags
    return features, metadata


def _population_stability_index(
    values: np.ndarray, expected_counts: list[int], edges: list[float]
) -> float:
    expected = np.asarray(expected_counts, dtype=float)
    observed, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    observed = observed.astype(float)
    observed[0] += np.count_nonzero(values < edges[0])
    observed[-1] += np.count_nonzero(values > edges[-1])
    expected_share = np.clip(expected / max(expected.sum(), 1), 1e-6, None)
    observed_share = np.clip(observed / max(observed.sum(), 1), 1e-6, None)
    return float(np.sum((observed_share - expected_share) * np.log(observed_share / expected_share)))


def _driver_label(feature: str, contribution: float, value: float) -> str:
    direction = "increases" if contribution >= 0 else "reduces"
    labels = {
        "days_of_cover": (
            "low days of cover" if value < 2 else "days of cover"
        ),
        "stock_units": "current stock level",
        "demand_mean_7d": "recent demand level",
        "demand_mean_14d": "14-day demand level",
        "demand_trend_7d_vs_14d": "recent demand trend",
        "demand_volatility_7d": "demand variability",
        "supply_received_7d": "recent supply received",
        "time_since_delivery_days": "time since last delivery",
        "expected_lead_time_hours": "expected delivery lead time",
        "expiring_units_7d": "units expiring soon",
        "district_admissions_weight": "district admissions weight",
        "group_share": "blood-group demand share",
        "recent_surge_flag": "recent surge indicator",
    }
    if feature == "group_share":
        base = labels[feature]
    elif feature.startswith("group_"):
        base = f"{feature.removeprefix('group_')} blood group"
    else:
        base = labels.get(feature, feature.replace("_", " "))
    return f"{base} {direction} estimated risk"


def _calibrated_probabilities(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    classifier = model["classifier"]
    raw = classifier.predict_proba(matrix)[:, 1]
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return model["calibrator"].predict_proba(logits.reshape(-1, 1))[:, 1]


def _drivers(
    model: dict[str, Any], matrix: np.ndarray, feature_names: Sequence[str]
) -> list[list[str]]:
    contributions = model["classifier"].predict(matrix, pred_contrib=True)
    contribution_array = np.asarray(contributions)
    if contribution_array.ndim != 2 or contribution_array.shape[1] != len(feature_names) + 1:
        raise ValueError("Unexpected LightGBM pred_contrib output shape")
    drivers: list[list[str]] = []
    for row, feature_values in zip(contribution_array, matrix, strict=True):
        indices = np.argsort(-np.abs(row[:-1]), kind="stable")[:3]
        drivers.append(
            [
                _driver_label(
                    feature_names[index], float(row[index]), float(feature_values[index])
                )
                for index in indices
            ]
        )
    return drivers


def score(
    df: pd.DataFrame,
    as_of: str | datetime | pd.Timestamp | None = None,
    horizons: tuple[str, ...] = ("24h", "48h"),
    top_k: int = 5,
) -> dict[str, Any]:
    """Score canonical daily readings and return a deterministic ranked watch-list."""
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if not horizons or len(set(horizons)) != len(horizons) or not set(horizons) <= {"24h", "48h"}:
        raise ValueError("horizons must contain 24h and/or 48h without duplicates")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame")
    if "timestamp" not in df:
        raise ValueError("Missing required inference column: timestamp")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    resolved_as_of = _timestamp(as_of) if as_of is not None else timestamps.max()
    bundle = _load_active_bundle()
    admissions_weights, group_shares = _reference_metadata()
    feature_frame, metadata = _feature_values(
        df,
        resolved_as_of,
        bundle["reference_stats"],
        admissions_weights,
        group_shares,
    )
    feature_frame = feature_frame.sort_values(
        ["node_id", "blood_group", "timestamp"], kind="stable"
    )
    feature_names = list(bundle["feature_schema"]["feature_order"])
    matrix = feature_frame.loc[:, feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Inference feature construction produced non-finite values")
    latest_indices = feature_frame.groupby(
        ["node_id", "blood_group"], sort=True
    ).tail(1).index
    latest = feature_frame.loc[latest_indices].copy()
    results_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    drift_flags: set[str] = set()
    psi_values: dict[str, float] = {}
    stats = bundle["reference_stats"]
    for feature in feature_names:
        values = latest[feature].to_numpy(dtype=float)
        stats_for_feature = stats[feature]
        psi = _population_stability_index(
            values, stats_for_feature["histogram"]["counts"],
            stats_for_feature["histogram"]["edges"],
        )
        psi_values[feature] = psi
        if (
            feature in stats["top_10_features"] and psi > 0.25
        ) or np.any(
            (values < stats_for_feature["minimum"])
            | (values > stats_for_feature["maximum"])
        ):
            drift_flags.add("distribution_shift")

    index_by_key = {
        (str(row.node_id), str(row.blood_group)): int(index)
        for index, row in latest.iterrows()
    }
    eligible_mask = np.array(
        [
            (
                metadata[(str(row.node_id), str(row.blood_group))]["history_days"] >= 14
                and not metadata[(str(row.node_id), str(row.blood_group))][
                    "insufficient_inputs"
                ]
            )
            for row in feature_frame.itertuples()
        ],
        dtype=bool,
    )
    for horizon in horizons:
        model = bundle["models"][horizon]
        probabilities = np.full(len(feature_frame), np.nan, dtype=float)
        if eligible_mask.any():
            probabilities[eligible_mask] = _calibrated_probabilities(
                model, matrix[eligible_mask]
            )
        for key, row_index in index_by_key.items():
            probability = probabilities[row_index]
            record = results_by_key.setdefault(
                key,
                {
                    "district": str(latest.loc[row_index, "district"]),
                    "blood_group": key[1],
                    "risk_24h": None,
                    "risk_48h": None,
                },
            )
            record[f"risk_{horizon}"] = (
                round(float(probability), 8) if np.isfinite(probability) else None
            )
    primary_horizon = "24h" if "24h" in horizons else horizons[0]
    primary_model = bundle["models"][primary_horizon]
    driver_keys = [
        key for key, row_index in index_by_key.items() if eligible_mask[row_index]
    ]
    if driver_keys:
        driver_rows = np.asarray(
            [index_by_key[key] for key in driver_keys], dtype=np.int64
        )
        driver_values = _drivers(primary_model, matrix[driver_rows], feature_names)
        for key, values in zip(driver_keys, driver_values, strict=True):
            results_by_key[key]["top_drivers"] = values
    for key in results_by_key:
        results_by_key[key].setdefault("top_drivers", [])

    for key, record in results_by_key.items():
        detail = metadata[key]
        rule_cover = float(latest.loc[index_by_key[key], "_rule_days_of_cover"])
        record["rule_days_of_cover"] = (
            round(rule_cover, 6) if np.isfinite(rule_cover) else None
        )
        record["confidence"] = (
            "insufficient_history"
            if detail["history_days"] < 14
            else "low"
            if detail["missing_supply"] or detail["insufficient_inputs"] or detail["history_gaps"]
            else "medium"
            if detail["history_days"] < 28
            or detail["estimated_demand"]
            or detail["missing_lead_time"]
            or detail["missing_expiry_window"]
            or detail["missing_surge_flag"]
            else "high"
        )
        record["quality_flags"] = sorted(
            set(detail["quality_flags"]) | drift_flags
        )
        record["model_version"] = bundle["manifest"]["model_version"]
        record["data_source"] = detail["data_source"]
        record["low_confidence_group"] = key[1] in LOW_CONFIDENCE_GROUPS

    ranking_horizon = "24h" if "24h" in horizons else horizons[0]
    ranked = []
    for key, record in results_by_key.items():
        value = record.get(f"risk_{ranking_horizon}")
        has_model_risk = value is not None
        rank_score = float(value) if has_model_risk else (
            1 / (1 + record["rule_days_of_cover"])
            if record["rule_days_of_cover"] is not None
            else -1.0
        )
        ranked.append((not has_model_risk, -rank_score, record["district"], key[1], record))
    ranked.sort(key=lambda item: item[:4])
    for rank, entry in enumerate(ranked, start=1):
        record = entry[4]
        record["rank"] = rank
        record["in_watchlist"] = rank <= top_k
    output_rows = [entry[4] for entry in ranked]
    source_values = sorted(set(row["data_source"] for row in output_rows))
    output_source = source_values[0] if len(source_values) == 1 else "derived"
    return {
        "as_of": resolved_as_of.isoformat().replace("+00:00", "Z"),
        "model_version": bundle["manifest"]["model_version"],
        "trained_on": bundle["manifest"]["trained_on"],
        "data_source": output_source,
        "drift": {
            "distribution_shift": "distribution_shift" in drift_flags,
            "psi_by_feature": {key: round(value, 6) for key, value in psi_values.items()},
        },
        "results": output_rows,
    }
