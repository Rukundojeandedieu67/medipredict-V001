"""Leakage-aware features and next-horizon observed-outcome labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = (
    "stock_units",
    "days_of_cover",
    "demand_mean_1d",
    "demand_mean_3d",
    "demand_mean_7d",
    "demand_mean_14d",
    "demand_trend_7d_vs_14d",
    "demand_volatility_7d",
    "supply_received_7d",
    "time_since_delivery_days",
    "expected_lead_time_hours",
    "expiring_units_7d",
    "district_admissions_weight",
    "weekday_sin",
    "weekday_cos",
    "group_share",
    "recent_surge_flag",
)
GROUP_FEATURES = tuple(f"group_{group}" for group in ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"))
FEATURE_COLUMNS = BASE_FEATURES + GROUP_FEATURES
TARGET_COLUMNS = ("y_24h", "y_48h")


def build_features(
    readings: pd.DataFrame,
    admissions_weights: dict[str, float],
    group_shares: dict[str, float],
    periods_per_day: int = 1,
    include_labels: bool = True,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build t-only features and labels based exclusively on outcomes after t."""
    if periods_per_day <= 0:
        raise ValueError("periods_per_day must be positive")
    required = {
        "timestamp",
        "node_id",
        "district",
        "blood_group",
        "stock_units",
        "demand_units",
        "supply_received",
        "lead_time_hours",
        "expiring_units_7d",
        "recent_surge_flag",
    }
    if include_labels:
        required.add("unmet_demand_units")
    missing = required.difference(readings.columns)
    if missing:
        raise ValueError(f"Simulation readings are missing required columns: {sorted(missing)}")

    frame = readings.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric_columns = (
        "stock_units",
        "demand_units",
        "supply_received",
        "lead_time_hours",
        "expiring_units_7d",
        "recent_surge_flag",
    )
    if include_labels:
        numeric_columns = numeric_columns + ("unmet_demand_units",)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame.sort_values(["node_id", "blood_group", "timestamp"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    if frame.empty:
        raise ValueError("No simulation reading series were supplied")

    horizon_24 = 24 * periods_per_day // 24
    horizon_48 = 48 * periods_per_day // 24
    window_sizes = {days: days * periods_per_day for days in (1, 3, 7, 14)}
    keys = ["node_id", "blood_group"]
    grouped = frame.groupby(keys, sort=False)
    features = pd.DataFrame(index=frame.index)
    features["stock_units"] = frame["stock_units"].astype(float)
    for days, window in window_sizes.items():
        rolling = grouped["demand_units"].rolling(
            window, min_periods=window
        ).mean()
        rolling.index = rolling.index.droplevel(keys)
        features[f"demand_mean_{days}d"] = rolling
    features["demand_trend_7d_vs_14d"] = (
        features["demand_mean_7d"] - features["demand_mean_14d"]
    )
    volatility = grouped["demand_units"].rolling(
        window_sizes[7], min_periods=window_sizes[7]
    ).std(ddof=0)
    volatility.index = volatility.index.droplevel(keys)
    features["demand_volatility_7d"] = volatility
    daily_mean = features["demand_mean_7d"] * periods_per_day
    features["days_of_cover"] = (
        features["stock_units"] / daily_mean.clip(lower=0.1)
    ).clip(upper=90)
    supply_rolling = grouped["supply_received"].rolling(
        window_sizes[7], min_periods=1
    ).sum()
    supply_rolling.index = supply_rolling.index.droplevel(keys)
    features["supply_received_7d"] = supply_rolling
    period_indices = grouped.cumcount().astype(np.int32)
    last_delivery = pd.Series(
        np.where(frame["supply_received"].to_numpy() > 0, period_indices, -1),
        index=frame.index,
    ).groupby([frame["node_id"], frame["blood_group"]], sort=False).cummax()
    features["time_since_delivery_days"] = np.where(
        last_delivery >= 0,
        (period_indices - last_delivery) / periods_per_day,
        np.nan,
    )
    features["expected_lead_time_hours"] = frame["lead_time_hours"].astype(float)
    features["expiring_units_7d"] = frame["expiring_units_7d"].astype(float)
    district_weights = frame["district"].map(admissions_weights)
    if district_weights.isna().any():
        unknown = sorted(frame.loc[district_weights.isna(), "district"].unique())
        raise ValueError(f"Missing admissions weights for districts: {unknown}")
    features["district_admissions_weight"] = district_weights.astype(float)
    weekday = frame["timestamp"].dt.dayofweek.to_numpy()
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    shares = frame["blood_group"].map(group_shares)
    if shares.isna().any():
        unknown = sorted(frame.loc[shares.isna(), "blood_group"].unique())
        raise ValueError(f"Missing RBC 2024 group shares: {unknown}")
    features["group_share"] = shares.astype(float)
    features["recent_surge_flag"] = frame["recent_surge_flag"].astype(float)
    for group in GROUP_FEATURES:
        features[group] = (frame["blood_group"] == group.removeprefix("group_")).astype(float)

    if include_labels:
        outcomes = frame["unmet_demand_units"].to_numpy() > 0
        future_24 = np.zeros(len(frame), dtype=bool)
        future_48 = np.zeros(len(frame), dtype=bool)
        outcome_series = pd.Series(outcomes, index=frame.index)
        outcome_groups = outcome_series.groupby(
            [frame["node_id"], frame["blood_group"]], sort=False
        )
        for offset in range(1, horizon_24 + 1):
            shifted = outcome_groups.shift(-offset).fillna(False).to_numpy(dtype=bool)
            future_24 |= shifted
        for offset in range(1, horizon_48 + 1):
            shifted = outcome_groups.shift(-offset).fillna(False).to_numpy(dtype=bool)
            future_48 |= shifted
        features["y_24h"] = future_24.astype("int8")
        features["y_48h"] = future_48.astype("int8")
        group_sizes = grouped["timestamp"].transform("size").to_numpy(dtype=np.int32)
        features["_future_complete"] = period_indices.to_numpy() + horizon_48 < group_sizes
    features["timestamp"] = frame["timestamp"]
    features["node_id"] = frame["node_id"]
    features["district"] = frame["district"]
    features["blood_group"] = frame["blood_group"]
    features["_period_index"] = period_indices
    features["_period_count"] = grouped["timestamp"].transform("size").to_numpy(dtype=np.int32)
    result = features
    if include_labels:
        result = result.loc[result["_future_complete"]].copy()
    if drop_incomplete:
        result.dropna(subset=list(BASE_FEATURES), inplace=True)
    return result
