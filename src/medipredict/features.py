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
        "unmet_demand_units",
    }
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
        "unmet_demand_units",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame.sort_values(["node_id", "blood_group", "timestamp"], inplace=True, kind="stable")

    pieces: list[pd.DataFrame] = []
    horizon_24 = 24 * periods_per_day // 24
    horizon_48 = 48 * periods_per_day // 24
    window_sizes = {days: days * periods_per_day for days in (1, 3, 7, 14)}
    for (_, _), series in frame.groupby(["node_id", "blood_group"], sort=False):
        series = series.copy()
        demand = series["demand_units"].astype(float)
        features = pd.DataFrame(index=series.index)
        features["stock_units"] = series["stock_units"].astype(float)
        for days, window in window_sizes.items():
            features[f"demand_mean_{days}d"] = demand.rolling(
                window, min_periods=window
            ).mean()
        features["demand_trend_7d_vs_14d"] = (
            features["demand_mean_7d"] - features["demand_mean_14d"]
        )
        features["demand_volatility_7d"] = demand.rolling(
            window_sizes[7], min_periods=window_sizes[7]
        ).std(ddof=0)
        daily_mean = features["demand_mean_7d"] * periods_per_day
        features["days_of_cover"] = (
            features["stock_units"] / daily_mean.clip(lower=0.1)
        ).clip(upper=90)
        features["supply_received_7d"] = series["supply_received"].rolling(
            window_sizes[7], min_periods=1
        ).sum()
        step_index = np.arange(len(series))
        last_delivery = np.maximum.accumulate(
            np.where(series["supply_received"].to_numpy() > 0, step_index, -1)
        )
        features["time_since_delivery_days"] = np.where(
            last_delivery >= 0, (step_index - last_delivery) / periods_per_day, np.nan
        )
        features["expected_lead_time_hours"] = series["lead_time_hours"].astype(float)
        features["expiring_units_7d"] = series["expiring_units_7d"].astype(float)
        district_weights = series["district"].map(admissions_weights)
        if district_weights.isna().any():
            unknown = sorted(series.loc[district_weights.isna(), "district"].unique())
            raise ValueError(f"Missing admissions weights for districts: {unknown}")
        features["district_admissions_weight"] = district_weights.astype(float)
        weekday = series["timestamp"].dt.dayofweek.to_numpy()
        features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
        features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
        shares = series["blood_group"].map(group_shares)
        if shares.isna().any():
            unknown = sorted(series.loc[shares.isna(), "blood_group"].unique())
            raise ValueError(f"Missing RBC 2024 group shares: {unknown}")
        features["group_share"] = shares.astype(float)
        features["recent_surge_flag"] = series["recent_surge_flag"].astype(float)
        for group in GROUP_FEATURES:
            features[group] = (series["blood_group"] == group.removeprefix("group_")).astype(float)

        outcomes = series["unmet_demand_units"].to_numpy() > 0
        future_24 = np.zeros(len(series), dtype=bool)
        future_48 = np.zeros(len(series), dtype=bool)
        for offset in range(1, horizon_24 + 1):
            future_24[:-offset] |= outcomes[offset:]
        for offset in range(1, horizon_48 + 1):
            future_48[:-offset] |= outcomes[offset:]
        features["y_24h"] = future_24.astype("int8")
        features["y_48h"] = future_48.astype("int8")
        features["_future_complete"] = (
            np.arange(len(series)) + horizon_48 < len(series)
        )
        features["timestamp"] = series["timestamp"]
        features["node_id"] = series["node_id"]
        features["district"] = series["district"]
        features["blood_group"] = series["blood_group"]
        features["_period_index"] = np.arange(len(series), dtype=np.int32)
        features["_period_count"] = len(series)
        pieces.append(features)

    if not pieces:
        raise ValueError("No simulation reading series were supplied")
    result = pd.concat(pieces, ignore_index=True)
    result = result.loc[result["_future_complete"]].copy()
    result.dropna(subset=list(BASE_FEATURES), inplace=True)
    return result
