import numpy as np
import pandas as pd

from medipredict.features import FEATURE_COLUMNS, build_features
from medipredict.modeling import (
    bootstrap_recall_intervals,
    paired_bootstrap_recall_difference,
    partition_seed_directories,
    time_partition_masks,
)


def test_labels_use_only_future_rows_with_current_stockout_excluded():
    days = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    unmet = np.zeros(40, dtype=int)
    unmet[30] = 1
    unmet[32] = 1
    readings = pd.DataFrame(
        {
            "timestamp": days,
            "node_id": "district:1",
            "district": "District One",
            "blood_group": "A-",
            "stock_units": 5,
            "demand_units": 1,
            "supply_received": 1,
            "lead_time_hours": 12.0,
            "expiring_units_7d": 0,
            "recent_surge_flag": 0,
            "unmet_demand_units": unmet,
        }
    )
    features = build_features(
        readings,
        {"District One": 1.0},
        {"A-": 1.0},
    )
    previous_day = features.loc[features["_period_index"] == 29].iloc[0]
    prediction_time = features.loc[features["_period_index"] == 30].iloc[0]
    next_day = features.loc[features["_period_index"] == 31].iloc[0]
    assert previous_day["y_24h"] == 1
    assert prediction_time["y_24h"] == 0
    assert prediction_time["y_48h"] == 1
    assert next_day["y_24h"] == 1
    assert next_day["y_48h"] == 1
    assert features["_period_index"].max() == 37
    assert not ({"unmet_demand_units", "y_24h", "y_48h"} & set(FEATURE_COLUMNS))


def test_seed_partitions_are_disjoint_and_60_20_20():
    seeds = [f"seed_{index:02d}" for index in range(20)]
    split = partition_seed_directories(seeds)
    assert (len(split.train), len(split.validation), len(split.test)) == (12, 4, 4)
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))


def test_time_windows_are_chronological_with_three_day_purges():
    indices = np.arange(100)
    train, validation, test = time_partition_masks(indices, 100, gap_days=3)
    assert indices[train].max() == 59
    assert indices[validation].min() == 63
    assert indices[validation].max() == 79
    assert indices[test].min() == 83
    assert not (np.any(train & validation) or np.any(validation & test))


def test_paired_bootstrap_recall_difference_is_zero_for_identical_rankings():
    labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    clusters = np.array(["day1"] * 3 + ["day2"] * 3)
    intervals = paired_bootstrap_recall_difference(
        labels, scores, scores, clusters, budgets=(1, 3), replicates=40, seed=5
    )
    assert intervals == {"1": [0.0, 0.0], "3": [0.0, 0.0]}


def test_bootstrap_recall_intervals_have_one_interval_per_budget():
    labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2])
    clusters = np.array(["day1"] * 3 + ["day2"] * 3)
    intervals = bootstrap_recall_intervals(
        labels, scores, clusters, budgets=(1, 3), replicates=40, seed=5
    )
    assert set(intervals) == {"1", "3"}
    assert intervals["1"][0] <= intervals["1"][1]
    assert intervals["3"] == [1.0, 1.0]
