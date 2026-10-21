import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medipredict.features import BASE_FEATURES, FEATURE_COLUMNS, build_features
from medipredict.inference.bundle import (
    DEFAULT_BUNDLE_DIR,
    load_bundle,
    verify_bundle,
)
from medipredict.inference.score import (
    _feature_values,
    _reference_metadata,
    score,
)
from scripts.freeze_model_bundle import freeze


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "files" / "sample_readings_30d_SIMULATED.csv"


def _sample() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE)
    frame["data_source"] = "simulated"
    return frame


def test_frozen_bundle_hashes_and_features_are_valid():
    manifest = verify_bundle()
    bundle = load_bundle()
    assert manifest["model_version"] == "v0.1.0"
    assert manifest["trained_on"] == "simulated"
    assert bundle["feature_schema"]["feature_order"] == list(FEATURE_COLUMNS)
    assert len(bundle["reference_stats"]["top_10_features"]) == 10
    assert set(bundle["models"]) == {"24h", "48h"}


def test_bundle_loader_rejects_hash_mismatch(tmp_path):
    copy = tmp_path / "v0.1.0"
    shutil.copytree(DEFAULT_BUNDLE_DIR, copy)
    with (copy / "metrics.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(copy)


def test_freezer_refuses_to_overwrite_versioned_bundle():
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze(DEFAULT_BUNDLE_DIR)


def test_inference_feature_parity_with_training_feature_builder():
    seed_dir = ROOT / "data" / "generated" / "seed_00"
    summary = json.loads(
        (seed_dir / "simulation_summary.json").read_text(encoding="utf-8")
    )
    readings = pd.read_csv(seed_dir / summary["stock_readings_file"])
    first_node, first_group = readings[["node_id", "blood_group"]].iloc[0]
    series = readings.loc[
        (readings["node_id"] == first_node)
        & (readings["blood_group"] == first_group)
    ].copy()
    admissions, shares = _reference_metadata()
    expected = build_features(series, admissions, shares)
    actual, _ = _feature_values(
        series,
        pd.to_datetime(series["timestamp"], utc=True).max(),
        load_bundle()["reference_stats"],
        admissions,
        shares,
    )
    expected = expected.set_index("timestamp")
    actual = actual.set_index("timestamp")
    shared = expected.index.intersection(actual.index)
    np.testing.assert_allclose(
        actual.loc[shared, list(BASE_FEATURES)].to_numpy(dtype=float),
        expected.loc[shared, list(BASE_FEATURES)].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.loc[shared, [name for name in FEATURE_COLUMNS if name.startswith("group_")]].to_numpy(dtype=float),
        expected.loc[shared, [name for name in FEATURE_COLUMNS if name.startswith("group_")]].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


def test_sample_scores_all_240_series_deterministically():
    sample = _sample()
    first = score(sample, top_k=5)
    second = score(sample, top_k=5)
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["data_source"] == "simulated"
    assert first["model_version"] == "v0.1.0"
    assert len(first["results"]) == 240
    assert all(row["risk_24h"] is not None for row in first["results"])
    assert all(row["risk_48h"] is not None for row in first["results"])
    assert all(row["confidence"] in {"medium", "high"} for row in first["results"])
    assert sum(row["in_watchlist"] for row in first["results"]) == 5
    assert all(len(row["top_drivers"]) == 3 for row in first["results"])
    assert all(
        row["low_confidence_group"] == (row["blood_group"] in {"A-", "B-", "AB-"})
        for row in first["results"]
    )


def test_insufficient_history_returns_rule_only():
    sample = _sample()
    sample = sample.loc[sample["timestamp"] <= "2026-09-08"]
    result = score(sample)
    assert len(result["results"]) == 240
    assert all(row["confidence"] == "insufficient_history" for row in result["results"])
    assert all(row["risk_24h"] is None and row["risk_48h"] is None for row in result["results"])
    assert all(not row["top_drivers"] for row in result["results"])


def test_as_of_and_drift_flags_are_applied():
    sample = _sample()
    sample.loc[sample["timestamp"] == "2026-09-30", "stock_units"] += 20000
    result = score(sample, as_of="2026-09-29", top_k=1)
    expected = score(
        _sample(),
        as_of="2026-09-29",
        top_k=1,
    )
    assert result["as_of"] == "2026-09-29T00:00:00Z"
    assert result == expected
    assert all(row["rank"] >= 1 for row in result["results"])

    drift = score(sample, top_k=1)
    assert drift["drift"]["distribution_shift"]
    assert all("distribution_shift" in row["quality_flags"] for row in drift["results"])


def test_demand_is_estimated_when_missing():
    sample = _sample().drop(columns=["demand_units"])
    result = score(sample, top_k=1)
    assert len(result["results"]) == 240
    assert all("demand_column_estimated" in row["quality_flags"] for row in result["results"])
    assert all("demand_estimated" in row["quality_flags"] for row in result["results"])


def test_partial_supply_and_missing_days_lower_confidence():
    sample = _sample()
    without_supply = score(sample.drop(columns=["supply_received"]), top_k=1)
    assert all(row["confidence"] == "low" for row in without_supply["results"])
    assert all(
        "supply_missing_assumed_zero" in row["quality_flags"]
        for row in without_supply["results"]
    )

    series_key = (sample.iloc[0]["district"], sample.iloc[0]["blood_group"])
    gap_mask = (
        (sample["district"] == series_key[0])
        & (sample["blood_group"] == series_key[1])
        & sample["timestamp"].between("2026-09-10", "2026-09-15")
    )
    with_gap = score(sample.loc[~gap_mask], top_k=1)
    affected = next(
        row
        for row in with_gap["results"]
        if row["district"] == series_key[0] and row["blood_group"] == series_key[1]
    )
    assert affected["confidence"] == "low"
    assert "missing_daily_observations" in affected["quality_flags"]
