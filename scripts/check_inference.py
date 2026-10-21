"""Run the frozen-bundle integrity, parity, determinism and latency checks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from medipredict.features import FEATURE_COLUMNS, build_features
from medipredict.inference.bundle import load_bundle, verify_bundle
from medipredict.inference.score import (
    _feature_values,
    _reference_metadata,
)
from medipredict.inference.score import score


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "files" / "sample_readings_30d_SIMULATED.csv"


def main() -> None:
    manifest = verify_bundle()
    bundle = load_bundle()
    seed_dir = ROOT / "data" / "generated" / "seed_00"
    summary = json.loads(
        (seed_dir / "simulation_summary.json").read_text(encoding="utf-8")
    )
    parity_readings = pd.read_csv(seed_dir / summary["stock_readings_file"])
    node_id, group = parity_readings[["node_id", "blood_group"]].iloc[0]
    parity_readings = parity_readings.loc[
        (parity_readings["node_id"] == node_id)
        & (parity_readings["blood_group"] == group)
    ]
    admissions, group_shares = _reference_metadata()
    expected = build_features(parity_readings, admissions, group_shares).set_index(
        "timestamp"
    )
    actual, _ = _feature_values(
        parity_readings,
        pd.to_datetime(parity_readings["timestamp"], utc=True).max(),
        bundle["reference_stats"],
        admissions,
        group_shares,
    )
    actual = actual.set_index("timestamp")
    shared = expected.index.intersection(actual.index)
    if not (
        np.allclose(
            actual.loc[shared, list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            expected.loc[shared, list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
    ):
        raise AssertionError("Inference features do not match training feature builder")

    sample = pd.read_csv(SAMPLE_PATH)
    sample["data_source"] = "simulated"
    cold_started = time.perf_counter()
    warm_output = score(sample, top_k=5)
    cold_start = time.perf_counter() - cold_started
    runs = []
    outputs = []
    for _ in range(7):
        started = time.perf_counter()
        outputs.append(score(sample, top_k=5))
        runs.append(time.perf_counter() - started)
    if any(output != warm_output for output in outputs):
        raise AssertionError("Scoring output changed across identical calls")
    if len(warm_output["results"]) != 240:
        raise AssertionError("Sample did not score all 240 district/group series")
    p95 = float(np.percentile(runs, 95))
    if p95 >= 2.0:
        raise AssertionError("p95 latency did not meet the 2-second target")
    print(
        json.dumps(
            {
                "hash_verification": "PASS",
                "feature_parity": "PASS",
                "model_version": manifest["model_version"],
                "determinism": "PASS",
                "series_scored": len(warm_output["results"]),
                "cold_start_seconds": cold_start,
                "latency_seconds": {
                    "p50": float(np.median(runs)),
                    "p95": p95,
                },
                "sample_watchlist": [
                    row
                    for row in warm_output["results"]
                    if row["in_watchlist"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
