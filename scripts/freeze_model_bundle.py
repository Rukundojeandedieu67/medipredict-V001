"""Freeze the accepted Phase 2 artifacts as an immutable versioned bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from medipredict.features import FEATURE_COLUMNS, build_features
from medipredict.inference.bundle import MODEL_VERSION, sha256_file
from medipredict.modeling import partition_seed_directories, time_partition_masks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / MODEL_VERSION
FEATURE_UNITS = {
    "stock_units": "units",
    "days_of_cover": "days",
    "demand_mean_1d": "units/day",
    "demand_mean_3d": "units/day",
    "demand_mean_7d": "units/day",
    "demand_mean_14d": "units/day",
    "demand_trend_7d_vs_14d": "units/day",
    "demand_volatility_7d": "units/day",
    "supply_received_7d": "units",
    "time_since_delivery_days": "days",
    "expected_lead_time_hours": "hours",
    "expiring_units_7d": "units",
    "district_admissions_weight": "fraction",
    "weekday_sin": "dimensionless",
    "weekday_cos": "dimensionless",
    "group_share": "fraction",
    "recent_surge_flag": "boolean",
}


def _git_commit() -> str:
    candidates = ["git"]
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Git\cmd\git.exe",
                r"C:\Program Files\Git\bin\git.exe",
            ]
        )
    for executable in candidates:
        try:
            result = subprocess.run(
                [executable, "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Could not determine the Git commit for the model manifest")


def _git_worktree_dirty() -> bool:
    candidates = ["git"]
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Git\cmd\git.exe",
                r"C:\Program Files\Git\bin\git.exe",
            ]
        )
    for executable in candidates:
        try:
            result = subprocess.run(
                [executable, "-C", str(ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Could not determine whether the Git worktree is clean")


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_features() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    generated_dir = ROOT / "data" / "generated"
    seed_dirs = sorted(
        path for path in generated_dir.glob("seed_*") if path.is_dir()
    )
    partitions = partition_seed_directories(path.name for path in seed_dirs)
    seed_numbers_by_directory = {
        path.name: int(
            json.loads(
                (path / "simulation_summary.json").read_text(encoding="utf-8")
            )["seed"]
        )
        for path in seed_dirs
    }
    train_seeds = set(partitions.train)
    reference_dir = ROOT / "data" / "reference"
    districts = pd.read_csv(reference_dir / "dim_district.csv")
    admissions = districts.set_index("district")["hospital_admissions"].astype(float)
    admissions_weights = (admissions / admissions.sum()).to_dict()
    mix = pd.read_csv(reference_dir / "rbc_2024_collected_by_group_rcbt.csv")
    totals = mix.groupby("blood_group")["units_collected"].sum()
    total_collected = float(totals.sum())
    group_shares = {
        group: float(totals[group] / total_collected)
        for group in ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")
    }

    feature_chunks: dict[str, list[np.ndarray]] = {
        feature: [] for feature in FEATURE_COLUMNS
    }
    seed_numbers: list[int] = []
    for seed_dir in seed_dirs:
        if seed_dir.name not in train_seeds:
            continue
        summary = json.loads(
            (seed_dir / "simulation_summary.json").read_text(encoding="utf-8")
        )
        seed_numbers.append(int(summary["seed"]))
        readings = pd.read_csv(seed_dir / summary["stock_readings_file"])
        features = build_features(
            readings,
            admissions_weights,
            group_shares,
            periods_per_day=1,
        )
        period_count = int(summary["period_days"])
        train_end = int(period_count * 0.6)
        training_mask, _, _ = time_partition_masks(
            features["_period_index"].to_numpy(dtype=np.int32),
            period_count,
            periods_per_day=1,
            gap_days=3,
        )
        training_mask &= features["_period_index"].to_numpy(dtype=np.int32) < train_end
        values = features.loc[training_mask, FEATURE_COLUMNS]
        for feature in FEATURE_COLUMNS:
            feature_chunks[feature].append(values[feature].to_numpy(dtype=np.float64))
    if len(seed_numbers) != len(partitions.train):
        raise ValueError("Generated seed directories do not match the model training split")
    return {
        feature: np.concatenate(chunks)
        for feature, chunks in feature_chunks.items()
    }, {
        "seed_splits": {
            "train": list(partitions.train),
            "validation": list(partitions.validation),
            "test": list(partitions.test),
        },
        "seed_numbers_by_split": {
            split: [seed_numbers_by_directory[name] for name in getattr(partitions, split)]
            for split in ("train", "validation", "test")
        },
        "training_seed_numbers": seed_numbers,
    }


def _feature_statistics(values: dict[str, np.ndarray], model: Any) -> dict[str, Any]:
    importances = np.asarray(model.feature_importances_, dtype=float)
    top_10 = [
        name
        for name, _ in sorted(
            zip(FEATURE_COLUMNS, importances, strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )[:10]
    ]
    stats: dict[str, Any] = {}
    for name, data in values.items():
        edges = np.histogram_bin_edges(data, bins=20)
        counts, edges = np.histogram(data, bins=edges)
        quantiles = np.quantile(
            data, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        stats[name] = {
            "count": int(data.size),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "minimum": float(np.min(data)),
            "maximum": float(np.max(data)),
            "quantiles": {
                key: float(value)
                for key, value in zip(
                    ("1%", "5%", "25%", "50%", "75%", "95%", "99%"),
                    quantiles,
                    strict=True,
                )
            },
            "histogram": {
                "edges": edges.tolist(),
                "counts": counts.tolist(),
            },
            "feature_importance": float(
                importances[FEATURE_COLUMNS.index(name)]
            ),
        }
    stats["top_10_features"] = top_10
    return stats


def _feature_schema(stats: dict[str, Any]) -> dict[str, Any]:
    features = []
    for name in FEATURE_COLUMNS:
        unit = FEATURE_UNITS.get(name)
        if unit is None and name.startswith("group_"):
            unit = "boolean"
        limits = stats[name]
        features.append(
            {
                "name": name,
                "dtype": "float32",
                "unit": unit,
                "allowed_range": [limits["minimum"], limits["maximum"]],
            }
        )
    return {"schema_version": "1.0.0", "feature_order": list(FEATURE_COLUMNS), "features": features}


def freeze(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing versioned model bundle: {output_dir}"
        )
    source_model = ROOT / "models" / "phase2_model.joblib"
    source_metrics = ROOT / "docs" / "metrics.json"
    source_card = ROOT / "docs" / "MODEL_CARD.md"
    if not all(path.is_file() for path in (source_model, source_metrics, source_card)):
        raise FileNotFoundError("Accepted Phase 2 model, metrics, or model card is missing")
    phase2 = joblib.load(source_model)
    feature_columns = list(FEATURE_COLUMNS)
    if phase2.get("data_source") != "simulated":
        raise ValueError("Only the accepted simulated-data model is eligible for v0.1.0")
    for horizon in ("24h", "48h"):
        if phase2["models"][horizon].get("feature_columns") != feature_columns:
            raise ValueError(f"Feature schema mismatch for {horizon}")

    values, seed_metadata = _training_features()
    reference_stats = _feature_statistics(values, phase2["models"]["24h"]["classifier"])
    target_parent = output_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = target_parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        for horizon in ("24h", "48h"):
            model = dict(phase2["models"][horizon])
            calibrator = model.pop("calibrator")
            joblib.dump(model, staging_dir / f"model_{horizon}.joblib")
            joblib.dump(calibrator, staging_dir / f"calibrator_{horizon}.joblib")
        _json_write(staging_dir / "feature_schema.json", _feature_schema(reference_stats))
        _json_write(staging_dir / "reference_stats.json", reference_stats)
        shutil.copyfile(source_metrics, staging_dir / "metrics.json")
        shutil.copyfile(source_card, staging_dir / "MODEL_CARD.md")
        file_hashes = {
            path.relative_to(staging_dir).as_posix(): sha256_file(path)
            for path in sorted(staging_dir.iterdir())
            if path.is_file()
        }
        manifest = {
            "model_version": output_dir.name,
            "trained_on": "simulated",
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_worktree_dirty(),
            "seeds": seed_metadata,
            "feature_count": len(FEATURE_COLUMNS),
            "files": file_hashes,
            "manifest_hash_note": "manifest.json is excluded from files because a file cannot contain its own SHA-256 hash.",
        }
        _json_write(staging_dir / "manifest.json", manifest)
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = freeze(args.output_dir)
    print(f"Created immutable model bundle: {result}")


if __name__ == "__main__":
    main()
