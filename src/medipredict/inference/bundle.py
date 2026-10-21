"""Integrity-checked loading for immutable model bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


MODEL_VERSION = "v0.1.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIR = REPOSITORY_ROOT / "models" / MODEL_VERSION
HORIZONS = ("24h", "48h")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != bundle_dir.name:
        raise ValueError("Model bundle version does not match its directory name")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model bundle manifest has no file hashes")

    expected: set[str] = set()
    for relative_path, expected_hash in files.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in model bundle manifest: {relative_path}")
        expected.add(relative.as_posix())
        path = bundle_dir / relative
        if not path.is_file():
            raise ValueError(f"Model bundle file is missing: {relative_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Model bundle hash mismatch: {relative_path}")

    actual = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        extras = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"Model bundle file list mismatch; unlisted={extras}, missing={missing}")
    return manifest


def load_bundle(bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> dict[str, Any]:
    manifest = verify_bundle(bundle_dir)
    models = {}
    for horizon in HORIZONS:
        model = joblib.load(bundle_dir / f"model_{horizon}.joblib")
        model["calibrator"] = joblib.load(
            bundle_dir / f"calibrator_{horizon}.joblib"
        )
        models[horizon] = model
    feature_schema = json.loads(
        (bundle_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    reference_stats = json.loads(
        (bundle_dir / "reference_stats.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_names = [feature["name"] for feature in feature_schema["features"]]
    for horizon, model in models.items():
        if model.get("label") != f"y_{horizon}":
            raise ValueError(f"Unexpected label in {horizon} model artifact")
        if model.get("feature_columns") != feature_names:
            raise ValueError(f"Feature schema mismatch in {horizon} model artifact")
    return {
        "manifest": manifest,
        "models": models,
        "feature_schema": feature_schema,
        "reference_stats": reference_stats,
        "metrics": metrics,
    }
