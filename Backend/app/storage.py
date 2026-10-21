"""Safe local dataset storage with bounded retention."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

RETENTION = timedelta(days=7)


class DatasetNotFoundError(LookupError):
    """Raised when a dataset is missing or has expired."""


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset_id: str) -> Path:
        try:
            parsed = uuid.UUID(dataset_id)
        except (ValueError, AttributeError) as error:
            raise DatasetNotFoundError(dataset_id) from error
        target = (self.root / str(parsed)).resolve()
        if target.parent != self.root:
            raise DatasetNotFoundError(dataset_id)
        return target

    def cleanup_expired(self, now: datetime | None = None) -> None:
        cutoff = (now or datetime.now(UTC)) - RETENTION
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            metadata_path = directory / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(metadata["created_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created < cutoff:
                resolved = directory.resolve()
                if resolved.parent == self.root:
                    shutil.rmtree(resolved)

    def create(self, metadata: dict[str, Any], frame: pd.DataFrame | None) -> str:
        self.cleanup_expired()
        dataset_id = str(uuid.uuid4())
        directory = self._dataset_dir(dataset_id)
        directory.mkdir()
        metadata = dict(metadata)
        metadata.update(
            {
                "dataset_id": dataset_id,
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        if frame is not None:
            (directory / "source.json").write_text(
                frame.to_json(orient="records", date_format="iso", force_ascii=False),
                encoding="utf-8",
            )
        self._write_metadata(directory, metadata)
        return dataset_id

    def get_metadata(self, dataset_id: str) -> dict[str, Any]:
        self.cleanup_expired()
        directory = self._dataset_dir(dataset_id)
        try:
            return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetNotFoundError(dataset_id) from error

    def save_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> None:
        directory = self._dataset_dir(dataset_id)
        if not directory.is_dir():
            raise DatasetNotFoundError(dataset_id)
        metadata = dict(metadata)
        metadata["updated_at"] = datetime.now(UTC).isoformat()
        self._write_metadata(directory, metadata)

    @staticmethod
    def _write_metadata(directory: Path, metadata: dict[str, Any]) -> None:
        temporary = directory / "metadata.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(directory / "metadata.json")

    def get_frame(self, dataset_id: str) -> pd.DataFrame:
        directory = self._dataset_dir(dataset_id)
        try:
            frame = pd.read_json(StringIO((directory / "source.json").read_text(encoding="utf-8")))
            metadata = self.get_metadata(dataset_id)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise DatasetNotFoundError(dataset_id) from error
        frame.attrs["source_filename"] = metadata["filename"]
        frame.attrs["source_row_offset"] = metadata.get("source_row_offset", 2)
        frame.attrs["header_row"] = metadata.get("header_row")
        frame.attrs["sensitive_columns_dropped"] = metadata.get(
            "sensitive_columns_dropped", []
        )
        return frame

    def delete(self, dataset_id: str) -> None:
        directory = self._dataset_dir(dataset_id)
        if not directory.is_dir():
            raise DatasetNotFoundError(dataset_id)
        shutil.rmtree(directory)

    def save_results(self, dataset_id: str, result: dict[str, Any]) -> None:
        directory = self._dataset_dir(dataset_id)
        if not directory.is_dir():
            raise DatasetNotFoundError(dataset_id)
        path = directory / "results.json"
        temporary = directory / "results.tmp"
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def get_results(self, dataset_id: str) -> dict[str, Any] | None:
        directory = self._dataset_dir(dataset_id)
        try:
            return json.loads((directory / "results.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
