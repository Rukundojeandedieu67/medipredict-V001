"""Clean supported datasets in a directory without modifying the originals.

Supported inputs: CSV, headerless .data, Parquet, Excel workbooks, and GeoJSON.
Cleaned tables are written as CSV files to ``data/cleaned`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".data", ".parquet", ".xlsx", ".xls", ".geojson"}
MISSING_VALUES = {"", "na", "n/a", "nan", "null", "none", "<na>"}


def _clean_column_name(name: Any) -> str:
    """Convert a column label to a stable snake_case name."""
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "unnamed"


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a table using safe, general-purpose rules."""
    cleaned = frame.copy()
    cleaned.columns = [_clean_column_name(column) for column in cleaned.columns]

    # Keep duplicate labels unique after normalization (e.g. "Facility Name" and
    # "facility-name" both become "facility_name").
    counts: dict[str, int] = {}
    unique_columns = []
    for column in cleaned.columns:
        counts[column] = counts.get(column, 0) + 1
        unique_columns.append(column if counts[column] == 1 else f"{column}_{counts[column]}")
    cleaned.columns = unique_columns

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        values = cleaned[column].astype("string").str.strip()
        missing_mask = values.str.casefold().isin(MISSING_VALUES)
        cleaned[column] = values.mask(missing_mask, pd.NA)

    cleaned = cleaned.dropna(how="all").drop_duplicates(ignore_index=True)
    return cleaned


def _read_geojson(path: Path) -> pd.DataFrame:
    """Read GeoJSON feature properties while retaining geometry as GeoJSON text."""
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    features = document.get("features", [])
    rows = []
    for feature in features:
        row = dict(feature.get("properties") or {})
        geometry = feature.get("geometry")
        if geometry is not None:
            row["geometry"] = json.dumps(geometry, ensure_ascii=False, separators=(",", ":"))
        if feature.get("id") is not None:
            row.setdefault("feature_id", feature["id"])
        rows.append(row)
    return pd.DataFrame(rows)


def _read_tables(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Read one input file, returning (sheet/table name, frame) pairs."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [(path.stem, pd.read_csv(path, low_memory=False))]
    if suffix == ".data":
        # UCI transfusion.data is a comma-separated, headerless table.
        frame = pd.read_csv(path, header=None, low_memory=False)
        if path.stem.lower() == "transfusion":
            expected = ["recency_months", "frequency_times", "monetary_cc", "time_months", "donated"]
            if frame.shape[1] == len(expected):
                frame.columns = expected
        return [(path.stem, frame)]
    if suffix == ".parquet":
        return [(path.stem, pd.read_parquet(path))]
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return [(f"{path.stem}_{sheet}", frame) for sheet, frame in sheets.items()]
    if suffix == ".geojson":
        return [(path.stem, _read_geojson(path))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_data_directory(
    data_dir: str | Path = "../data",
    output_dir: str | Path | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Clean supported files in ``data_dir`` and save output tables as CSV.

    Parameters
    ----------
    data_dir:
        Directory containing source datasets.
    output_dir:
        Destination directory. Defaults to ``<data_dir>/cleaned``.
    recursive:
        Include supported files in subdirectories.
    overwrite:
        Replace prior output files with the same names.

    Returns
    -------
    dict
        A report with written files, row counts, and skipped/failed inputs.
    """
    source_dir = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else source_dir / "cleaned"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {source_dir}")
    if destination == source_dir:
        raise ValueError("output_dir cannot be the source data directory; use a separate output folder")

    destination.mkdir(parents=True, exist_ok=True)
    candidates = source_dir.rglob("*") if recursive else source_dir.iterdir()
    input_files = sorted(
        path for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and destination not in path.resolve().parents
    )

    report: dict[str, Any] = {"input_dir": str(source_dir), "output_dir": str(destination), "processed": [], "skipped": [], "errors": []}
    for path in input_files:
        try:
            for table_name, frame in _read_tables(path):
                cleaned = clean_dataframe(frame)
                # Keep nested source folders when recursive scanning is enabled.
                relative_parent = path.parent.relative_to(source_dir)
                target_dir = destination / relative_parent
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{_clean_column_name(table_name)}.csv"
                if target.exists() and not overwrite:
                    report["skipped"].append({"file": str(path), "reason": f"output exists: {target}"})
                    continue
                cleaned.to_csv(target, index=False, na_rep="")
                report["processed"].append({
                    "file": str(path), "output": str(target),
                    "rows_in": int(len(frame)), "rows_out": int(len(cleaned)),
                    "columns": list(cleaned.columns),
                })
        except Exception as error:  # report a bad source without stopping the batch
            report["errors"].append({"file": str(path), "error": f"{type(error).__name__}: {error}"})

    supported = set(input_files)
    all_files = source_dir.rglob("*") if recursive else source_dir.iterdir()
    report["skipped"].extend(
        {"file": str(path), "reason": "unsupported file format"}
        for path in sorted(all_files)
        if path.is_file() and path not in supported and destination not in path.resolve().parents
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean supported datasets without changing source files.")
    parser.add_argument("data_dir", nargs="?", default="../data", help="directory containing source datasets")
    parser.add_argument("--output-dir", help="destination for cleaned CSV files (default: <data_dir>/cleaned)")
    parser.add_argument("--recursive", action="store_true", help="scan subdirectories")
    parser.add_argument("--no-overwrite", action="store_true", help="skip outputs that already exist")
    args = parser.parse_args()

    report = clean_data_directory(
        args.data_dir,
        args.output_dir,
        recursive=args.recursive,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
