"""Map uploaded tabular data to the daily district/group readings schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from medipredict.ingest.mapping import CANONICAL_COLUMNS, REQUIRED_COLUMNS, normalize_header


UNIT_COLUMNS = (
    "stock_units",
    "demand_units",
    "supply_received",
    "expired_units",
    "unmet_demand_units",
)
OPTIONAL_COLUMNS = (
    "demand_units",
    "supply_received",
    "expired_units",
    "unmet_demand_units",
    "facility_id",
)
OUTPUT_COLUMNS = (
    "timestamp",
    "district",
    "blood_group",
    "stock_units",
    "demand_units",
    "supply_received",
    "expired_units",
    "unmet_demand_units",
    "data_source",
    "demand_estimated",
    "quality_flags",
)
REJECTION_EXAMPLES_LIMIT = 200
DETAIL_EXAMPLES_LIMIT = 200


class MappingError(ValueError):
    """A missing or ambiguous user-confirmed column mapping."""


@dataclass
class NormalizationResult:
    data: pd.DataFrame
    report: dict[str, Any]


def _official_districts(reference_dir: str | None = None) -> list[str]:
    from pathlib import Path

    if reference_dir is None:
        path = Path(__file__).resolve().parents[3] / "data" / "reference" / "dim_district.csv"
    else:
        path = Path(reference_dir) / "dim_district.csv"
    districts = pd.read_csv(path)
    return districts["district"].astype(str).tolist()


def _blood_group(value: Any) -> str | None:
    if pd.isna(value):
        return None
    compact = re.sub(r"[^A-Z0-9+-]", "", str(value).strip().upper())
    compact = compact.replace("RH", "")
    for suffix in ("POSITIVE", "NEGATIVE", "POS", "NEG"):
        if compact.endswith(suffix):
            compact = compact[: -len(suffix)] + ("+" if suffix.startswith("POS") else "-")
            break
    match = re.fullmatch(r"(AB|A|B|O)([+-])", compact)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return None


def _district_match(value: Any, official: list[str]) -> tuple[str | None, str | None, float]:
    if pd.isna(value):
        return None, None, 0.0
    text = str(value).strip()
    if not text:
        return None, None, 0.0
    exact_by_key = {normalize_header(name): name for name in official}
    normalized = normalize_header(text)
    exact = exact_by_key.get(normalized)
    if exact is not None:
        return exact, None, 1.0
    ranked = sorted(
        (
            (SequenceMatcher(None, normalized, normalize_header(name)).ratio(), name)
            for name in official
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )
    confidence, candidate = ranked[0]
    if confidence >= 0.92:
        return candidate, None, confidence
    if confidence >= 0.80:
        return None, candidate, confidence
    return None, None, confidence


def _sensitive_column(column: Any) -> bool:
    name = normalize_header(column)
    sensitive_phrases = (
        "patient",
        "name",
        "nom",
        "national id",
        "id national",
        "nid",
        "phone",
        "telephone",
        "mobile",
        "address",
        "adresse",
        "passport",
        "identity number",
    )
    return any(
        name == phrase
        or name.startswith(phrase + " ")
        or (" " + phrase + " ") in (" " + name + " ")
        for phrase in sensitive_phrases
    )


def _source_row_numbers(frame: pd.DataFrame) -> pd.Series:
    offset = int(frame.attrs.get("source_row_offset", 2))
    try:
        indices = pd.to_numeric(pd.Index(frame.index), errors="raise").to_numpy()
    except (TypeError, ValueError):
        indices = np.arange(len(frame))
    return pd.Series(indices + offset, index=frame.index, dtype="int64")


def _aggregate_duplicates(
    frame: pd.DataFrame, has_facility_rows: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["timestamp", "district", "blood_group"]
    duplicates = frame.duplicated(keys, keep=False)
    duplicate_groups = int(
        frame.loc[duplicates, keys].drop_duplicates().shape[0]
    )
    duplicate_rows = int(len(frame) - frame[keys].drop_duplicates().shape[0])
    duplicate_row_numbers = (
        frame.loc[frame.duplicated(keys, keep="first"), "_source_row_number"]
        .astype(int)
        .tolist()
    )
    if not duplicate_rows:
        return frame, {
            "duplicate_groups": 0,
            "duplicate_rows": 0,
            "duplicate_row_numbers": [],
        }
    additive = [
        column
        for column in (
            "demand_units",
            "supply_received",
            "expired_units",
            "unmet_demand_units",
        )
        if column in frame
    ]
    aggregations: dict[str, Any] = {
        "stock_units": "sum" if has_facility_rows else "last",
        "_source_row_number": "last",
        "demand_estimated": "max",
    }
    for column in additive:
        aggregations[column] = lambda values: values.sum(min_count=1)
    aggregated = frame.groupby(keys, sort=True, as_index=False, dropna=False).agg(
        aggregations
    )
    return aggregated, {
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_numbers": duplicate_row_numbers,
    }


def _derive_demand(frame: pd.DataFrame) -> pd.DataFrame:
    if "demand_units" not in frame:
        frame["demand_units"] = np.nan
    if "supply_received" not in frame:
        frame["supply_received"] = np.nan
    if "expired_units" not in frame:
        frame["expired_units"] = np.nan
    frame["demand_estimated"] = False
    if frame["supply_received"].notna().any():
        for _, indices in frame.groupby(["district", "blood_group"], sort=False).groups.items():
            ordered = frame.loc[indices].sort_values("timestamp", kind="stable")
            previous_stock = ordered["stock_units"].shift(1).fillna(
                ordered["stock_units"]
            )
            demand_estimate = (
                previous_stock
                + ordered["supply_received"].fillna(0)
                - ordered["expired_units"].fillna(0)
                - ordered["stock_units"]
            ).clip(lower=0)
            missing = ordered["demand_units"].isna()
            if missing.any():
                frame.loc[ordered.index[missing], "demand_units"] = demand_estimate.loc[
                    missing
                ]
                frame.loc[ordered.index[missing], "demand_estimated"] = True
    return frame


def normalize_dataframe(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    *,
    reference_dir: str | None = None,
    now: pd.Timestamp | None = None,
) -> NormalizationResult:
    """Normalize one tabular upload after a user has confirmed its column mapping."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Uploaded table must contain at least one row")
    if len(frame) > 500_000:
        raise ValueError("Uploaded table exceeds the 500,000-row limit")
    if not isinstance(mapping, dict):
        raise MappingError("A user-confirmed canonical-to-source mapping is required")
    unknown_targets = sorted(set(mapping) - set(CANONICAL_COLUMNS))
    if unknown_targets:
        raise MappingError(f"Unsupported canonical mapping targets: {unknown_targets}")
    missing_required = [
        name for name in REQUIRED_COLUMNS if not mapping.get(name)
    ]
    if missing_required:
        raise MappingError(f"Confirm mappings for required columns: {missing_required}")
    missing_sources = sorted(
        {source for source in mapping.values() if source not in frame.columns}
    )
    if missing_sources:
        raise MappingError(f"Mapped source columns do not exist: {missing_sources}")
    duplicate_sources = {
        source for source in mapping.values() if list(mapping.values()).count(source) > 1
    }
    if duplicate_sources:
        raise MappingError(f"A source column cannot map to multiple fields: {sorted(duplicate_sources)}")

    selected_columns = set(mapping.values())
    district_source = mapping.get("district")
    reported_sensitive_columns = list(frame.attrs.get("sensitive_columns_dropped", []))
    reported_sensitive_columns.extend(
        column
        for column in frame.columns
        if column not in selected_columns and _sensitive_column(column)
    )
    sensitive_sources = sorted(
        str(column)
        for column in selected_columns
        if column != district_source and _sensitive_column(column)
    )
    if sensitive_sources:
        raise MappingError(
            f"Potential patient-identifying columns cannot be mapped: {sensitive_sources}"
        )
    reported_sensitive_columns = list(dict.fromkeys(reported_sensitive_columns))
    sensitive_columns = [
        column for column in reported_sensitive_columns if column in frame.columns
    ]
    source = frame.drop(columns=sensitive_columns).copy()
    source_row_numbers = _source_row_numbers(frame)
    mapped = pd.DataFrame(index=source.index)
    for canonical, source_column in mapping.items():
        mapped[canonical] = source[source_column]
    for column in OPTIONAL_COLUMNS:
        if column not in mapped:
            mapped[column] = np.nan
    mapped["_source_row_number"] = source_row_numbers

    rejected: dict[int, set[str]] = {}
    district_suggestions: list[dict[str, Any]] = []

    def reject(index: Any, reason: str) -> None:
        row_number = int(mapped.loc[index, "_source_row_number"])
        rejected.setdefault(row_number, set()).add(reason)

    parsed_dates = pd.to_datetime(
        mapped["timestamp"], errors="coerce", utc=True, format="mixed"
    )
    for index in mapped.index[parsed_dates.isna()]:
        reject(index, "invalid_timestamp")
    mapped["timestamp"] = parsed_dates

    official_districts = _official_districts(reference_dir)
    normalized_districts: list[str | None] = []
    for index, value in mapped["district"].items():
        district, suggestion, confidence = _district_match(value, official_districts)
        normalized_districts.append(district)
        if suggestion is not None:
            district_suggestions.append(
                {
                    "row_number": int(mapped.loc[index, "_source_row_number"]),
                    "input": str(value),
                    "suggestion": suggestion,
                    "confidence": round(confidence, 3),
                }
            )
            reject(index, "district_needs_confirmation")
        elif district is None:
            reject(index, "unknown_district")
    mapped["district"] = normalized_districts

    blood_groups = mapped["blood_group"].map(_blood_group)
    for index in mapped.index[blood_groups.isna()]:
        reject(index, "invalid_blood_group")
    mapped["blood_group"] = blood_groups

    for column in UNIT_COLUMNS:
        if column not in mapped:
            continue
        raw_values = mapped[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        invalid = raw_values.notna() & values.isna()
        for index in mapped.index[invalid]:
            reject(index, f"invalid_{column}")
        negative = values < 0
        for index in mapped.index[negative.fillna(False)]:
            reject(index, f"negative_{column}")
        mapped[column] = values
    for index in mapped.index[mapped["stock_units"].isna()]:
        reject(index, "missing_stock_units")

    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    elif now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    future = mapped["timestamp"] > now
    for index in mapped.index[future.fillna(False)]:
        reject(index, "future_timestamp")

    stock_over_limit = mapped["stock_units"] > 10_000
    for index in mapped.index[stock_over_limit.fillna(False)]:
        # Unusually large stock is reviewed but retained unless it violates unit validity.
        pass

    bad_indices = [
        index
        for index in mapped.index
        if int(mapped.loc[index, "_source_row_number"]) in rejected
    ]
    accepted = mapped.drop(index=bad_indices).copy()
    facility_mapping = mapping.get("facility_id")
    has_facility_rows = False
    if facility_mapping is not None and "facility_id" in accepted:
        accepted["facility_id"] = accepted["facility_id"].astype("string").str.strip()
        accepted.loc[accepted["facility_id"] == "", "facility_id"] = pd.NA
        has_facility_rows = accepted["facility_id"].notna().any()
    if has_facility_rows:
        accepted.drop(columns=["facility_id"], inplace=True)
    accepted["demand_estimated"] = False
    accepted, duplicate_report = _aggregate_duplicates(accepted, has_facility_rows)
    accepted = _derive_demand(accepted)

    quality_flags: dict[str, list[str]] = {}
    insufficient_series = 0
    for (district, group), indices in accepted.groupby(
        ["district", "blood_group"], sort=True
    ).groups.items():
        series = accepted.loc[indices].sort_values("timestamp", kind="stable")
        unique_days = series["timestamp"].dt.floor("D").drop_duplicates().sort_values()
        days = int(len(unique_days))
        flags: list[str] = []
        if days < 14:
            flags.append("insufficient_history")
            insufficient_series += 1
        demand_available = series["demand_units"].notna().any()
        supply_available = series["supply_received"].notna().any()
        if not demand_available and not supply_available:
            flags.append("insufficient_inputs")
        elif series["demand_estimated"].any():
            flags.append("demand_estimated")
        if flags:
            quality_flags[f"{district}|{group}"] = flags

    gap_summaries: list[dict[str, Any]] = []
    total_missing_days = 0
    for (district, group), indices in accepted.groupby(
        ["district", "blood_group"], sort=True
    ).groups.items():
        days = (
            accepted.loc[indices, "timestamp"]
            .dt.floor("D")
            .drop_duplicates()
            .sort_values()
        )
        if len(days) < 2:
            continue
        gaps = (days.diff().dt.days - 1).dropna()
        missing_days = int(gaps.clip(lower=0).sum())
        if missing_days:
            total_missing_days += missing_days
            if len(gap_summaries) < DETAIL_EXAMPLES_LIMIT:
                gap_summaries.append(
                    {
                        "district": district,
                        "blood_group": group,
                        "missing_days": missing_days,
                    }
                )

    rejected_rows = [
        {"row_number": row_number, "reasons": sorted(reasons)}
        for row_number, reasons in sorted(rejected.items())
    ]
    rejected_examples = rejected_rows[:REJECTION_EXAMPLES_LIMIT]
    if len(accepted):
        start = accepted["timestamp"].min()
        end = accepted["timestamp"].max()
        date_range = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        }
    else:
        date_range = {"start": None, "end": None}

    present_output_columns = [
        column for column in OUTPUT_COLUMNS if column in accepted.columns
    ]
    accepted = accepted[present_output_columns].sort_values(
        ["timestamp", "district", "blood_group"], kind="stable"
    ).reset_index(drop=True)
    accepted["data_source"] = "entered"
    accepted["quality_flags"] = [
        quality_flags.get(f"{row.district}|{row.blood_group}", [])
        for row in accepted.itertuples(index=False)
    ]

    null_rates = {
        column: (
            float(accepted[column].isna().mean())
            if len(accepted)
            else 0.0
        )
        for column in ("timestamp", "district", "blood_group", "stock_units")
        + OPTIONAL_COLUMNS[:-1]
    }
    warnings = []
    if reported_sensitive_columns:
        warnings.append(
            {
                "code": "sensitive_columns_dropped",
                "message": "Potential patient-identifying columns were removed and were not retained.",
                "columns": [str(column) for column in reported_sensitive_columns],
            }
        )
    if has_facility_rows:
        warnings.append(
            {
                "code": "facility_rows_aggregated",
                "message": "Facility-level readings were summed to district and blood group; facility identifiers were removed.",
            }
        )
    if duplicate_report["duplicate_rows"]:
        warnings.append(
            {
                "code": "duplicate_readings_aggregated",
                "message": "Duplicate district/group/timestamp readings were aggregated.",
                "row_numbers": duplicate_report["duplicate_row_numbers"][
                    :REJECTION_EXAMPLES_LIMIT
                ],
                "count": duplicate_report["duplicate_rows"],
            }
        )
    over_stock_rows = [
        int(mapped.loc[index, "_source_row_number"])
        for index in mapped.index[stock_over_limit.fillna(False)]
    ]
    if over_stock_rows:
        warnings.append(
            {
                "code": "stock_above_10000",
                "message": "Stock values above 10,000 units require review.",
                "row_numbers": over_stock_rows[:REJECTION_EXAMPLES_LIMIT],
                "count": len(over_stock_rows),
            }
        )
    if district_suggestions:
        warnings.append(
            {
                "code": "district_match_confirmation_required",
                "message": "Some district names have suggestions but require user confirmation.",
                "suggestions": district_suggestions[:REJECTION_EXAMPLES_LIMIT],
            }
        )
    if rejected_rows:
        warnings.append(
            {
                "code": "rows_rejected",
                "message": "Some source rows did not pass validation.",
                "count": len(rejected_rows),
            }
        )

    report = {
        "input_rows": int(len(frame)),
        "accepted_rows": int(len(accepted)),
        "rejected_row_count": len(rejected_rows),
        "rejected_rows": rejected_examples,
        "rejected_rows_truncated": len(rejected_rows) > len(rejected_examples),
        "null_rates": null_rates,
        "date_range": date_range,
        "series_count": int(
            accepted[["district", "blood_group"]].drop_duplicates().shape[0]
        ),
        "series_with_less_than_14_days": insufficient_series,
        "quality_flags_by_series": quality_flags,
        "negative_unit_rows": sum(
            any(reason.startswith("negative_") for reason in reasons)
            for reasons in rejected.values()
        ),
        "stock_above_10000": {
            "count": len(over_stock_rows),
            "row_numbers": over_stock_rows[:REJECTION_EXAMPLES_LIMIT],
        },
        "future_date_rows": sum(
            "future_timestamp" in reasons for reasons in rejected.values()
        ),
        "duplicate_groups": duplicate_report["duplicate_groups"],
        "duplicate_rows_removed": duplicate_report["duplicate_rows"],
        "duplicate_row_numbers": duplicate_report["duplicate_row_numbers"][
            :REJECTION_EXAMPLES_LIMIT
        ],
        "gaps": {
            "series_with_gaps": len(gap_summaries),
            "missing_calendar_days": total_missing_days,
            "examples": gap_summaries,
            "examples_truncated": (
                len(gap_summaries) >= DETAIL_EXAMPLES_LIMIT
                and total_missing_days > sum(item["missing_days"] for item in gap_summaries)
            ),
        },
        "demand_estimated_rows": int(accepted["demand_estimated"].sum()),
        "insufficient_input_series": sum(
            "insufficient_inputs" in flags for flags in quality_flags.values()
        ),
        "data_source": "entered",
        "warnings": warnings,
    }
    return NormalizationResult(accepted, report)
