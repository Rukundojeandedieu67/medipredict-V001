"""Bounded readers for tabular uploads and preview-only documents."""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from medipredict.ingest.mapping import header_quality, suggest_mappings


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 500_000
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
PREVIEW_SUFFIXES = {".pdf", ".docx"}


class UploadReadError(ValueError):
    """A safe, user-correctable upload parsing error."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class UploadResult:
    filename: str
    file_type: str
    frame: pd.DataFrame | None
    mapping_suggestions: dict[str, dict[str, Any]]
    delimiter: str | None = None
    encoding: str | None = None
    header_row: int | None = None
    sheet_name: str | None = None
    sheet_names: list[str] | None = None
    preview_only: bool = False
    extracted_tables: list[dict[str, Any]] | None = None

    def public_metadata(self, preview_rows: int = 20) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "header_row": self.header_row,
            "sheet_name": self.sheet_name,
            "sheet_names": self.sheet_names or [],
            "preview_only": self.preview_only,
            "mapping_suggestions": self.mapping_suggestions,
            "preview": (
                self.frame.head(preview_rows).where(pd.notna(self.frame), None).to_dict("records")
                if self.frame is not None
                else []
            ),
            "extracted_tables": self.extracted_tables or [],
        }


def _header_signal(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).strip()


def _is_sensitive_header(value: Any) -> bool:
    words = set(_header_signal(str(value)).split())
    if words & {
        "patient",
        "nid",
        "phone",
        "telephone",
        "mobile",
        "address",
        "adresse",
        "passport",
    }:
        return True
    if {"national", "id"} <= words or {"identity", "number"} <= words:
        return True
    if words & {"name", "nom"} and "district" not in words:
        return True
    return False


def _decode_csv(content: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UploadReadError("encoding_error", "CSV encoding is not supported")


def _csv_delimiter(lines: list[str]) -> str:
    sample = "\n".join(lines[:30])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        counts = {
            delimiter: sum(line.count(delimiter) for line in lines[:30])
            for delimiter in (",", ";", "\t")
        }
        return max(counts, key=counts.get)


def _detect_header_row(rows: list[list[Any]], max_scan: int = 30) -> int:
    best_index = 0
    best_quality = (-1, -1.0)
    for index, row in enumerate(rows[:max_scan]):
        quality = header_quality(row)
        if quality > best_quality:
            best_index = index
            best_quality = quality
    return best_index


def _set_frame_metadata(
    frame: pd.DataFrame,
    filename: str,
    header_row: int | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    privacy_columns = [
        column for column in frame.columns if _is_sensitive_header(column)
    ]
    if privacy_columns:
        frame.drop(columns=privacy_columns, inplace=True)
    frame.attrs["source_filename"] = filename
    frame.attrs["source_row_offset"] = (header_row + 1) if header_row is not None else 1
    frame.attrs["header_row"] = header_row
    frame.attrs["sensitive_columns_dropped"] = [str(column) for column in privacy_columns]
    if len(frame) > MAX_ROWS:
        raise UploadReadError(
            "row_limit_exceeded",
            f"Upload exceeds the {MAX_ROWS:,}-row limit",
            {"max_rows": MAX_ROWS},
        )
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise UploadReadError(
            "duplicate_columns", "Upload contains duplicate column headers", {"columns": duplicates}
        )
    return frame


def _read_delimited(content: bytes, filename: str, extension: str) -> UploadResult:
    text, encoding = _decode_csv(content)
    lines = text.splitlines()
    delimiter = "\t" if extension == ".tsv" else _csv_delimiter(lines)
    try:
        parsed_rows = list(csv.reader(lines, delimiter=delimiter))
    except csv.Error as error:
        raise UploadReadError("csv_parse_error", f"Could not parse delimited file: {error}") from error
    header_row = _detect_header_row(parsed_rows)
    frame = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        header=header_row,
        engine="python",
        nrows=MAX_ROWS + 1,
    )
    frame = _set_frame_metadata(frame, filename, header_row + 1)
    return UploadResult(
        filename=filename,
        file_type=extension.lstrip("."),
        frame=frame,
        mapping_suggestions=suggest_mappings(frame.columns),
        delimiter=delimiter,
        encoding=encoding,
        header_row=header_row + 1,
    )


def _excel_sheet_score(sheet: pd.DataFrame) -> tuple[int, float]:
    scan = sheet.head(30)
    best = (-1, -1.0)
    for _, row in scan.iterrows():
        quality = header_quality(row.tolist())
        if quality > best:
            best = quality
    return best


def _read_excel(
    content: bytes, filename: str, selected_sheet: str | None
) -> UploadResult:
    try:
        engine = "xlrd" if Path(filename).suffix.lower() == ".xls" else "openpyxl"
        workbook = pd.ExcelFile(io.BytesIO(content), engine=engine)
    except Exception as error:
        raise UploadReadError("excel_parse_error", "Could not open Excel workbook") from error
    sheet_names = list(workbook.sheet_names)
    candidates: list[tuple[tuple[int, float], int, str, pd.DataFrame]] = []
    for name in sheet_names:
        if re.search(r"readme|notes?|instructions?|about|cover", name, re.I):
            continue
        raw = pd.read_excel(
            workbook,
            sheet_name=name,
            header=None,
            nrows=30,
        )
        header_index = _detect_header_row(raw.values.tolist())
        quality = _excel_sheet_score(raw)
        candidates.append((quality, header_index, name, raw))
    if selected_sheet is not None:
        if selected_sheet not in sheet_names:
            raise UploadReadError(
                "invalid_sheet",
                f"Requested sheet {selected_sheet!r} does not exist",
                {"available_sheets": sheet_names},
            )
        match = next((item for item in candidates if item[2] == selected_sheet), None)
        if match is None:
            raise UploadReadError(
                "notes_sheet_not_allowed",
                "README/notes sheets are excluded from upload ingestion",
                {"sheet_name": selected_sheet},
            )
    else:
        if not candidates:
            raise UploadReadError("no_data_sheet", "Workbook has no candidate data sheet")
        match = max(
            candidates,
            key=lambda item: (
                item[0][0],
                item[0][1],
                bool(re.search(r"stock|reading|inventory|donnee|donnees", item[2], re.I)),
                item[2],
            ),
        )
    _, header_index, chosen_name, _ = match
    frame = pd.read_excel(
        workbook,
        sheet_name=chosen_name,
        header=header_index,
        nrows=MAX_ROWS + 1,
    )
    frame = _set_frame_metadata(frame, filename, header_index + 1)
    return UploadResult(
        filename=filename,
        file_type=Path(filename).suffix.lower().lstrip("."),
        frame=frame,
        mapping_suggestions=suggest_mappings(frame.columns),
        header_row=header_index + 1,
        sheet_name=chosen_name,
        sheet_names=sheet_names,
    )


def _read_json(content: bytes, filename: str, json_lines: bool) -> UploadResult:
    try:
        text, encoding = _decode_csv(content)
        if json_lines:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(text)
            if isinstance(payload, dict):
                records = next(
                    (
                        payload[key]
                        for key in ("records", "data", "items", "readings")
                        if isinstance(payload.get(key), list)
                    ),
                    [payload],
                )
            else:
                records = payload
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise UploadReadError(
                "invalid_json_shape",
                "JSON uploads must contain an array of row objects",
            )
        frame = _set_frame_metadata(pd.DataFrame.from_records(records), filename)
    except UploadReadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UploadReadError(
            "json_parse_error", f"Could not parse JSON upload: {error}"
        ) from error
    return UploadResult(
        filename=filename,
        file_type="jsonl" if json_lines else "json",
        frame=frame,
        mapping_suggestions=suggest_mappings(frame.columns),
        encoding=encoding,
    )


def _read_parquet(content: bytes, filename: str) -> UploadResult:
    try:
        frame = pd.read_parquet(io.BytesIO(content), engine="pyarrow")
    except ImportError as error:
        raise UploadReadError(
            "parquet_dependency_missing",
            "Parquet reading requires the pyarrow dependency",
        ) from error
    except Exception as error:
        raise UploadReadError("parquet_parse_error", "Could not parse Parquet upload") from error
    frame = _set_frame_metadata(frame, filename)
    return UploadResult(
        filename=filename,
        file_type="parquet",
        frame=frame,
        mapping_suggestions=suggest_mappings(frame.columns),
    )


def _document_preview(content: bytes, filename: str, extension: str) -> UploadResult:
    extracted: list[dict[str, Any]] = []

    def safe_table(rows: list[list[Any]]) -> tuple[list[list[Any]], list[str]]:
        if not rows:
            return rows, []
        sensitive_indices = [
            index
            for index, header in enumerate(rows[0])
            if _is_sensitive_header(header)
        ]
        sensitive_headers = [str(rows[0][index]) for index in sensitive_indices]
        safe_rows = [
            [cell for index, cell in enumerate(row) if index not in sensitive_indices]
            for row in rows
        ]
        return safe_rows, sensitive_headers

    if extension == ".docx":
        try:
            from docx import Document

            document = Document(io.BytesIO(content))
        except ImportError as error:
            raise UploadReadError(
                "docx_dependency_missing",
                "DOCX table previews require the python-docx dependency",
            ) from error
        except Exception as error:
            raise UploadReadError("docx_parse_error", "Could not open DOCX document") from error
        for index, table in enumerate(document.tables, start=1):
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            rows, sensitive_columns = safe_table(rows)
            extracted.append(
                {
                    "table_index": index,
                    "rows": rows[:21],
                    "mapping_suggestions": suggest_mappings(rows[0] if rows else []),
                    "sensitive_columns_dropped": sensitive_columns,
                }
            )
    else:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as document:
                for page_number, page in enumerate(document.pages, start=1):
                    for table_index, rows in enumerate(page.extract_tables() or [], start=1):
                        rows, sensitive_columns = safe_table(rows)
                        extracted.append(
                            {
                                "page": page_number,
                                "table_index": table_index,
                                "rows": rows[:21],
                                "mapping_suggestions": suggest_mappings(
                                    rows[0] if rows else []
                                ),
                                "sensitive_columns_dropped": sensitive_columns,
                            }
                        )
        except ImportError as error:
            raise UploadReadError(
                "pdf_dependency_missing",
                "PDF table previews require the pdfplumber dependency",
            ) from error
        except Exception as error:
            raise UploadReadError("pdf_parse_error", "Could not extract PDF table preview") from error
    return UploadResult(
        filename=filename,
        file_type=extension.lstrip("."),
        frame=None,
        mapping_suggestions={},
        preview_only=True,
        extracted_tables=extracted,
    )


def read_upload(
    filename: str,
    content: bytes,
    sheet_name: str | None = None,
) -> UploadResult:
    """Parse a supported upload; PDF/DOCX contents are always preview-only."""
    if not filename or Path(filename).name != filename:
        raise UploadReadError("invalid_filename", "Upload must have a simple filename")
    if not isinstance(content, bytes) or not content:
        raise UploadReadError("empty_upload", "Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadReadError(
            "upload_too_large",
            f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            {"max_bytes": MAX_UPLOAD_BYTES},
        )
    extension = Path(filename).suffix.lower()
    if extension in {".csv", ".tsv"}:
        return _read_delimited(content, filename, extension)
    if extension in {".xlsx", ".xls"}:
        return _read_excel(content, filename, sheet_name)
    if extension == ".json":
        return _read_json(content, filename, json_lines=False)
    if extension == ".jsonl":
        return _read_json(content, filename, json_lines=True)
    if extension == ".parquet":
        return _read_parquet(content, filename)
    if extension in {".pdf", ".docx"}:
        return _document_preview(content, filename, extension)
    raise UploadReadError(
        "unsupported_file_type",
        "Supported file types are CSV, TSV, Excel, JSON, JSONL, Parquet, PDF, and DOCX",
        {"extension": extension},
    )
