import io
import json
from pathlib import Path

import pandas as pd
import pytest
from docx import Document
from pypdf import PdfWriter

from medipredict.ingest.mapping import REQUIRED_COLUMNS, suggest_mappings
from medipredict.ingest.normalize import MappingError, normalize_dataframe
from medipredict.ingest.readers import UploadReadError, read_upload
from medipredict.inference.score import score


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "sample_readings_30d_SIMULATED.csv"


def _mapping() -> dict[str, str]:
    return {
        "timestamp": "jour",
        "district": "nom_district",
        "blood_group": "groupe_sanguin",
        "stock_units": "stock",
        "demand_units": "consommation",
        "supply_received": "livraison",
        "expired_units": "perime",
        "unmet_demand_units": "manquant",
    }


def _french_rows() -> bytes:
    return (
        "Rapport de stock\n"
        "jour;nom_district;groupe_sanguin;stock;consommation;livraison;perime;manquant\n"
        "2026-01-01;Huye;O POS;10;3;4;0;0\n"
        "2026-01-02;Huye;Opos;11;3;4;0;0\n"
        "2026-01-02;Huye;O Rh+;12;2;3;0;0\n"
        "2026-01-02;Huye;O+;13;1;2;0;0\n"
        "2026-01-03;Huye;A NEG;-1;1;0;0;1\n"
        "2099-01-01;Huye;B+;5;1;0;0;0\n"
    ).encode("utf-8")


def _uploaded_french() -> pd.DataFrame:
    parsed = read_upload("messy_french_semicolon.csv", _french_rows())
    assert parsed.delimiter == ";"
    assert parsed.encoding == "utf-8-sig"
    assert parsed.header_row == 2
    assert parsed.mapping_suggestions["timestamp"]["column"] == "jour"
    return parsed.frame


def test_french_semicolon_header_and_normalization_report_rows():
    result = normalize_dataframe(
        _uploaded_french(),
        _mapping(),
        now=pd.Timestamp("2026-10-21T00:00:00Z"),
    )
    report = result.report
    assert report["duplicate_rows_removed"] == 2
    assert report["duplicate_row_numbers"] == [5, 6]
    assert report["negative_unit_rows"] == 1
    assert report["future_date_rows"] == 1
    assert {"row_number": 8, "reasons": ["future_timestamp"]} in report["rejected_rows"]
    assert {"row_number": 7, "reasons": ["negative_stock_units"]} in report["rejected_rows"]
    duplicate = result.data.loc[result.data["timestamp"] == "2026-01-02"].iloc[0]
    assert duplicate["stock_units"] == 13
    assert duplicate["demand_units"] == 6
    assert duplicate["supply_received"] == 9
    assert result.data["blood_group"].tolist() == ["O+", "O+"]


def test_headerless_json_formats_and_mapping_confirmation():
    rows = [
        {
            "day": "2026-01-01",
            "district": "Huye",
            "blood type": "a neg",
            "closing stock": 12,
        }
    ]
    json_result = read_upload("readings.json", json.dumps(rows).encode("utf-8"))
    jsonl_result = read_upload(
        "readings.jsonl",
        b"\n".join(json.dumps(row).encode("utf-8") for row in rows),
    )
    assert json_result.mapping_suggestions["stock_units"]["column"] == "closing stock"
    assert jsonl_result.mapping_suggestions["blood_group"]["column"] == "blood type"
    suggestions = suggest_mappings(rows[0].keys())
    mapping = {
        canonical: item["column"]
        for canonical, item in suggestions.items()
        if item["column"]
    }
    with pytest.raises(MappingError, match="required"):
        normalize_dataframe(json_result.frame, {})
    normalized = normalize_dataframe(json_result.frame, mapping)
    assert normalized.data.iloc[0]["blood_group"] == "A-"
    assert normalized.data.iloc[0]["data_source"] == "entered"


def test_excel_chooses_stock_sheet_and_skips_readme():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([["Instructions"], ["Do not load"]]).to_excel(
            writer, sheet_name="README", index=False, header=False
        )
        pd.DataFrame(
            {
                "timestamp": ["2026-01-01"],
                "district": ["Huye"],
                "blood_group": ["O+"],
                "stock_units": [20],
            }
        ).to_excel(writer, sheet_name="Stock", index=False)
    result = read_upload("messy_two_sheets.xlsx", buffer.getvalue())
    assert result.sheet_names == ["README", "Stock"]
    assert result.sheet_name == "Stock"
    assert len(result.frame) == 1
    assert result.frame.iloc[0]["stock_units"] == 20


def test_parquet_is_supported():
    buffer = io.BytesIO()
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "district": ["Huye"],
            "blood_group": ["O+"],
            "stock_units": [8],
        }
    ).to_parquet(buffer, index=False)
    result = read_upload("readings.parquet", buffer.getvalue())
    assert result.file_type == "parquet"
    assert result.frame.iloc[0]["district"] == "Huye"


def test_pdf_and_docx_tables_are_preview_only():
    document = Document()
    table = document.add_table(rows=2, cols=5)
    for cell, value in zip(
        table.rows[0].cells,
        ("jour", "nom_district", "groupe sanguin", "stock", "patient_name"),
        strict=True,
    ):
        cell.text = value
    for cell, value in zip(
        table.rows[1].cells,
        ("2026-01-01", "Huye", "O+", "4", "REDACTED"),
        strict=True,
    ):
        cell.text = value
    docx_bytes = io.BytesIO()
    document.save(docx_bytes)
    docx_result = read_upload("table.docx", docx_bytes.getvalue())
    assert docx_result.preview_only
    assert docx_result.frame is None
    assert docx_result.extracted_tables[0]["mapping_suggestions"]["stock_units"]["column"] == "stock"
    assert docx_result.extracted_tables[0]["rows"][0] == [
        "jour",
        "nom_district",
        "groupe sanguin",
        "stock",
    ]
    assert "REDACTED" not in json.dumps(docx_result.public_metadata())
    assert docx_result.extracted_tables[0]["sensitive_columns_dropped"] == [
        "patient_name"
    ]

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_result = read_upload("table.pdf", pdf_bytes.getvalue())
    assert pdf_result.preview_only
    assert pdf_result.frame is None


def test_privacy_guard_removes_patient_columns_before_preview_and_output():
    content = (
        "timestamp,district,blood_group,stock_units,patient_name,national_id,phone,address\n"
        "2026-01-01,Huye,O+,4,REDACTED,REDACTED,REDACTED,REDACTED\n"
    ).encode()
    parsed = read_upload("with_patient_columns_FAKE.csv", content)
    assert not {"patient_name", "national_id", "phone", "address"} & set(parsed.frame.columns)
    assert "REDACTED" not in json.dumps(parsed.public_metadata())
    mapping = {
        "timestamp": "timestamp",
        "district": "district",
        "blood_group": "blood_group",
        "stock_units": "stock_units",
    }
    normalized = normalize_dataframe(parsed.frame, mapping)
    assert any(
        warning["code"] == "sensitive_columns_dropped"
        for warning in normalized.report["warnings"]
    )
    assert not {"patient_name", "national_id", "phone", "address"} & set(
        normalized.data.columns
    )


def test_facility_rows_sum_stock_and_emit_warning():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "district": ["Huye", "Huye"],
            "blood_group": ["O+", "O+"],
            "stock_units": [4, 6],
            "demand_units": [2, 3],
            "supply_received": [1, 2],
            "facility_id": ["F1", "F2"],
        }
    )
    mapping = {
        "timestamp": "timestamp",
        "district": "district",
        "blood_group": "blood_group",
        "stock_units": "stock_units",
        "demand_units": "demand_units",
        "supply_received": "supply_received",
        "facility_id": "facility_id",
    }
    result = normalize_dataframe(frame, mapping)
    assert len(result.data) == 1
    assert result.data.iloc[0]["stock_units"] == 10
    assert result.data.iloc[0]["demand_units"] == 5
    assert any(
        warning["code"] == "facility_rows_aggregated"
        for warning in result.report["warnings"]
    )
    assert "facility_id" not in result.data


def test_missing_demand_is_estimated_when_supply_exists():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "district": ["Huye", "Huye"],
            "blood_group": ["O+", "O+"],
            "stock_units": [10, 9],
            "supply_received": [2, 3],
            "expired_units": [0, 1],
        }
    )
    mapping = {
        "timestamp": "timestamp",
        "district": "district",
        "blood_group": "blood_group",
        "stock_units": "stock_units",
        "supply_received": "supply_received",
        "expired_units": "expired_units",
    }
    result = normalize_dataframe(frame, mapping)
    assert result.data["demand_units"].tolist() == [2, 3]
    assert result.report["demand_estimated_rows"] == 2
    assert "demand_estimated" in result.report["quality_flags_by_series"]["Huye|O+"]


def test_no_demand_or_supply_is_insufficient_and_scores_rule_only():
    sample = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-01", periods=20, freq="D").astype(str),
            "district": ["Huye"] * 20,
            "blood_group": ["O+"] * 20,
            "stock_units": [20] * 20,
        }
    )
    mapping = {key: key for key in REQUIRED_COLUMNS}
    normalized = normalize_dataframe(sample, mapping)
    assert normalized.report["insufficient_input_series"] == 1
    assert "insufficient_inputs" in normalized.report["quality_flags_by_series"]["Huye|O+"]
    scored = score(normalized.data)
    row = scored["results"][0]
    assert row["risk_24h"] is None and row["risk_48h"] is None
    assert "insufficient_inputs" in row["quality_flags"]


def test_sample_30_day_upload_scores_all_240_series():
    parsed = read_upload(SAMPLE.name, SAMPLE.read_bytes())
    suggestions = parsed.mapping_suggestions
    mapping = {
        canonical: suggestions[canonical]["column"]
        for canonical in suggestions
        if suggestions[canonical]["column"] is not None
    }
    normalized = normalize_dataframe(parsed.frame, mapping)
    assert normalized.report["series_count"] == 240
    assert normalized.report["series_with_less_than_14_days"] == 0
    results = score(normalized.data, top_k=10)
    assert len(results["results"]) == 240
    assert all(row["confidence"] in {"medium", "high"} for row in results["results"])


def test_upload_size_and_extension_limits():
    with pytest.raises(UploadReadError, match="25 MB"):
        read_upload("large.csv", b"a" * (25 * 1024 * 1024 + 1))
    with pytest.raises(UploadReadError, match="Supported file types"):
        read_upload("image.png", b"\x89PNG")
