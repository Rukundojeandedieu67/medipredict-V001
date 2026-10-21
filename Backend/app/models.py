"""Typed API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DataSource = Literal["real", "derived", "simulated", "entered"]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetaResponse(APIModel):
    data_source: DataSource
    as_of: datetime


class ErrorDetail(APIModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(APIModel):
    error: ErrorDetail
    data_source: DataSource = "entered"
    as_of: datetime


class HealthResponse(MetaResponse):
    status: Literal["ok"]
    model_version: str
    storage_status: Literal["ok", "unavailable"]


class ModelInfoResponse(MetaResponse):
    model_version: str
    trained_on: DataSource
    metrics: dict[str, Any]
    limitations: str
    feature_list: list[str]


class CanonicalField(APIModel):
    name: str
    required: bool
    dtype: str
    unit: str | None = None


class CanonicalSchemaResponse(MetaResponse):
    fields: list[CanonicalField]
    notes: list[str]


class DatasetUploadResponse(MetaResponse):
    dataset_id: str
    filename: str
    file_type: str
    delimiter: str | None = None
    encoding: str | None = None
    detected_sheet: str | None = None
    available_sheets: list[str]
    preview_only: bool
    mapping_suggestions: dict[str, dict[str, Any]]
    preview: list[dict[str, Any]]
    extracted_tables: list[dict[str, Any]]
    validation_report: dict[str, Any]


class MappingRequest(APIModel):
    mapping: dict[str, str]


class MappingResponse(MetaResponse):
    dataset_id: str
    status: Literal["mapping_confirmed"]
    validation_report: dict[str, Any]
    preview: list[dict[str, Any]]


class ScoreRequest(APIModel):
    as_of: datetime | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    horizons: list[Literal["24h", "48h"]] = Field(
        default_factory=lambda: ["24h", "48h"], min_length=1, max_length=2
    )


class ScoreRecord(APIModel):
    district: str
    blood_group: str
    risk_24h: float | None
    risk_48h: float | None
    rule_days_of_cover: float | None
    rank: int = Field(ge=1)
    in_watchlist: bool
    top_drivers: list[str]
    confidence: Literal["high", "medium", "low", "insufficient_history"]
    quality_flags: list[str]
    model_version: str
    data_source: DataSource
    low_confidence_group: bool


class ScoreDatasetResponse(MetaResponse):
    dataset_id: str
    model_version: str
    trained_on: DataSource
    drift: dict[str, Any]
    total: int


class ResultsResponse(MetaResponse):
    dataset_id: str
    total: int
    offset: int
    limit: int
    results: list[ScoreRecord]


class PredictReading(APIModel):
    timestamp: str = Field(min_length=1, max_length=64)
    district: str = Field(min_length=1, max_length=128)
    blood_group: str = Field(min_length=1, max_length=16)
    stock_units: float
    demand_units: float | None = None
    supply_received: float | None = None
    expired_units: float | None = None
    unmet_demand_units: float | None = None
    facility_id: str | None = Field(default=None, max_length=128)


class PredictRequest(APIModel):
    rows: list[PredictReading] = Field(min_length=1, max_length=5000)
    as_of: datetime | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    horizons: list[Literal["24h", "48h"]] = Field(
        default_factory=lambda: ["24h", "48h"], min_length=1, max_length=2
    )


class PredictResponse(MetaResponse):
    model_version: str
    trained_on: DataSource
    drift: dict[str, Any]
    total: int
    validation_report: dict[str, Any]
    results: list[ScoreRecord]


class DistrictRecord(APIModel):
    district: str
    pcode: str
    province: str
    lat: float
    lon: float


class DistrictsResponse(MetaResponse):
    districts: list[DistrictRecord]


class DataInventoryResponse(MetaResponse):
    rows: list[dict[str, str]]


class DeleteDatasetResponse(MetaResponse):
    deleted: bool
    dataset_id: str


class StubResponse(MetaResponse):
    error: ErrorDetail
