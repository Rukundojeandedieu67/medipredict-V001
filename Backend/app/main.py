"""FastAPI contract for the MediPredict upload and scoring workflow."""

from __future__ import annotations

import csv
import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from medipredict.ingest.mapping import CANONICAL_COLUMNS, REQUIRED_COLUMNS, suggest_mappings
from medipredict.ingest.normalize import MappingError, normalize_dataframe
from medipredict.ingest.readers import MAX_UPLOAD_BYTES, UploadReadError, read_upload
from medipredict.inference.bundle import DEFAULT_BUNDLE_DIR, load_bundle
from medipredict.inference.score import score

from app.models import (
    CanonicalField,
    CanonicalSchemaResponse,
    DataInventoryResponse,
    DeleteDatasetResponse,
    DatasetUploadResponse,
    DistrictRecord,
    DistrictsResponse,
    ErrorDetail,
    ErrorEnvelope,
    HealthResponse,
    MappingRequest,
    MappingResponse,
    ModelInfoResponse,
    PredictReading,
    PredictRequest,
    PredictResponse,
    ResultsResponse,
    ScoreDatasetResponse,
    ScoreRecord,
    ScoreRequest,
)
from app.storage import DatasetNotFoundError, DatasetStore

LOGGER = logging.getLogger("medipredict.api")
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    upload_dir: Path
    cors_origins: tuple[str, ...]
    rate_limit_per_minute: int

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        )
        return cls(
            api_key=os.getenv("MEDIPREDICT_API_KEY") or os.getenv("API_KEY"),
            upload_dir=Path(
                os.getenv("MEDIPREDICT_UPLOAD_DIR", str(BACKEND_DIR / "data" / "uploads"))
            ),
            cors_origins=origins,
            rate_limit_per_minute=max(
                1, int(os.getenv("MEDIPREDICT_RATE_LIMIT_PER_MINUTE", "120"))
            ),
        )


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _now() -> datetime:
    return datetime.now(UTC)


def _model_version() -> str:
    try:
        return str(json.loads((DEFAULT_BUNDLE_DIR / "manifest.json").read_text())["model_version"])
    except (OSError, KeyError, json.JSONDecodeError):
        return "unavailable"


def _json_records(frame: pd.DataFrame, count: int = 20) -> list[dict[str, Any]]:
    return json.loads(
        frame.head(count).to_json(orient="records", date_format="iso", force_ascii=False)
    )


def _suggested_mapping(suggestions: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        canonical: str(item["column"])
        for canonical, item in suggestions.items()
        if item.get("column") is not None
    }


def _preliminary_report(
    frame: pd.DataFrame | None, suggestions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if frame is None:
        return {
            "status": "preview_only",
            "message": "Document tables require human review and cannot be ingested directly.",
        }
    mapping = _suggested_mapping(suggestions)
    missing = [column for column in REQUIRED_COLUMNS if column not in mapping]
    if missing:
        return {"status": "mapping_required", "missing_required_mappings": missing}
    try:
        result = normalize_dataframe(frame, mapping)
    except (ValueError, MappingError) as error:
        return {"status": "mapping_review_required", "message": str(error)}
    return {"status": "suggested_mapping_preview", **result.report}


def _api_error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorEnvelope(
            error=ErrorDetail(code=code, message=message, details=details or {}),
            data_source="entered",
            as_of=_now(),
        ).model_dump(mode="json"),
        headers={"X-Request-ID": getattr(request.state, "request_id", str(uuid.uuid4()))},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings or Settings.from_environment()
    store = DatasetStore(configuration.upload_dir)

    @asynccontextmanager
    async def retention_lifespan(_app: FastAPI):
        store.cleanup_expired()
        cleanup_task = asyncio.create_task(run_retention_cleanup(store))
        try:
            yield
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="MediPredict API",
        version="0.1.0",
        description=(
            "Typed upload, validation and simulated-model watch-list scoring API. "
            "Scores are decision-support only and are not operationally validated."
        ),
        lifespan=retention_lifespan,
    )
    app.state.settings = configuration
    app.state.store = store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configuration.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["Content-Disposition", "X-Request-ID"],
    )

    class RequestGuardMiddleware(BaseHTTPMiddleware):
        def __init__(self, app: Any):
            super().__init__(app)
            self.requests: dict[str, deque[float]] = defaultdict(deque)

        async def dispatch(self, request: Request, call_next: Any) -> Response:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit():
                if int(content_length) > MAX_UPLOAD_BYTES + 1024 * 1024:
                    return _api_error_response(
                        request,
                        413,
                        "request_too_large",
                        "Request exceeds the 26 MB request limit.",
                        {"max_bytes": MAX_UPLOAD_BYTES + 1024 * 1024},
                    )
            now = time.monotonic()
            client = request.client.host if request.client else "unknown"
            for address, queue in list(self.requests.items()):
                while queue and queue[0] <= now - 60:
                    queue.popleft()
                if not queue:
                    self.requests.pop(address, None)
            recent = self.requests[client]
            if len(recent) >= configuration.rate_limit_per_minute:
                return _api_error_response(
                    request,
                    429,
                    "rate_limit_exceeded",
                    "Too many requests; try again shortly.",
                    {"limit_per_minute": configuration.rate_limit_per_minute},
                )
            recent.append(now)
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

    app.add_middleware(RequestGuardMiddleware)

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
        return _api_error_response(
            request, error.status_code, error.code, error.message, error.details
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return _api_error_response(
            request,
            422,
            "request_validation_error",
            "Request data failed validation.",
            {
                "errors": [
                    {
                        "loc": item.get("loc", []),
                        "msg": item.get("msg", "Invalid value"),
                        "type": item.get("type", "value_error"),
                    }
                    for item in error.errors()
                ]
            },
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
        code_by_status = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            413: "request_too_large",
            429: "rate_limit_exceeded",
            501: "not_implemented",
        }
        detail = error.detail
        message = detail if isinstance(detail, str) else "Request could not be completed."
        return _api_error_response(
            request,
            error.status_code,
            code_by_status.get(error.status_code, "http_error"),
            message,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error request_id=%s", request.state.request_id)
        return _api_error_response(
            request, 500, "internal_error", "The server could not complete the request."
        )

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def require_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
        expected = configuration.api_key
        if not expected:
            raise APIError(
                503,
                "api_key_not_configured",
                "Set MEDIPREDICT_API_KEY before using protected API endpoints.",
            )
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise APIError(401, "invalid_api_key", "A valid X-API-Key is required.")

    router = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(require_api_key)],
        responses={
            401: {"model": ErrorEnvelope, "description": "Invalid or missing API key"},
            422: {"model": ErrorEnvelope, "description": "Request validation error"},
            429: {"model": ErrorEnvelope, "description": "Rate limit exceeded"},
            500: {"model": ErrorEnvelope, "description": "Internal server error"},
        },
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        try:
            store.root.stat()
            storage_status: Literal["ok", "unavailable"] = "ok"
        except OSError:
            storage_status = "unavailable"
        return HealthResponse(
            status="ok",
            model_version=_model_version(),
            storage_status=storage_status,
            data_source="simulated",
            as_of=_now(),
        )

    @router.get("/model/info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        try:
            bundle = load_bundle()
        except (OSError, ValueError, KeyError) as error:
            raise APIError(503, "model_unavailable", "The frozen model bundle is unavailable.") from error
        card_path = DEFAULT_BUNDLE_DIR / "MODEL_CARD.md"
        limitations = card_path.read_text(encoding="utf-8")
        return ModelInfoResponse(
            model_version=bundle["manifest"]["model_version"],
            trained_on="simulated",
            metrics=bundle["metrics"],
            limitations=limitations,
            feature_list=bundle["feature_schema"]["feature_order"],
            data_source="simulated",
            as_of=_now(),
        )

    @router.get("/templates/readings.csv")
    async def download_readings_template() -> Response:
        path = REPOSITORY_ROOT / "data" / "templates" / "readings_template.csv"
        if not path.is_file():
            raise APIError(404, "template_not_found", "The readings template is unavailable.")
        return Response(
            path.read_bytes(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="readings_template.csv"',
                "X-Data-Source": "derived",
                "X-As-Of": _now().isoformat(),
            },
        )

    @router.get("/templates/schema", response_model=CanonicalSchemaResponse)
    async def canonical_schema() -> CanonicalSchemaResponse:
        units = {
            "timestamp": "UTC date/time; daily readings are end-of-day",
            "district": None,
            "blood_group": None,
            "stock_units": "units",
            "demand_units": "units",
            "supply_received": "units",
            "expired_units": "units",
            "unmet_demand_units": "units",
            "facility_id": None,
        }
        fields = [
            CanonicalField(
                name=name,
                required=name in REQUIRED_COLUMNS,
                dtype="string" if name in {"timestamp", "district", "blood_group", "facility_id"} else "number",
                unit=units[name],
            )
            for name in CANONICAL_COLUMNS
        ]
        return CanonicalSchemaResponse(
            fields=fields,
            notes=[
                "Stock is closing stock at end of day.",
                "Facility-level rows are aggregated to district/group/day.",
                "Every uploaded row remains entered data and is not added to training.",
            ],
            data_source="derived",
            as_of=_now(),
        )

    @router.get("/data-inventory", response_model=DataInventoryResponse)
    async def data_inventory() -> DataInventoryResponse:
        path = REPOSITORY_ROOT / "data" / "templates" / "data_inventory_and_gaps.csv"
        if not path.is_file():
            raise APIError(404, "inventory_not_found", "The data inventory file is unavailable.")
        try:
            rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
        except (OSError, pd.errors.ParserError) as error:
            raise APIError(500, "inventory_read_error", "Could not read the data inventory.") from error
        return DataInventoryResponse(rows=rows, data_source="derived", as_of=_now())

    @router.post("/datasets", response_model=DatasetUploadResponse)
    async def create_dataset(
        file: UploadFile = File(...),
        sheet_name: str | None = None,
    ) -> DatasetUploadResponse:
        filename = Path(file.filename or "").name
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise APIError(
                413,
                "upload_too_large",
                "Upload exceeds the 25 MB limit.",
                {"max_bytes": MAX_UPLOAD_BYTES},
            )
        try:
            upload = read_upload(filename, content, sheet_name)
        except UploadReadError as error:
            raise APIError(422, error.code, str(error), error.details) from error
        frame = upload.frame
        metadata = {
            "filename": filename,
            "file_type": upload.file_type,
            "delimiter": upload.delimiter,
            "encoding": upload.encoding,
            "header_row": upload.header_row,
            "source_row_offset": (
                frame.attrs.get("source_row_offset", 2) if frame is not None else 2
            ),
            "detected_sheet": upload.sheet_name,
            "available_sheets": upload.sheet_names or [],
            "preview_only": upload.preview_only,
            "mapping_suggestions": upload.mapping_suggestions,
            "extracted_tables": upload.extracted_tables or [],
            "mapping": None,
            "validation_report": None,
            "sensitive_columns_dropped": (
                frame.attrs.get("sensitive_columns_dropped", []) if frame is not None else []
            ),
        }
        if frame is not None:
            metadata["validation_report"] = _preliminary_report(
                frame, upload.mapping_suggestions
            )
        dataset_id = store.create(metadata, frame)
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=filename,
            file_type=upload.file_type,
            delimiter=upload.delimiter,
            encoding=upload.encoding,
            detected_sheet=upload.sheet_name,
            available_sheets=upload.sheet_names or [],
            preview_only=upload.preview_only,
            mapping_suggestions=upload.mapping_suggestions,
            preview=_json_records(frame) if frame is not None else [],
            extracted_tables=upload.extracted_tables or [],
            validation_report=metadata["validation_report"] or {},
            data_source="entered",
            as_of=_now(),
        )

    @router.put("/datasets/{dataset_id}/mapping", response_model=MappingResponse)
    async def confirm_mapping(
        dataset_id: str, body: MappingRequest
    ) -> MappingResponse:
        try:
            metadata = store.get_metadata(dataset_id)
            if metadata["preview_only"]:
                raise APIError(
                    422,
                    "preview_only",
                    "PDF/DOCX tables cannot be ingested; provide a reviewed CSV or spreadsheet.",
                )
            frame = store.get_frame(dataset_id)
            normalized = normalize_dataframe(frame, body.mapping)
        except DatasetNotFoundError as error:
            raise APIError(404, "dataset_not_found", "Dataset was not found or has expired.") from error
        except MappingError as error:
            raise APIError(422, "mapping_invalid", str(error)) from error
        except ValueError as error:
            raise APIError(422, "normalization_failed", str(error)) from error
        metadata["mapping"] = body.mapping
        metadata["validation_report"] = normalized.report
        store.save_metadata(dataset_id, metadata)
        return MappingResponse(
            dataset_id=dataset_id,
            status="mapping_confirmed",
            validation_report=normalized.report,
            preview=_json_records(normalized.data),
            data_source="entered",
            as_of=_now(),
        )

    @router.post(
        "/datasets/{dataset_id}/score", response_model=ScoreDatasetResponse
    )
    async def score_dataset(
        dataset_id: str, body: ScoreRequest
    ) -> ScoreDatasetResponse:
        try:
            metadata = store.get_metadata(dataset_id)
            if metadata["preview_only"]:
                raise APIError(422, "preview_only", "Preview-only documents cannot be scored.")
            mapping = metadata.get("mapping")
            if not mapping:
                raise APIError(409, "mapping_required", "Confirm a column mapping before scoring.")
            normalized = normalize_dataframe(store.get_frame(dataset_id), mapping)
            if normalized.data.empty:
                raise APIError(
                    422, "no_valid_rows", "No valid canonical reading rows remain after validation."
                )
            result = score(
                normalized.data,
                as_of=body.as_of,
                horizons=tuple(body.horizons),
                top_k=body.top_k,
            )
        except DatasetNotFoundError as error:
            raise APIError(404, "dataset_not_found", "Dataset was not found or has expired.") from error
        except (MappingError, ValueError) as error:
            raise APIError(422, "scoring_input_invalid", str(error)) from error
        store.save_results(dataset_id, result)
        return ScoreDatasetResponse(
            dataset_id=dataset_id,
            model_version=result["model_version"],
            trained_on=result["trained_on"],
            drift=result["drift"],
            total=len(result["results"]),
            data_source=result["data_source"],
            as_of=pd.Timestamp(result["as_of"]).to_pydatetime(),
        )

    @router.get("/datasets/{dataset_id}/results", response_model=ResultsResponse)
    async def get_dataset_results(
        dataset_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
        district: str | None = None,
        blood_group: str | None = None,
        confidence: Literal["high", "medium", "low", "insufficient_history"] | None = None,
        in_watchlist: bool | None = None,
        sort_by: Literal["risk_24h", "risk_48h", "rank", "rule_days_of_cover"] = "risk_24h",
        descending: bool = True,
    ) -> ResultsResponse:
        try:
            result = store.get_results(dataset_id)
        except DatasetNotFoundError as error:
            raise APIError(404, "dataset_not_found", "Dataset was not found or has expired.") from error
        if result is None:
            raise APIError(409, "results_not_ready", "Score the dataset before requesting results.")
        rows = result["results"]
        if district:
            rows = [row for row in rows if row["district"].casefold() == district.casefold()]
        if blood_group:
            rows = [row for row in rows if row["blood_group"] == blood_group.upper()]
        if confidence:
            rows = [row for row in rows if row["confidence"] == confidence]
        if in_watchlist is not None:
            rows = [row for row in rows if row["in_watchlist"] is in_watchlist]
        def result_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
            value = row.get(sort_by)
            ordered_value = 0 if value is None else value
            if descending and isinstance(ordered_value, (int, float)):
                ordered_value = -ordered_value
            return (
                value is None,
                ordered_value,
                row["district"],
                row["blood_group"],
            )

        rows.sort(key=result_sort_key)
        return ResultsResponse(
            dataset_id=dataset_id,
            total=len(rows),
            offset=offset,
            limit=limit,
            results=rows[offset : offset + limit],
            data_source=result["data_source"],
            as_of=pd.Timestamp(result["as_of"]).to_pydatetime(),
        )

    @router.get("/datasets/{dataset_id}/results/export.csv")
    async def export_dataset_results(dataset_id: str) -> StreamingResponse:
        try:
            result = store.get_results(dataset_id)
        except DatasetNotFoundError as error:
            raise APIError(404, "dataset_not_found", "Dataset was not found or has expired.") from error
        if result is None:
            raise APIError(409, "results_not_ready", "Score the dataset before exporting results.")
        stream = StringIO()
        rows = result["results"]
        fieldnames = list(rows[0].keys()) if rows else [
            field for field in ScoreRecord.model_fields
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="watchlist-{dataset_id}.csv"',
                "X-Data-Source": result["data_source"],
                "X-As-Of": result["as_of"],
            },
        )

    @router.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest) -> PredictResponse:
        source_rows = [row.model_dump(exclude_none=True) for row in body.rows]
        frame = pd.DataFrame(source_rows)
        mapping = {
            field: field
            for field in CANONICAL_COLUMNS
            if field in frame.columns and (field in REQUIRED_COLUMNS or frame[field].notna().any())
        }
        try:
            normalized = normalize_dataframe(frame, mapping)
            if normalized.data.empty:
                raise APIError(422, "no_valid_rows", "No valid canonical reading rows were provided.")
            result = score(
                normalized.data,
                as_of=body.as_of,
                horizons=tuple(body.horizons),
                top_k=body.top_k,
            )
        except (MappingError, ValueError) as error:
            raise APIError(422, "prediction_input_invalid", str(error)) from error
        return PredictResponse(
            model_version=result["model_version"],
            trained_on=result["trained_on"],
            drift=result["drift"],
            total=len(result["results"]),
            validation_report=normalized.report,
            results=result["results"],
            data_source=result["data_source"],
            as_of=pd.Timestamp(result["as_of"]).to_pydatetime(),
        )

    @router.get("/reference/districts", response_model=DistrictsResponse)
    async def reference_districts() -> DistrictsResponse:
        path = REPOSITORY_ROOT / "data" / "reference" / "dim_district.csv"
        try:
            frame = pd.read_csv(path)
            districts = [
                DistrictRecord(
                    district=str(row.district),
                    pcode=str(row.adm2_pcode),
                    province=str(row.province),
                    lat=float(row.lat),
                    lon=float(row.lon),
                )
                for row in frame.itertuples(index=False)
            ]
        except (OSError, pd.errors.ParserError, AttributeError, ValueError) as error:
            raise APIError(
                500, "district_reference_error", "Could not load the district reference data."
            ) from error
        return DistrictsResponse(
            districts=districts, data_source="derived", as_of=_now()
        )

    @router.delete("/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: str) -> DeleteDatasetResponse:
        try:
            store.delete(dataset_id)
        except DatasetNotFoundError as error:
            raise APIError(404, "dataset_not_found", "Dataset was not found or has expired.") from error
        return DeleteDatasetResponse(
            deleted=True,
            dataset_id=dataset_id,
            data_source="entered",
            as_of=_now(),
        )

    @router.post(
        "/readings",
        status_code=501,
        responses={501: {"model": ErrorEnvelope, "description": "Reserved for Phase 4"}},
    )
    async def readings_stub() -> None:
        raise HTTPException(status_code=501, detail="Reading ingestion is reserved for Phase 4.")

    @router.get(
        "/inventory/summary",
        status_code=501,
        responses={501: {"model": ErrorEnvelope, "description": "Reserved for Phase 4"}},
    )
    async def inventory_stub() -> None:
        raise HTTPException(status_code=501, detail="Live inventory is reserved for Phase 4.")

    @router.get(
        "/stream",
        status_code=501,
        responses={501: {"model": ErrorEnvelope, "description": "Reserved for Phase 4"}},
    )
    async def stream_stub() -> None:
        raise HTTPException(status_code=501, detail="Live streaming is reserved for Phase 4.")

    app.include_router(router)
    return app


async def run_retention_cleanup(store: DatasetStore) -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            store.cleanup_expired()
        except OSError:
            LOGGER.exception("Scheduled dataset-retention cleanup failed")
            raise


app = create_app()
