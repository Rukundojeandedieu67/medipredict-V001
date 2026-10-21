"""Safe tabular upload parsing, mapping suggestions, and validation."""

from medipredict.ingest.mapping import suggest_mappings
from medipredict.ingest.normalize import NormalizationResult, normalize_dataframe
from medipredict.ingest.readers import UploadReadError, UploadResult, read_upload

__all__ = [
    "NormalizationResult",
    "UploadReadError",
    "UploadResult",
    "normalize_dataframe",
    "read_upload",
    "suggest_mappings",
]
