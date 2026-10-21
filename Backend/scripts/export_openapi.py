"""Export the backend OpenAPI contract to the repository root."""

from __future__ import annotations

import json
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from app.main import app

repository_root = backend_root.parent
output = repository_root / "openapi.json"
output.write_text(
    json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"OpenAPI contract exported to {output}")
