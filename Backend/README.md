# MediPredict backend

The backend is intentionally isolated in `Backend\` with its own Python 3.11
virtual environment at `Backend\.venv\`. It installs the root project in editable
mode so the API reuses the verified `medipredict.ingest` and
`medipredict.inference` packages rather than maintaining duplicate logic.

From the repository root, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\Backend\tasks.ps1
Initialize-BackendEnvironment
Start-Backend
```

The execution-policy command affects only the current PowerShell process.
The first setup creates `Backend\.venv\`, installs both the project and API
dependencies, and writes a local random API key to the ignored `Backend\.env`.
Keep that file private. The server listens on `http://127.0.0.1:8000`; protected
routes use `X-API-Key`. `/health` is public. OpenAPI is available at
`/openapi.json`, and its checked-in export is the repository-root `openapi.json`.
Set `MEDIPREDICT_HOST` and `MEDIPREDICT_PORT` in `Backend\.env` to change the
bind address or avoid a port conflict.

Protected API routes use `/api/v1`. Uploaded tabular files are normalized by
the shared ingestion library, stored outside any frontend web root under
`Backend\data\uploads\` by default, and deleted after seven days. The stored
tabular representation has already had patient-identifier columns removed.
PDF and DOCX files are table previews only and are not stored as source files.
Set `MEDIPREDICT_UPLOAD_DIR` to a protected location outside this repository
when required by the deployment environment.

Configure `CORS_ORIGINS` as a comma-separated allow-list. Requests are
rate-limited per client address. `POST /api/v1/readings`,
`GET /api/v1/inventory/summary`, and `GET /api/v1/stream` are explicit 501
stubs until the later live-service phase; no transfers are dispatched.

Run backend checks and regenerate the contract with:

```powershell
Test-Backend
Export-BackendOpenApi
```
