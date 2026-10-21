$BackendRoot = $PSScriptRoot
$BackendPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"

function Initialize-BackendEnvironment {
    param([switch]$Force)
    $needsInstall = $Force
    if (-not (Test-Path $BackendPython)) {
        & python -m venv (Join-Path $BackendRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Could not create Backend\.venv" }
        $needsInstall = $true
    }
    $stamp = Join-Path $BackendRoot ".venv\.backend-ready"
    if (-not (Test-Path $stamp)) { $needsInstall = $true }
    elseif (-not $Force) {
        $stampTime = (Get-Item $stamp).LastWriteTimeUtc
        $manifests = @(
            (Join-Path $BackendRoot "requirements.txt"),
            (Join-Path $BackendRoot "..\pyproject.toml")
        )
        foreach ($manifest in $manifests) {
            if ((Get-Item $manifest).LastWriteTimeUtc -gt $stampTime) {
                $needsInstall = $true
            }
        }
    }
    if ($needsInstall) {
        & $BackendPython -m pip install --disable-pip-version-check -e (Join-Path $BackendRoot "..")
        if ($LASTEXITCODE -ne 0) { throw "Could not install the MediPredict package into Backend\.venv" }
        & $BackendPython -m pip install --disable-pip-version-check -r (Join-Path $BackendRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) { throw "Could not install the backend requirements" }
        Set-Content -Path $stamp -Value (Get-Date -Format o) -Encoding ascii
    }
    if (-not (Test-Path (Join-Path $BackendRoot ".env"))) {
        $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        $bytes = New-Object byte[] 32
        $random.GetBytes($bytes)
        $random.Dispose()
        $key = [Convert]::ToBase64String($bytes)
        @(
            "MEDIPREDICT_API_KEY=$key"
            "MEDIPREDICT_HOST=127.0.0.1"
            "MEDIPREDICT_PORT=8000"
            "CORS_ORIGINS=http://localhost:5173"
            "MEDIPREDICT_RATE_LIMIT_PER_MINUTE=120"
        ) | Set-Content -Path (Join-Path $BackendRoot ".env") -Encoding ascii
        Write-Host "Created Backend\.env with a random local API key; keep it private."
    }
}

function Start-Backend {
    Initialize-BackendEnvironment
    $hostSetting = $env:MEDIPREDICT_HOST
    if (-not $hostSetting) {
        $hostLine = Get-Content (Join-Path $BackendRoot ".env") |
            Where-Object { $_ -like "MEDIPREDICT_HOST=*" } |
            Select-Object -First 1
        if ($hostLine) { $hostSetting = $hostLine.Substring("MEDIPREDICT_HOST=".Length) }
    }
    if (-not $hostSetting) { $hostSetting = "127.0.0.1" }
    $portSetting = $env:MEDIPREDICT_PORT
    if (-not $portSetting) {
        $portLine = Get-Content (Join-Path $BackendRoot ".env") |
            Where-Object { $_ -like "MEDIPREDICT_PORT=*" } |
            Select-Object -First 1
        if ($portLine) { $portSetting = $portLine.Substring("MEDIPREDICT_PORT=".Length) }
    }
    if (-not $portSetting) { $portSetting = "8000" }
    Push-Location $BackendRoot
    try {
        & $BackendPython -m uvicorn app.main:app --host $hostSetting --port $portSetting
        if ($LASTEXITCODE -ne 0) { throw "Backend API exited with code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}

function Test-Backend {
    Initialize-BackendEnvironment
    Push-Location $BackendRoot
    try {
        & $BackendPython -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Backend tests failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}

function Export-BackendOpenApi {
    Initialize-BackendEnvironment
    Push-Location $BackendRoot
    try {
        & $BackendPython scripts\export_openapi.py
        if ($LASTEXITCODE -ne 0) { throw "OpenAPI export failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}
