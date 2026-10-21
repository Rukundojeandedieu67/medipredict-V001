. (Join-Path $PSScriptRoot "Backend\tasks.ps1")

function data {
    Push-Location $PSScriptRoot
    try {
        python scripts\prepare_reference.py
        if ($LASTEXITCODE -ne 0) { throw "Reference data preparation failed" }
        python scripts\calibrate_simulation.py
        if ($LASTEXITCODE -ne 0) { throw "Simulation calibration failed" }
    }
    finally {
        Pop-Location
    }
}

function train {
    Push-Location $PSScriptRoot
    try {
        python scripts\train_model.py
        if ($LASTEXITCODE -ne 0) { throw "Model training failed" }
    }
    finally {
        Pop-Location
    }
}

function test {
    Push-Location $PSScriptRoot
    try {
        python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Project tests failed" }
        Test-Backend
    }
    finally {
        Pop-Location
    }
}

function api {
    Start-Backend
}

function web {
    $webPackage = Join-Path $PSScriptRoot "web\package.json"
    if (-not (Test-Path $webPackage)) {
        throw "The web frontend is not implemented yet; it is the next approved stage."
    }
    Push-Location (Split-Path -Parent $webPackage)
    try {
        npm run dev
        if ($LASTEXITCODE -ne 0) { throw "Web development server failed" }
    }
    finally {
        Pop-Location
    }
}

function gen-api {
    Export-BackendOpenApi
}
