# Spawns backend + frontend in two new PowerShell windows so you can iterate
# on both at once. Usage:  ./scripts/run_dev.ps1

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$root'; if (Test-Path .venv/Scripts/Activate.ps1) { .venv/Scripts/Activate.ps1 }; python -m backend.main"
)

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$root/frontend'; npm run dev"
)

Write-Host "Backend on http://localhost:8000 — frontend on http://localhost:5173"
