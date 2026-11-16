$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv-mujoco4\Scripts\python.exe')) {
    python -m venv .venv-mujoco4
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 or later is required.' }
}
& .\.venv-mujoco4\Scripts\python.exe -m pip install -r requirements-sysid.txt -r requirements-robot.txt
if ($LASTEXITCODE -ne 0) { throw 'MuJoCo 4 dependency installation failed.' }
& .\.venv-mujoco4\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'MuJoCo 4 validation failed.' }
