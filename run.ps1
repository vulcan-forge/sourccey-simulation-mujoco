param([switch]$Headless, [switch]$FullElevatorRange, [int]$UnityPort = 0)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run .\setup.ps1 first.' }
$simArgs = @('-m', 'sourccey.app')
if ($Headless) { $simArgs += '--headless' }
if ($FullElevatorRange) { $simArgs += '--full-elevator-range' }
if ($UnityPort -gt 0) { $simArgs += @('--unity-port', "$UnityPort") }
& $pythonExe @simArgs
exit $LASTEXITCODE
