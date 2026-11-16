@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-mujoco4\Scripts\python.exe" (
    echo Run setup-mujoco4.cmd first.
    exit /b 1
)
if not defined SOURCCEY_RENDERER set "SOURCCEY_RENDERER=filament"
".venv-mujoco4\Scripts\python.exe" %*
exit /b %errorlevel%
