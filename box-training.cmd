@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m examples.box_training --review %*
exit /b %errorlevel%
