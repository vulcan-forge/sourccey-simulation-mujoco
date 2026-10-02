@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m sourccey.build
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pytest -q
exit /b %errorlevel%
