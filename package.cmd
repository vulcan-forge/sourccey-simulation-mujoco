@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" package.py
exit /b %errorlevel%
