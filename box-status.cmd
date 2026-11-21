@echo off
setlocal
cd /d "%~dp0"
".venv-mujoco4\Scripts\python.exe" -m examples.box_status %*
exit /b %errorlevel%
