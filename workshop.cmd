@echo off
setlocal
cd /d "%~dp0"
call mujoco4.cmd -m examples.sandbox %*
exit /b %errorlevel%
