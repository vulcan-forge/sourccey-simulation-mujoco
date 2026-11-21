@echo off
setlocal
cd /d "%~dp0"
call mujoco4.cmd -m examples.box_policy_playlist %*
exit /b %errorlevel%
