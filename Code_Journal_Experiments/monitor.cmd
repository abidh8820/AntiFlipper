@echo off
setlocal
set PROFILE=%~1
if "%PROFILE%"=="" set PROFILE=core
set INTERVAL=%~2
if "%INTERVAL%"=="" set INTERVAL=10
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" monitor.py --profile "%PROFILE%" --watch --interval %INTERVAL%
) else (
  python monitor.py --profile "%PROFILE%" --watch --interval %INTERVAL%
)
