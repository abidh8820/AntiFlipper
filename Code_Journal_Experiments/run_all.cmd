@echo off
setlocal
set PROFILE=%~1
if "%PROFILE%"=="" set PROFILE=core
set WORKERS=%~2
if "%WORKERS%"=="" set WORKERS=1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Profile "%PROFILE%" -Workers "%WORKERS%"
exit /b %ERRORLEVEL%
