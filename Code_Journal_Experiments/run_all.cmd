@echo off
setlocal
set PROFILE=%~1
if "%PROFILE%"=="" set PROFILE=core
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Profile "%PROFILE%"
exit /b %ERRORLEVEL%
