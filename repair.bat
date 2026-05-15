@echo off
REM ============================================================
REM Utopia Client - repair tool (v0.9.84+)
REM
REM Stops NSSM service, purges stale Python caches (.pyc / __pycache__)
REM and restarts. Use when self-update partially applied (e.g. mismatched
REM bytecode after extract).
REM
REM REQUIRES Administrator privileges (service stop/start).
REM ============================================================

set "SYS=%SystemRoot%\System32"
set "NET=%SYS%\net.exe"
set "TASKKILL=%SYS%\taskkill.exe"

if exist "%SYS%\chcp.com" "%SYS%\chcp.com" 65001 >nul

REM --- Administrator check ---
"%NET%" session >nul 2>&1
if errorlevel 1 (
    echo [!] Administrator privileges required.
    echo     Right-click repair.bat -^> "Run as administrator".
    pause
    exit /b 1
)

set "INSTALL_DIR=%APPDATA%\utopia-client"
set "PKG_DIR=%INSTALL_DIR%\utopia_client"
set "NSSM=%INSTALL_DIR%\nssm.exe"

if not exist "%INSTALL_DIR%" (
    echo [!] Install dir not found: %INSTALL_DIR%
    pause
    exit /b 1
)

echo === Utopia Client repair ===
echo Install dir: %INSTALL_DIR%
echo.

REM --- Stop services ---
echo [1/4] Stopping services...
if exist "%NSSM%" (
    "%NSSM%" stop UtopiaClientWatchdog >nul 2>&1
    "%NSSM%" stop UtopiaClient >nul 2>&1
) else (
    echo     NSSM not found, attempting taskkill...
    "%TASKKILL%" /F /IM python.exe /FI "WINDOWTITLE eq UtopiaClient*" >nul 2>&1
)

REM Wait for processes to release file locks
ping -n 4 127.0.0.1 >nul

REM --- Purge bytecode caches ---
echo [2/4] Purging __pycache__ and .pyc...
if exist "%PKG_DIR%" (
    for /d /r "%PKG_DIR%" %%d in (__pycache__) do (
        if exist "%%d" rd /s /q "%%d" 2>nul
    )
    del /s /q "%PKG_DIR%\*.pyc" >nul 2>&1
)

REM --- Optional: clear log tail (keeps logs dir but rotates) ---
echo [3/4] Rotating log files...
if exist "%INSTALL_DIR%\logs" (
    for %%f in ("%INSTALL_DIR%\logs\*.log") do (
        if exist "%%f" type nul > "%%f"
    )
)

REM --- Restart services ---
echo [4/4] Starting services...
if exist "%NSSM%" (
    "%NSSM%" start UtopiaClient
    "%NSSM%" start UtopiaClientWatchdog
    echo.
    echo Status:
    "%NSSM%" status UtopiaClient
) else (
    echo     NSSM not found — please run install.bat for full setup.
)

echo.
echo === Repair complete ===
pause
