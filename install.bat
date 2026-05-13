@echo off
REM ============================================================
REM Utopia Client - Windows installer v2 (NSSM service + watchdog)
REM
REM Installs embedded Python 3.12 + dependencies into %APPDATA%
REM and registers two Windows services via NSSM:
REM   UtopiaClient         - main daemon (utopia_client.main run)
REM   UtopiaClientWatchdog - local heartbeat watcher
REM Both auto-start on boot (no need to be logged in).
REM
REM REQUIRES Administrator privileges (NSSM service install).
REM Old shortcut-style installer kept as install_legacy.bat.
REM Removal: nssm stop/remove UtopiaClient, UtopiaClientWatchdog.
REM ============================================================

set "SYS=%SystemRoot%\System32"
set "CHCP=%SYS%\chcp.com"
set "CURL=%SYS%\curl.exe"
set "TAR=%SYS%\tar.exe"
set "XCOPY=%SYS%\xcopy.exe"
set "NET=%SYS%\net.exe"
set "SC=%SYS%\sc.exe"
set "TASKKILL=%SYS%\taskkill.exe"

if exist "%CHCP%" "%CHCP%" 65001 >nul

setlocal enabledelayedexpansion

REM --- Administrator check (NSSM install service requires it) ---
"%NET%" session >nul 2>&1
if errorlevel 1 (
    echo [!] Administrator privileges required.
    echo     Right-click install.bat -^> "Run as administrator".
    pause
    exit /b 1
)

set "INSTALL_DIR=%APPDATA%\utopia-client"
set "LOGS_DIR=%INSTALL_DIR%\logs"
set "PYTHON_VERSION=3.12.7"
set "PYTHON_SHORT=312"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "NSSM_URL=https://nssm.cc/release/nssm-2.24.zip"

echo === Utopia Client installer v2 (NSSM) ===
echo Install dir: %INSTALL_DIR%
echo.

if not exist "%CURL%" (
    echo [!] curl.exe not found at %CURL%
    pause
    exit /b 1
)
if not exist "%TAR%" (
    echo [!] tar.exe not found at %TAR%
    pause
    exit /b 1
)
if not exist "%XCOPY%" (
    echo [!] xcopy.exe not found at %XCOPY%
    pause
    exit /b 1
)

REM --- Tear down previous services / processes if present ---
echo [0/7] Removing old services and processes (if any)...
if exist "%INSTALL_DIR%\nssm.exe" (
    "%INSTALL_DIR%\nssm.exe" stop UtopiaClientWatchdog >nul 2>&1
    "%INSTALL_DIR%\nssm.exe" stop UtopiaClient >nul 2>&1
    "%INSTALL_DIR%\nssm.exe" remove UtopiaClientWatchdog confirm >nul 2>&1
    "%INSTALL_DIR%\nssm.exe" remove UtopiaClient confirm >nul 2>&1
)
"%TASKKILL%" /F /IM python.exe /FI "WINDOWTITLE eq UtopiaClient*" >nul 2>&1

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
cd /d "%INSTALL_DIR%"

if exist "python\python.exe" goto :skip_python
echo [1/7] Downloading Python %PYTHON_VERSION% embedded...
"%CURL%" -fsSL -o python.zip "%PYTHON_URL%"
if errorlevel 1 goto :err
if not exist python mkdir python
"%TAR%" -xf python.zip -C python
if errorlevel 1 goto :err
del /q python.zip
goto :after_python
:skip_python
echo [1/7] Python already installed.
:after_python

> "python\python%PYTHON_SHORT%._pth" echo python%PYTHON_SHORT%.zip
>>"python\python%PYTHON_SHORT%._pth" echo .
>>"python\python%PYTHON_SHORT%._pth" echo ..
>>"python\python%PYTHON_SHORT%._pth" echo import site

if exist "python\Scripts\pip.exe" goto :skip_pip
echo [2/7] Installing pip...
"%CURL%" -fsSL -o get-pip.py "%GETPIP_URL%"
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" get-pip.py --no-warn-script-location
if errorlevel 1 goto :err
del /q get-pip.py
goto :after_pip
:skip_pip
echo [2/7] pip already installed.
:after_pip

echo [3/7] Upgrading pip + installing PyTorch CUDA 12.1 (~2.5 GB)...
"%INSTALL_DIR%\python\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [!] PyTorch CUDA install failed, fallback to CPU-only torch...
    "%INSTALL_DIR%\python\python.exe" -m pip install torch --quiet
    if errorlevel 1 goto :err
)

echo [4/7] Installing requirements (incl. neurocore[client] from VPS mirror)...
"%INSTALL_DIR%\python\python.exe" -m pip install hatchling numpy --quiet
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" -m pip install -r "%~dp0requirements.txt" --no-build-isolation --quiet
if errorlevel 1 goto :err

echo [5/7] Copying client code...
"%XCOPY%" /e /i /y "%~dp0utopia_client" "%INSTALL_DIR%\utopia_client" >nul

echo [6/7] Downloading NSSM (Non-Sucking Service Manager)...
if exist "%INSTALL_DIR%\nssm.exe" goto :skip_nssm
"%CURL%" -fsSL -o "%INSTALL_DIR%\nssm.zip" "%NSSM_URL%"
if errorlevel 1 goto :err
"%TAR%" -xf "%INSTALL_DIR%\nssm.zip" -C "%INSTALL_DIR%"
if errorlevel 1 goto :err
REM nssm-2.24 extracts to subfolder nssm-2.24/win64/nssm.exe
copy /Y "%INSTALL_DIR%\nssm-2.24\win64\nssm.exe" "%INSTALL_DIR%\nssm.exe" >nul
if errorlevel 1 goto :err
del /q "%INSTALL_DIR%\nssm.zip"
goto :after_nssm
:skip_nssm
echo [6/7] NSSM already present.
:after_nssm

echo [7/7] Registering NSSM services...

REM --- Main service: utopia_client.main run ---
"%INSTALL_DIR%\nssm.exe" install UtopiaClient ^
    "%INSTALL_DIR%\python\python.exe" -m utopia_client.main run
if errorlevel 1 goto :err
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppDirectory "%INSTALL_DIR%"
REM Pass user's APPDATA so service (running as LocalSystem) sees the same
REM config.json / heartbeat.txt / client.log as manual run.
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppEnvironmentExtra "APPDATA=%APPDATA%"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRestartDelay 5000
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppStdout "%LOGS_DIR%\stdout.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppStderr "%LOGS_DIR%\stderr.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRotateFiles 1
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRotateBytes 10485760
"%INSTALL_DIR%\nssm.exe" set UtopiaClient Start SERVICE_AUTO_START
"%INSTALL_DIR%\nssm.exe" set UtopiaClient Description "Utopia distributed evolution client"

REM --- Watchdog service: utopia_client.watchdog ---
"%INSTALL_DIR%\nssm.exe" install UtopiaClientWatchdog ^
    "%INSTALL_DIR%\python\python.exe" -m utopia_client.watchdog
if errorlevel 1 goto :err
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppDirectory "%INSTALL_DIR%"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppEnvironmentExtra "APPDATA=%APPDATA%"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRestartDelay 5000
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppStdout "%LOGS_DIR%\watchdog.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppStderr "%LOGS_DIR%\watchdog.err"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRotateFiles 1
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRotateBytes 1048576
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog Start SERVICE_AUTO_START
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog Description "Utopia client local heartbeat watchdog"

echo Starting services...
"%INSTALL_DIR%\nssm.exe" start UtopiaClient
"%INSTALL_DIR%\nssm.exe" start UtopiaClientWatchdog

REM --- Compatibility: keep old shortcut launcher (manual run still works) ---
> "%INSTALL_DIR%\utopia-client.bat" echo @echo off
>>"%INSTALL_DIR%\utopia-client.bat" echo if exist "%%SystemRoot%%\System32\chcp.com" "%%SystemRoot%%\System32\chcp.com" 65001 ^>nul
>>"%INSTALL_DIR%\utopia-client.bat" echo cd /d "%INSTALL_DIR%"
>>"%INSTALL_DIR%\utopia-client.bat" echo "%INSTALL_DIR%\python\python.exe" -m utopia_client.main %%*

echo.
echo === Installation complete ===
echo Services:  UtopiaClient, UtopiaClientWatchdog (auto-start on boot)
echo Logs:      %LOGS_DIR%
echo Status:    "%INSTALL_DIR%\nssm.exe" status UtopiaClient
echo Stop:      "%INSTALL_DIR%\nssm.exe" stop UtopiaClient
echo Remove:    "%INSTALL_DIR%\nssm.exe" remove UtopiaClient confirm
echo.
echo Manual run (no service) via legacy shortcut:
echo   "%INSTALL_DIR%\utopia-client.bat" run
echo   "%INSTALL_DIR%\utopia-client.bat" benchmark
echo.
pause
goto :eof

:err
echo.
echo [!] Install failed. Code %ERRORLEVEL%
pause
exit /b 1
