@echo off
REM ============================================================
REM Utopia Client - Windows installer v2 (NSSM-сервис + watchdog)
REM
REM Ставит embedded Python 3.12 + зависимости в %APPDATA%\utopia-client,
REM скачивает NSSM и регистрирует ДВА Windows-сервиса:
REM   UtopiaClient         — основной процесс (utopia_client.main run)
REM   UtopiaClientWatchdog — heartbeat-сторож (utopia_client.watchdog)
REM Оба стартуют при загрузке системы (Шефа можно не логинить).
REM
REM ТРЕБУЕТСЯ запуск от Administrator (NSSM install service).
REM Старый ярлык-инсталлер сохранён рядом как install_legacy.bat.
REM Снос сервисов: nssm stop/remove UtopiaClient, UtopiaClientWatchdog.
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

REM --- Проверка прав Administrator (NSSM install service требует) ---
"%NET%" session >nul 2>&1
if errorlevel 1 (
    echo [!] Требуется запуск от Administrator.
    echo     Правый клик на install.bat -^> "Запустить от имени администратора".
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

REM --- Снос старых процессов/сервисов, если уже стояли ---
echo [0/7] Снимаю старые сервисы и процессы (если есть)...
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

echo [4/7] Installing requirements (включая neurocore[client] из VPS-зеркала)...
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
REM nssm-2.24 распаковывается в подпапку nssm-2.24/win64/nssm.exe
copy /Y "%INSTALL_DIR%\nssm-2.24\win64\nssm.exe" "%INSTALL_DIR%\nssm.exe" >nul
if errorlevel 1 goto :err
del /q "%INSTALL_DIR%\nssm.zip"
goto :after_nssm
:skip_nssm
echo [6/7] NSSM already present.
:after_nssm

echo [7/7] Регистрация NSSM-сервисов...

REM --- Основной сервис: utopia_client.main run ---
"%INSTALL_DIR%\nssm.exe" install UtopiaClient ^
    "%INSTALL_DIR%\python\python.exe" -m utopia_client.main run
if errorlevel 1 goto :err
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppDirectory "%INSTALL_DIR%"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRestartDelay 5000
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppStdout "%LOGS_DIR%\stdout.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppStderr "%LOGS_DIR%\stderr.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRotateFiles 1
"%INSTALL_DIR%\nssm.exe" set UtopiaClient AppRotateBytes 10485760
"%INSTALL_DIR%\nssm.exe" set UtopiaClient Start SERVICE_AUTO_START
"%INSTALL_DIR%\nssm.exe" set UtopiaClient Description "Utopia distributed evolution client"

REM --- Watchdog-сервис: utopia_client.watchdog ---
"%INSTALL_DIR%\nssm.exe" install UtopiaClientWatchdog ^
    "%INSTALL_DIR%\python\python.exe" -m utopia_client.watchdog
if errorlevel 1 goto :err
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppDirectory "%INSTALL_DIR%"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRestartDelay 5000
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppStdout "%LOGS_DIR%\watchdog.log"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppStderr "%LOGS_DIR%\watchdog.err"
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRotateFiles 1
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog AppRotateBytes 1048576
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog Start SERVICE_AUTO_START
"%INSTALL_DIR%\nssm.exe" set UtopiaClientWatchdog Description "Utopia client local heartbeat watchdog"

echo Стартую сервисы...
"%INSTALL_DIR%\nssm.exe" start UtopiaClient
"%INSTALL_DIR%\nssm.exe" start UtopiaClientWatchdog

REM --- Совместимость со старым ярлыком (ручной запуск тоже доступен) ---
> "%INSTALL_DIR%\utopia-client.bat" echo @echo off
>>"%INSTALL_DIR%\utopia-client.bat" echo if exist "%%SystemRoot%%\System32\chcp.com" "%%SystemRoot%%\System32\chcp.com" 65001 ^>nul
>>"%INSTALL_DIR%\utopia-client.bat" echo cd /d "%INSTALL_DIR%"
>>"%INSTALL_DIR%\utopia-client.bat" echo "%INSTALL_DIR%\python\python.exe" -m utopia_client.main %%*

echo.
echo === Installation complete ===
echo Сервисы:   UtopiaClient, UtopiaClientWatchdog (auto-start при boot)
echo Логи:      %LOGS_DIR%
echo Статус:    "%INSTALL_DIR%\nssm.exe" status UtopiaClient
echo Стоп:      "%INSTALL_DIR%\nssm.exe" stop UtopiaClient
echo Снос:      "%INSTALL_DIR%\nssm.exe" remove UtopiaClient confirm
echo.
echo Ручной запуск (без сервиса) — старый ярлык:
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
