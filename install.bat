@echo off
REM ============================================================
REM Utopia Client - Windows installer
REM Installs embedded Python 3.12 + pip + dependencies into %APPDATA%
REM Uses absolute paths to system utilities to bypass broken PATH.
REM ============================================================

set "SYS=%SystemRoot%\System32"
set "CHCP=%SYS%\chcp.com"
set "CURL=%SYS%\curl.exe"
set "TAR=%SYS%\tar.exe"
set "XCOPY=%SYS%\xcopy.exe"

if exist "%CHCP%" "%CHCP%" 65001 >nul

setlocal enabledelayedexpansion

set "INSTALL_DIR=%APPDATA%\utopia-client"
set "PYTHON_VERSION=3.12.7"
set "PYTHON_SHORT=312"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo === Utopia Client installer ===
echo Install dir: %INSTALL_DIR%
echo.

if not exist "%CURL%" (
    echo [!] curl.exe not found at %CURL%
    echo     Need Windows 10 1803+ or install curl manually.
    pause
    exit /b 1
)
if not exist "%TAR%" (
    echo [!] tar.exe not found at %TAR%
    echo     Need Windows 10 17063+ or install tar manually.
    pause
    exit /b 1
)
if not exist "%XCOPY%" (
    echo [!] xcopy.exe not found at %XCOPY%
    pause
    exit /b 1
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
cd /d "%INSTALL_DIR%"

if exist "python\python.exe" goto :skip_python
echo [1/5] Downloading Python %PYTHON_VERSION% embedded...
"%CURL%" -fsSL -o python.zip "%PYTHON_URL%"
if errorlevel 1 goto :err
if not exist python mkdir python
"%TAR%" -xf python.zip -C python
if errorlevel 1 goto :err
del /q python.zip
goto :after_python
:skip_python
echo [1/5] Python already installed.
:after_python

REM Always (re)write the _pth file so the installer is idempotent.
REM ".." adds %INSTALL_DIR% to sys.path so utopia_client package is found.
> "python\python%PYTHON_SHORT%._pth" echo python%PYTHON_SHORT%.zip
>>"python\python%PYTHON_SHORT%._pth" echo .
>>"python\python%PYTHON_SHORT%._pth" echo ..
>>"python\python%PYTHON_SHORT%._pth" echo import site

if exist "python\Scripts\pip.exe" goto :skip_pip
echo [2/5] Installing pip...
"%CURL%" -fsSL -o get-pip.py "%GETPIP_URL%"
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" get-pip.py --no-warn-script-location
if errorlevel 1 goto :err
del /q get-pip.py
goto :after_pip
:skip_pip
echo [2/5] pip already installed.
:after_pip

echo [3/6] Upgrading pip + installing PyTorch CUDA 12.1 (~2.5 GB, может быть долго)...
"%INSTALL_DIR%\python\python.exe" -m pip install --upgrade pip --quiet
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [!] PyTorch CUDA install failed, fallback to CPU-only torch...
    "%INSTALL_DIR%\python\python.exe" -m pip install torch --quiet
    if errorlevel 1 goto :err
)

echo [4/6] Installing requirements (включая neurocore[client] из git)...
REM hatchling нужен в основной среде, чтобы neurocore собрался без build isolation
REM (embedded Python + изолированная сборка ломаются на python._pth).
"%INSTALL_DIR%\python\python.exe" -m pip install hatchling numpy --quiet
if errorlevel 1 goto :err
"%INSTALL_DIR%\python\python.exe" -m pip install -r "%~dp0requirements.txt" --no-build-isolation --quiet
if errorlevel 1 goto :err

echo [5/6] Copying client code...
"%XCOPY%" /e /i /y "%~dp0utopia_client" "%INSTALL_DIR%\utopia_client" >nul

echo [6/6] Creating launcher...
> "%INSTALL_DIR%\utopia-client.bat" echo @echo off
>>"%INSTALL_DIR%\utopia-client.bat" echo if exist "%%SystemRoot%%\System32\chcp.com" "%%SystemRoot%%\System32\chcp.com" 65001 ^>nul
>>"%INSTALL_DIR%\utopia-client.bat" echo cd /d "%INSTALL_DIR%"
>>"%INSTALL_DIR%\utopia-client.bat" echo "%INSTALL_DIR%\python\python.exe" -m utopia_client.main %%*

echo.
echo === Installation complete ===
echo Run:
echo   "%INSTALL_DIR%\utopia-client.bat" benchmark
echo   "%INSTALL_DIR%\utopia-client.bat" run
echo.
pause
goto :eof

:err
echo.
echo [!] Install failed. Code %ERRORLEVEL%
pause
exit /b 1
