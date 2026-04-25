@echo off
REM ============================================================
REM Utopia Client — Windows installer
REM Ставит embedded Python 3.12 + pip + зависимости в %APPDATA%
REM Использует абсолютные пути к системным утилитам — не зависит
REM от пользовательского PATH (на части машин он урезан).
REM ============================================================

REM Пути к системным утилитам (Windows 10/11)
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
echo Папка установки: %INSTALL_DIR%
echo.

REM ---- Проверка обязательных утилит ----
if not exist "%CURL%" (
    echo [!] Не найден %CURL%
    echo     Нужна Windows 10 1803+ или установите curl вручную.
    pause
    exit /b 1
)
if not exist "%TAR%" (
    echo [!] Не найден %TAR%
    echo     Нужна Windows 10 17063+ или установите tar вручную.
    pause
    exit /b 1
)
if not exist "%XCOPY%" (
    echo [!] Не найден %XCOPY%
    pause
    exit /b 1
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
cd /d "%INSTALL_DIR%"

REM ---- Python embedded ----
if not exist "python\python.exe" (
    echo [1/5] Скачиваю Python %PYTHON_VERSION% embedded...
    "%CURL%" -fsSL -o python.zip "%PYTHON_URL%" || goto :err
    if not exist python mkdir python
    "%TAR%" -xf python.zip -C python || goto :err
    del /q python.zip
    REM Включаем site-packages в embedded Python (перезаписываем _pth)
    > "python\python%PYTHON_SHORT%._pth" (
        echo python%PYTHON_SHORT%.zip
        echo .
        echo import site
    )
) else (
    echo [1/5] Python уже установлен.
)

REM ---- pip ----
if not exist "python\Scripts\pip.exe" (
    echo [2/5] Ставлю pip...
    "%CURL%" -fsSL -o get-pip.py "%GETPIP_URL%" || goto :err
    "%INSTALL_DIR%\python\python.exe" get-pip.py --no-warn-script-location || goto :err
    del /q get-pip.py
) else (
    echo [2/5] pip уже установлен.
)

REM ---- Зависимости ----
echo [3/5] Ставлю зависимости (requests, websockets, psutil, numpy)...
"%INSTALL_DIR%\python\python.exe" -m pip install --upgrade pip --quiet || goto :err
"%INSTALL_DIR%\python\python.exe" -m pip install -r "%~dp0requirements.txt" numpy --quiet || goto :err

REM ---- Код клиента ----
echo [4/5] Копирую код клиента...
"%XCOPY%" /e /i /y "%~dp0utopia_client" "%INSTALL_DIR%\utopia_client" >nul

REM ---- launcher ----
echo [5/5] Создаю launcher...
> "%INSTALL_DIR%\utopia-client.bat" (
    echo @echo off
    echo if exist "%%SystemRoot%%\System32\chcp.com" "%%SystemRoot%%\System32\chcp.com" 65001 ^>nul
    echo "%INSTALL_DIR%\python\python.exe" -m utopia_client.main %%*
)

echo.
echo === Установка завершена ===
echo Запуск:
echo   "%INSTALL_DIR%\utopia-client.bat" benchmark
echo   "%INSTALL_DIR%\utopia-client.bat" run
echo.
pause
goto :eof

:err
echo.
echo [!] Ошибка установки. Код %ERRORLEVEL%
pause
exit /b 1
