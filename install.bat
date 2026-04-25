@echo off
REM ============================================================
REM Utopia Client — Windows installer
REM Ставит embedded Python 3.12 + venv + зависимости в %APPDATA%
REM ============================================================
setlocal enabledelayedexpansion

set "INSTALL_DIR=%APPDATA%\utopia-client"
set "PYTHON_VERSION=3.12.7"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo === Utopia Client installer ===
echo Папка установки: %INSTALL_DIR%
echo.

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
cd /d "%INSTALL_DIR%"

REM ---- Python embedded ----
if not exist "python\python.exe" (
    echo [1/5] Качаю Python %PYTHON_VERSION% embedded...
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile 'python.zip' -UseBasicParsing" || goto :err
    powershell -Command "Expand-Archive -Path 'python.zip' -DestinationPath 'python' -Force" || goto :err
    del /q python.zip
    REM Включаем site-packages в embedded Python
    for %%f in (python\python*._pth) do (
        powershell -Command "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
    )
) else (
    echo [1/5] Python уже установлен.
)

REM ---- pip ----
if not exist "python\Scripts\pip.exe" (
    echo [2/5] Ставлю pip...
    powershell -Command "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile 'get-pip.py' -UseBasicParsing" || goto :err
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
xcopy /e /i /y "%~dp0utopia_client" "%INSTALL_DIR%\utopia_client" >nul

REM ---- launcher ----
echo [5/5] Создаю launcher...
> "%INSTALL_DIR%\utopia-client.bat" (
    echo @echo off
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
