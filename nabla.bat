@echo off
REM Nabla Launcher for Windows
REM Double-click this file to launch Nabla

cd /d "%~dp0"

REM Check for virtual environment
if exist ".venv\Scripts\python.exe" (
    echo [+] Using existing virtual environment
    .venv\Scripts\python.exe -m nabla
    goto :end
)

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

REM Create venv
echo [+] Creating virtual environment...
python -m venv .venv

REM Install dependencies
echo [+] Installing dependencies...
.venv\Scripts\pip.exe install --upgrade pip -q
.venv\Scripts\pip.exe install -e . -q

REM Launch
echo [+] Launching Nabla...
.venv\Scripts\python.exe -m nabla

:end
if %errorlevel% neq 0 (
    echo.
    echo [!] Nabla exited with code %errorlevel%
)
pause
