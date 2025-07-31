@echo off
REM Michigan Guardianship AI - Launch Script (Windows)

echo ====================================
echo Michigan Guardianship AI
echo ====================================
echo.

REM Find Python executable
set PYTHON_CMD=
set PYTHON_FOUND=0

REM Check for python3 (some Windows installations use this)
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    set PYTHON_FOUND=1
    goto :check_version
)

REM Check for python (most common on Windows)
where python >nul 2>&1
if %errorlevel% equ 0 (
    REM Check if it's Python 3
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do (
        set VERSION=%%i
        if "!VERSION:~0,1!"=="3" (
            set PYTHON_CMD=python
            set PYTHON_FOUND=1
            goto :check_version
        )
    )
)

REM Check for py launcher (Windows Python Launcher)
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py -3
        set PYTHON_FOUND=1
        goto :check_version
    )
)

if %PYTHON_FOUND% equ 0 (
    echo Error: Python 3.8 or higher is required but not found.
    echo Please install Python from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:check_version
echo Using Python: %PYTHON_CMD%

REM Check Python version (at least 3.8)
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo Error: Python 3.8 or higher is required.
    echo Please upgrade your Python installation.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo.
    echo No .env file found.
    echo Please run '%PYTHON_CMD% setup.py' first to configure your environment.
    pause
    exit /b 1
)

REM Check if dependencies are installed
%PYTHON_CMD% -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Dependencies not installed.
    echo Please run '%PYTHON_CMD% setup.py' first to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting the Michigan Guardianship AI server...
echo.
echo The application will open at: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Optional: Try to open browser automatically after a short delay
start /b cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5000"

REM Start the Flask application
%PYTHON_CMD% app.py