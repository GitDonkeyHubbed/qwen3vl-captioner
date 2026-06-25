@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Running install diagnostics...
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe doctor.py
) else (
    echo [WARN] Virtual environment not found - run setup.bat first.
    echo        Trying system Python as a fallback...
    where python >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        python doctor.py
    ) else (
        echo [FAIL] No Python found. Run setup.bat first.
    )
)

echo.
echo If you are reporting an installation issue on GitHub, please
echo copy the report above into the issue.
echo.
pause
