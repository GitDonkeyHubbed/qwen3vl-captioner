@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Qwen3-VL Captioner - Portable Setup
echo   This will install everything needed to run the app.
echo ============================================================
echo.

REM --- Step 1: Get or verify uv ---
echo [1/4] Checking for uv package manager...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo      uv not found. Installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install uv. Please install manually from https://astral.sh/uv
        pause
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    echo      uv installed successfully.
) else (
    echo      uv found.
)
echo.

REM --- Step 2: Install Python via uv ---
echo [2/4] Installing Python 3.12 via uv...
uv python install 3.12
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python 3.12.
    pause
    exit /b 1
)
echo      Python 3.12 ready.
echo.

REM --- Step 3: Create virtual environment and install deps ---
echo [3/4] Creating virtual environment and installing dependencies...
cd /d "%~dp0"

uv venv --python 3.12 .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

uv pip install --python .venv\Scripts\python.exe PyQt6>=6.7 Pillow>=10.0 huggingface-hub>=0.25 numpy>=1.26 pynvml>=11.0
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo      Core dependencies installed.
echo.

REM --- Step 4: Install llama-cpp-python with Qwen3-VL support ---
echo [4/4] Installing llama-cpp-python with Qwen3-VL support...
echo.
echo      Using JamePeng's fork with Qwen3-VL vision handler support.
echo      Source: https://github.com/JamePeng/llama-cpp-python
echo.

REM Install JamePeng's llama-cpp-python v0.3.24 with CUDA 12.4 support
REM This wheel includes Qwen3VLChatHandler and works with CUDA 12.1+ drivers
echo      Installing llama-cpp-python v0.3.24 (CUDA 12.4)...
uv pip install --python .venv\Scripts\python.exe "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.24-cu124-Basic-win-20260208/llama_cpp_python-0.3.24%%2Bcu124.basic-cp312-cp312-win_amd64.whl"

if %ERRORLEVEL% EQU 0 (
    echo      llama-cpp-python with Qwen3-VL support installed successfully!
    goto :install_done
)

echo.
echo [ERROR] Failed to install llama-cpp-python.
echo        This may indicate a network issue or missing CUDA drivers.
echo.
echo        Manual installation:
echo        1. Download the wheel from:
echo           https://github.com/JamePeng/llama-cpp-python/releases
echo        2. Install with: .venv\Scripts\pip.exe install [downloaded-file.whl]
echo.
pause
exit /b 1

:cuda_done
echo      llama-cpp-python with CUDA support installed successfully!

:install_done
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To launch the app, double-click:  run.bat
echo ============================================================
echo.
pause
