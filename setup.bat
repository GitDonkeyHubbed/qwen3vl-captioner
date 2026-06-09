@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Qwen3-VL Captioner - Portable Setup
echo   This will install everything needed to run the app.
echo ============================================================
echo.
echo   PREREQUISITES (install these on your system first):
echo     - Windows 10/11 (64-bit)
echo     - NVIDIA GPU with current drivers
echo     - NVIDIA CUDA Toolkit 12.4 - 13.x  (NOT just the GPU driver)
echo       Install:  winget install Nvidia.CUDA
echo.
echo   Setup will install Python 3.12 and the llama-cpp wheel that
echo   matches your CUDA Toolkit version. Re-run setup after installing CUDA.
echo.

REM --- Step 1: Get or verify uv ---
echo [1/5] Checking for uv package manager...
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
echo [2/5] Installing Python 3.12 via uv...
uv python install 3.12
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python 3.12.
    pause
    exit /b 1
)
echo      Python 3.12 ready.
echo.

REM --- Step 3: Create virtual environment and install deps ---
echo [3/5] Creating virtual environment and installing dependencies...
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

REM --- Step 4: Detect CUDA Toolkit and select matching wheel ---
echo [4/5] Detecting CUDA Toolkit...
set "CUDA_VERSION="
set "CUDA_WHEEL=cu124"
for /f "usebackq tokens=1,2 delims=|" %%A in (`powershell -NoProfile -Command "$root='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'; if(-not(Test-Path $root)){Write-Output 'MISSING|cu124'; exit}; $dir=Get-ChildItem $root -Directory | Where-Object { $_.Name -match '^v' } | Sort-Object Name -Descending | Select-Object -First 1; if(-not $dir){Write-Output 'MISSING|cu124'; exit}; $ver=[version]($dir.Name.TrimStart('v')); $tag=if($ver.Major -ge 13){'cu130'}elseif($ver -ge [version]'12.8'){'cu128'}else{'cu124'}; Write-Output ($dir.Name+'|'+$tag)"`) do (
    set "CUDA_VERSION=%%A"
    set "CUDA_WHEEL=%%B"
)

if /I "!CUDA_VERSION!"=="MISSING" (
    echo      [WARNING] CUDA Toolkit not found.
    echo                GPU drivers alone are not enough - the CUDA Toolkit provides
    echo                runtime DLLs required by llama-cpp-python.
    echo.
    echo                Install with:  winget install Nvidia.CUDA
    echo                Or download:   https://developer.nvidia.com/cuda-downloads
    echo.
    echo                Continuing with default wheel ^(!CUDA_WHEEL!^) - you may need to
    echo                re-run setup.bat after installing CUDA Toolkit.
    echo.
) else (
    echo      Found CUDA Toolkit !CUDA_VERSION! - using !CUDA_WHEEL! wheel.
)
echo.

REM --- Step 5: Install llama-cpp-python with Qwen3-VL support ---
echo [5/5] Installing llama-cpp-python with Qwen3-VL support...
echo.
echo      Using JamePeng's fork with Qwen3-VL vision handler support.
echo      Source: https://github.com/JamePeng/llama-cpp-python
echo.

set "WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.24-!CUDA_WHEEL!-Basic-win-20260208/llama_cpp_python-0.3.24%%2B!CUDA_WHEEL!.basic-cp312-cp312-win_amd64.whl"
echo      Installing llama-cpp-python v0.3.24 ^(!CUDA_WHEEL!^)...
uv pip install --python .venv\Scripts\python.exe "!WHEEL_URL!"

if %ERRORLEVEL% EQU 0 (
    echo      llama-cpp-python with Qwen3-VL support installed successfully!
    goto :install_done
)

echo.
echo [ERROR] Failed to install llama-cpp-python.
echo        This may indicate a network issue or a wheel/CUDA version mismatch.
echo.
echo        Manual installation:
echo        1. Download the wheel matching your CUDA version from:
echo           https://github.com/JamePeng/llama-cpp-python/releases
echo        2. Install with: .venv\Scripts\pip.exe install [downloaded-file.whl]
echo.
pause
exit /b 1

:install_done
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To launch the app, double-click:  run.bat
echo ============================================================
echo.
pause
