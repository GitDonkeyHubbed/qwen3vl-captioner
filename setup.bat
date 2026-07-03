@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Qwen3-VL Captioner - Portable Setup
echo   This will install everything needed to run the app.
echo ============================================================
echo.
echo   PREREQUISITES (must already be on your system):
echo     - Windows 10/11 (64-bit)
echo     - NVIDIA GPU with a current driver
echo     - NVIDIA CUDA Toolkit 12.4 or newer (NOT just the driver)
echo       Install:  winget install Nvidia.CUDA
echo.
echo   Setup installs Python 3.12 and the llama-cpp-python wheel
echo   that MATCHES your CUDA Toolkit version. If you install or
echo   upgrade CUDA later, re-run this setup.
echo.

REM --- Step 1: Get or verify uv ---
echo [1/6] Checking for uv package manager...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo      uv not found. Installing uv...
    REM Pinned installer version - a moving install.ps1 would execute whatever
    REM the latest script happens to be at install time.
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/0.11.26/install.ps1 | iex"
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
echo [2/6] Installing Python 3.12 via uv...
uv python install 3.12
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python 3.12.
    pause
    exit /b 1
)
echo      Python 3.12 ready.
echo.

REM --- Step 3: Create virtual environment and install deps ---
echo [3/6] Creating virtual environment and installing dependencies...
cd /d "%~dp0"

uv venv --python 3.12 .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo      Core dependencies installed.
echo.

REM --- Step 4: Detect CUDA Toolkit and select the matching wheel ---
REM Uses the same detection logic as the app itself (engine/cuda_setup.py)
echo [4/6] Detecting CUDA Toolkit...
set "CUDA_VERSION="
set "CUDA_WHEEL=cu124"
for /f "usebackq tokens=1,2 delims=;" %%A in (`.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from engine.cuda_setup import detect_cuda_toolkit, recommended_wheel_tag; t = detect_cuda_toolkit(); v = 'v{}.{}'.format(t[0][0], t[0][1]) if t else 'MISSING'; print(v + ';' + recommended_wheel_tag(t[0] if t else None))"`) do (
    set "CUDA_VERSION=%%A"
    set "CUDA_WHEEL=%%B"
)

if /I "!CUDA_VERSION!"=="MISSING" (
    echo.
    echo      [WARNING] CUDA Toolkit NOT FOUND.
    echo                GPU drivers alone are not enough - llama-cpp-python
    echo                needs the CUDA Toolkit's runtime DLLs.
    echo.
    echo                Install it now from another window:
    echo                    winget install Nvidia.CUDA
    echo                or: https://developer.nvidia.com/cuda-downloads
    echo.
    echo                Continuing with the default ^(!CUDA_WHEEL!^) wheel.
    echo                IMPORTANT: re-run setup.bat after installing CUDA.
    echo.
) else (
    echo      Found CUDA Toolkit !CUDA_VERSION! - selecting the !CUDA_WHEEL! wheel.
)
echo.

REM --- Step 5: Install llama-cpp-python with Qwen3-VL support ---
echo [5/6] Installing llama-cpp-python (Qwen VL build, !CUDA_WHEEL!)...
echo.
echo      Using JamePeng's fork with Qwen VL vision handler support.
echo      v0.3.40 also supports the newer Qwen3.5 / 3.6 model families.
echo      Source: https://github.com/JamePeng/llama-cpp-python
echo.

set "WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.40-!CUDA_WHEEL!-win-20260608/llama_cpp_python-0.3.40%%2B!CUDA_WHEEL!-cp312-cp312-win_amd64.whl"
uv pip install --python .venv\Scripts\python.exe "!WHEEL_URL!"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install llama-cpp-python.
    echo        This is usually a network issue.
    echo.
    echo        Manual installation:
    echo        1. Download the wheel matching your CUDA version from:
    echo           https://github.com/JamePeng/llama-cpp-python/releases
    echo           CUDA 13.1+ -^> cu131     CUDA 13.0  -^> cu130
    echo           CUDA 12.8+ -^> cu128     CUDA 12.6+ -^> cu126
    echo           CUDA 12.4+ -^> cu124
    echo        2. Install with: .venv\Scripts\pip.exe install [downloaded-file.whl]
    echo.
    pause
    exit /b 1
)
echo      llama-cpp-python ^(!CUDA_WHEEL!^) installed successfully!
echo.

REM --- Step 6: Verify the install actually works ---
echo [6/6] Verifying the engine loads...
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); from engine.cuda_setup import setup_cuda_dll_path; setup_cuda_dll_path(); import llama_cpp; print('      Engine OK - llama_cpp ' + llama_cpp.__version__)"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] The engine did not load cleanly. The app may still work,
    echo           but if it fails, run diagnose.bat for a full report and
    echo           include its output if you open a GitHub issue.
    echo.
)

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To launch the app, double-click:   run.bat
echo   If anything goes wrong, run:       diagnose.bat
echo ============================================================
echo.
pause
