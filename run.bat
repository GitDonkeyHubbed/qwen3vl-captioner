@echo off
cd /d "%~dp0"

REM Add CUDA to PATH for DLL loading (prevents ggml.dll / access violation errors)
set "CUDA_ADDED="
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
    if not defined CUDA_ADDED (
        set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D\bin;%PATH%"
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%D"
        set "CUDA_ADDED=1"
    )
)
if not defined CUDA_ADDED (
    echo [WARNING] CUDA Toolkit not found. Install from https://developer.nvidia.com/cuda-downloads
    echo           or run: winget install Nvidia.CUDA
    echo           Then re-run setup.bat to install the matching llama-cpp-python wheel.
    echo.
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Please run setup.bat first.
    pause
    exit /b 1
)

echo Starting Qwen3-VL Captioner...
.venv\Scripts\python.exe app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
