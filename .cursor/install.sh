#!/usr/bin/env bash
# Cloud Agent bootstrap for Qwen3-VL Captioner.
#
# Idempotent: safe to re-run. Sets up a Python 3.12 virtualenv with the app's
# runtime + dev dependencies and the system Qt/OpenGL libraries PyQt6 needs to
# load (headless or on the VM's desktop display). The GGUF inference engine
# (JamePeng's llama-cpp-python fork) is built best-effort — the GUI and the
# full test suite run without it, and it needs a GPU + model to actually
# caption, neither of which a base dev VM has.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] Installing system libraries for PyQt6..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3.12-venv \
    libegl1 libgl1 libglib2.0-0 libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
    libdbus-1-3 fonts-dejavu-core

echo "[2/4] Creating Python 3.12 virtualenv (.venv)..."
if [[ ! -x ".venv/bin/python" ]]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip --quiet

echo "[3/4] Installing runtime + dev dependencies..."
.venv/bin/python -m pip install --quiet -r requirements.txt -r requirements-dev.txt

echo "[4/4] Building the GGUF inference engine (best-effort, CPU)..."
# The fork carries the Qwen3-VL patches; the stock PyPI package has no Linux
# wheel. A source build needs cmake + a C/C++ toolchain (present in the base
# image). Never fail setup if this step can't complete.
if .venv/bin/python -c "import llama_cpp" 2>/dev/null; then
    echo "      llama-cpp-python already present."
elif CMAKE_ARGS="-DGGML_NATIVE=off" .venv/bin/python -m pip install --quiet \
        "llama_cpp_python @ git+https://github.com/JamePeng/llama-cpp-python"; then
    echo "      llama-cpp-python built successfully."
else
    echo "      [WARN] Engine build skipped/failed — GUI + tests still work."
fi

echo
echo "Setup complete."
echo "  Run tests:   QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q"
echo "  Launch GUI:  DISPLAY=:1 ./run.sh   (needs the VM desktop display)"
