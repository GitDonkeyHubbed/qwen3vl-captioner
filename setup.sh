#!/bin/bash
# Qwen3-VL Captioner — macOS setup
#
# Installs uv, Python 3.12, all dependencies, the llama-cpp-python Metal
# build (GPU acceleration on Apple Silicon), and mlx-vlm for the MLX
# backend. Re-run safely at any time.
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  Qwen3-VL Captioner - macOS Setup"
echo "============================================================"
echo

if [[ "$(uname)" != "Darwin" ]]; then
    echo "[ERROR] This script is for macOS. On Windows use setup.bat."
    exit 1
fi

ARCH="$(uname -m)"
echo "Detected architecture: $ARCH"
echo

# --- Step 1: uv package manager ---
echo "[1/5] Checking for uv package manager..."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "      uv not found. Installing..."
    # Pinned installer version — a moving `astral.sh/uv/install.sh` would
    # execute whatever the latest script happens to be at install time.
    curl -LsSf https://astral.sh/uv/0.11.26/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "      uv ready."
echo

# --- Step 2: Python 3.12 ---
echo "[2/5] Installing Python 3.12..."
uv python install 3.12
echo "      Python 3.12 ready."
echo

# --- Step 3: Virtual environment + dependencies ---
echo "[3/5] Creating virtual environment and installing dependencies..."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
echo "      Core dependencies installed."
echo

# --- Step 4: llama-cpp-python (GGUF engine) ---
echo "[4/5] Installing llama-cpp-python (Qwen3-VL build)..."
LLAMA_VERSION="0.3.40"
LLAMA_TAG="v0.3.40-Metal-macos-20260607"
if [[ "$ARCH" == "arm64" ]]; then
    echo "      Apple Silicon: installing the Metal GPU wheel..."
    uv pip install --python .venv/bin/python \
        "https://github.com/JamePeng/llama-cpp-python/releases/download/${LLAMA_TAG}/llama_cpp_python-${LLAMA_VERSION}-cp312-cp312-macosx_11_0_arm64.whl"
else
    echo "      Intel Mac: no prebuilt wheel — building from source (CPU only)."
    echo "      This needs Xcode Command Line Tools and can take 10+ minutes."
    uv pip install --python .venv/bin/python \
        "llama_cpp_python @ git+https://github.com/JamePeng/llama-cpp-python"
fi
echo

# --- Step 5: MLX backend (Apple Silicon only) ---
echo "[5/5] Installing the MLX backend..."
if [[ "$ARCH" == "arm64" ]]; then
    uv pip install --python .venv/bin/python mlx-vlm
    echo "      mlx-vlm installed (MLX models will appear in the model list)."
else
    echo "      Skipped — MLX requires Apple Silicon."
fi
echo

# --- Verify ---
echo "Verifying the engines load..."
.venv/bin/python -c "import llama_cpp; print('      llama.cpp OK -', llama_cpp.__version__)"
if [[ "$ARCH" == "arm64" ]]; then
    .venv/bin/python -c "import mlx_vlm; print('      MLX OK')" || true
fi

echo
echo "============================================================"
echo "  Setup complete!"
echo
echo "  To launch the app:            ./run.sh"
echo "  If anything goes wrong:       .venv/bin/python doctor.py"
echo "============================================================"
