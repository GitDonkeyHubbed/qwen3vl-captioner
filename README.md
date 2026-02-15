<p align="center">
  <img src="assets/VL_GGUF_Captioner GUI Screenshot 2.png" alt="VL-CAPTIONER Studio Pro" width="900"/>
</p>

<h1 align="center"> Uncensored QWEN 3 VL-CAPTIONER Studio Pro</h1>
<h3 align="center">GPU-Accelerated Image Captioning with Qwen3-VL — Desktop GUI</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.12-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/engine-llama.cpp-orange?logo=llama" alt="Engine"/>
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.x-green?logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/license-MIT-purple" alt="License"/>
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey?logo=windows" alt="Platform"/>
</p>

---

A fully offline, portable desktop application for generating high-quality image captions using the **Qwen3-VL 8B** vision-language model. Built with a professional PyQt6 dark-themed GUI, GGUF quantized model inference via **llama-cpp-python**, and full CUDA GPU acceleration.

Designed for AI artists, dataset curators, and Stable Diffusion / Flux trainers who need accurate, customizable captions for their image datasets.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🖼️ One-click folder import** | Load an entire dataset directory at once — all images appear in a searchable sidebar |
| **⚡ Batch captioning** | Caption all imported images sequentially with a single click and progress tracking |
| **🎯 7 target presets** | Optimized prompts for Stable Diffusion, Flux 1, Flux 2, Z-Image, Chroma, Pony (SDXL), and Qwen Image |
| **📐 Adjustable quality** | Control caption length (short/medium/long/very long), max tokens, and temperature |
| **✏️ Custom prompts** | Write your own captioning prompt or edit the preset templates |
| **🏷️ Prefix & Suffix** | Automatically prepend/append fixed text to every caption (e.g., `sks style,`, `masterpiece`) |
| **💾 Export to .txt** | Save all captions as sidecar `.txt` files alongside the original images |
| **📊 Dataset view** | Table view of all images with dimensions, file size, and caption status |
| **🎮 Real-time GPU monitoring** | Live VRAM usage percentage and RAM stats in the nav bar |
| **📥 Built-in model downloader** | Download GGUF models directly from HuggingFace within the app |
| **🔔 Notification system** | Activity log with bell icon badge for model loads, batch completion, etc. |
| **🎨 Premium dark theme** | Zinc-based color palette with blue accent, smooth transitions, and modern aesthetics |

---

## 📸 Screenshots

<p align="center">
  <img src="assets/screenshot.png" alt="Main Interface" width="800"/>
  <img src="assets/VL_GGUF_Captioner GUI Screenshot 2.png" alt="Main Interface" width="800"/>
  <img src="assets/VL_GGUF_Captioner GUI Screenshot 4.png" alt="Main Interface" width="800"/> 
  <br/>
  <em>Main workspace: file browser, image viewer, caption editor, and model settings</em>
</p>

---

## 🚀 Quick Start (Windows)

### Prerequisites
- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with 8+ GB VRAM (recommended for 8B model)
- **CUDA 12.x drivers** installed ([Download from NVIDIA](https://developer.nvidia.com/cuda-downloads))

### Installation

**1. Clone or download this repository:**
```bash
git clone https://github.com/GitDonkeyHubbed/qwen3vl-captioner.git
cd qwen3vl-captioner
```

**2. Run the automated setup:**
```batch
setup.bat
```

This will:
- Install [uv](https://astral.sh/uv) (fast Python package manager) if not present
- Install Python 3.12 via uv
- Create a virtual environment (`.venv/`)
- Install all dependencies (PyQt6, Pillow, huggingface-hub, etc.)
- Install [JamePeng's llama-cpp-python fork](https://github.com/JamePeng/llama-cpp-python) with Qwen3-VL + CUDA 12.4 support

**3. Download the GGUF model files:**

You need two files — the main model and the vision encoder. Place them in the **parent directory** of this app (one level above `qwen3vl-captioner/`):

| File | Size | Link |
|------|------|------|
| `Qwen3-VL-8B-Instruct-abliterated-v1.Q6_K.gguf` | ~6.3 GB | [HuggingFace](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF) |
| `Qwen3-VL-8B-Instruct-abliterated-v1.mmproj-f16.gguf` | ~1.1 GB | Same repo |

> **Tip:** You can also download models from within the app using the built-in downloader (⬇ button next to the model selector).

**4. Launch the app:**
```batch
run.bat
```

---

## 📁 Project Structure

```
qwen3vl-captioner/
├── app.py                  # Application entry point
├── run.bat                 # Launch script (Windows)
├── setup.bat               # Automated installer (Windows)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
│
├── engine/                 # Inference backend
│   ├── __init__.py
│   ├── inference.py        # Qwen3VLEngine — GGUF model loading & captioning
│   └── model_downloader.py # HuggingFace model download manager
│
├── gui/                    # PyQt6 user interface
│   ├── __init__.py
│   ├── main_window.py      # Main window orchestrator
│   ├── settings_panel.py   # Right panel — presets, parameters, batch controls
│   ├── file_browser.py     # Left sidebar — thumbnail list with search
│   ├── image_viewer.py     # Center — image preview with zoom controls
│   ├── caption_panel.py    # Bottom — caption display, edit, and save
│   ├── dataset_panel.py    # Dataset table view
│   ├── theme.py            # Dark/light theme color system & QSS stylesheet
│   ├── config.py           # User config persistence (~/.vlcaptioner/)
│   ├── notification_panel.py
│   ├── app_settings_dialog.py
│   └── model_download_manager.py
│
└── assets/
    └── screenshot.png      # GUI screenshot for README
```

---

## 🔧 How It Works

### Engine
The app uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (JamePeng's fork with Qwen3-VL support) to run **GGUF quantized** models directly on your GPU via CUDA. No cloud API, no internet required after model download.

- **Model:** Qwen3-VL 8B Instruct (abliterated variant for uncensored captioning)
- **Quantization:** Q6_K (~6.3 GB) or Q8_0 (~8.1 GB) — excellent quality-to-size ratio
- **Vision encoder:** Separate mmproj file handles image understanding
- **Inference:** GPU-accelerated with streaming token output

### Captioning Workflow

1. **Open Folder** → Select your dataset directory (all images load instantly)
2. **Load Model** → One-click model loading with CUDA auto-detection
3. **Configure** → Choose a target preset (SD, Flux, etc.), adjust length and temperature
4. **Caption** → Click individual images + "Regenerate Caption", or "Batch Caption All" for the entire dataset
5. **Export** → Save all captions as `.txt` sidecar files next to the originals

### Target Presets

| Preset | Use Case |
|--------|----------|
| **Stable Diffusion** | Comma-separated booru-style tags for SD 1.5 / SDXL training |
| **Flux 1** | Natural language descriptions optimized for Black Forest Labs Flux.1 |
| **Flux 2** | Updated format for Flux.2 model training |
| **Z-Image** | Structured captions for Z-Image architecture |
| **Chroma** | Scene descriptions for Chroma model fine-tuning |
| **Pony (SDXL)** | Pony Diffusion V6 tag format with quality markers |
| **Qwen Image** | General-purpose detailed image descriptions |

---

## ⌨️ Manual Installation (Advanced)

If `setup.bat` doesn't work or you prefer manual setup:

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install llama-cpp-python with CUDA support
# Download the appropriate wheel from:
# https://github.com/JamePeng/llama-cpp-python/releases
pip install llama_cpp_python-0.3.24+cu124.basic-cp312-cp312-win_amd64.whl

# 4. Run
python app.py
```

### Linux / macOS (Experimental)

The GUI is cross-platform (PyQt6) but the setup scripts are Windows-only. For other platforms:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For CUDA on Linux:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# For Metal on macOS:
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

python app.py
```

> **Note:** Linux/macOS support is experimental. The CUDA DLL preloading in `engine/inference.py` is Windows-specific but will be safely skipped on other platforms.

---

## 🛠️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 64-bit | Windows 11 |
| **GPU** | NVIDIA GTX 1070 (8 GB) | NVIDIA RTX 3060+ (12 GB) |
| **VRAM** | 8 GB | 12+ GB |
| **RAM** | 16 GB | 32 GB |
| **Storage** | ~10 GB (model + app) | ~15 GB (both quants) |
| **CUDA** | 12.0 | 12.4 |
| **Python** | 3.11 | 3.12 |

---

## 📝 Caption Output Format

Captions are saved as plain `.txt` files with the same name as the image:

```
my_image.jpg     →  my_image.txt
photo_001.png    →  photo_001.txt
```

This is the standard sidecar format expected by most training tools (Kohya, EveryDream, SimpleTuner, etc.).

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Model not found"** | Place `.gguf` files in the parent directory (one level above `qwen3vl-captioner/`) |
| **"CUDA not available"** | Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) and restart |
| **Blank image preview** | Fixed in v1.4.2 — Qt image allocation limit raised to handle large files |
| **Slow model loading** | Normal — first load takes 30-60s. Subsequent loads are faster |
| **Out of VRAM** | Use Q6_K instead of Q8_0, or reduce `max_tokens` |
| **"access violation"** | CUDA DLLs not found. Run `run.bat` (sets PATH automatically) |

---

## 🤝 Credits

- **[Qwen3-VL](https://huggingface.co/Qwen)** — Vision-language model by Alibaba DAMO Academy
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Python bindings for llama.cpp
- **[JamePeng's fork](https://github.com/JamePeng/llama-cpp-python)** — Added Qwen3-VL chat handler support
- **[prithi's GGUF quants](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF)** — High-quality GGUF model quantizations
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** — Cross-platform GUI framework

---

## 📄 License

[MIT License](LICENSE) — free to use, modify, and distribute.

---

<p align="center">
  <strong>Made with ❤️ for the AI art community</strong>
  <br/>
  <em>If this tool helps your workflow, consider giving it a ⭐!</em>
</p>
