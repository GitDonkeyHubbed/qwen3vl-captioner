<p align="center">
  <img src="assets/VL_GGUF_Captioner GUI Screenshot 2.png" alt="QWEN 3 VL ABL Captioner" width="900"/>
</p>

<h1 align="center">QWEN 3 VL ABL Captioner V1.4.3 — GGUF + MLX Engines</h1>
<h3 align="center">Professional GPU-Accelerated Image Captioning for Datasets</h3>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.4.3-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/GPU-CUDA%2012.4%E2%80%9313.x-green?logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Apple%20Silicon-Metal%20%2B%20MLX-black?logo=apple" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey" alt="Platform"/>
</p>

<p align="center">
  <a href="https://github.com/GitDonkeyHubbed/qwen3vl-captioner/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/%E2%AC%87%20DOWNLOAD%20LATEST%20VERSION%20%E2%80%94%20Click%20Here-2EA043?style=for-the-badge&logo=github&logoColor=white" alt="Download Latest Version" height="46"/>
  </a>
</p>

<p align="center">
  <b>New here? Just click the green button above — it downloads the whole app as a <code>.zip</code>.</b>
</p>

### ✅ Get started in 3 easy steps

1. **Download & unzip** — click the big green button above, then unzip the file anywhere (Desktop is fine).
2. **Install it (one time):**
   - **Windows** → double-click **`setup.bat`**
   - **macOS** → run **`./setup.sh`**

   This automatically installs Python and the GPU engine for you. Just wait for it to finish.
3. **Run the app:**
   - **Windows** → double-click **`run.bat`**
   - **macOS** → run **`./run.sh`**

> 🪟 **Windows note:** you also need the NVIDIA **CUDA Toolkit** for GPU speed. If you don't have it, install it with one command: `winget install Nvidia.CUDA`

<sub>🧰 <b>Power users:</b> prefer a specific tagged release? Grab it from the <a href="https://github.com/GitDonkeyHubbed/qwen3vl-captioner/releases/latest">Releases page</a>.</sub>

---

## 🩺 What's New in V1.4.3 — Health-Check Fixes

No new features — a sweep of bug and consistency fixes from a full repository health check.

- **No more phantom "update available" popup** — the in-app version was stuck at 1.4.1 and now matches the build.
- **The window no longer freezes** while downloading a model's vision encoder — it runs in the background behind a progress dialog instead of going *Not Responding*.
- **Safer downloads & settings** — an interrupted download can no longer be finalized as a corrupt model, downloads check free disk space first, and `config.json` is written atomically so a crash can't wipe your token/settings.
- **Preset fix** — turning a preset off now restores *your* prefix/suffix instead of leaving the preset's tags applied.
- Plus smaller robustness fixes (caption edge cases, deterministic encoder pick, a `diagnose.bat` bug, and doc corrections).

---

## 🔒 What's New in V1.4.2 — Security & Dependency Maintenance

No new features — just keeping things safe and clean for everyone.

- **Pillow patched to >=12.2.0** — fixes 5 CVEs (2 HIGH, 3 MODERATE): integer overflow / OOB writes when loading certain PSD and font files, a FITS decompression bomb, and a PDF trailer denial-of-service. If you're on an older install, run `setup.bat` / `setup.sh` again or `pip install --upgrade Pillow` inside your venv.
- `nvidia-ml-py>=12.0` replaces the deprecated `pynvml` package — same module, eliminates an import FutureWarning for NVIDIA GPU users.
- `huggingface-hub>=0.32` floor raised; `hf_xet>=1.0` pinned as an explicit dependency.
- Windows smoke CI restored after a PowerShell incompatibility broke it silently.
- Added `SECURITY.md` — vulnerability reports now go through GitHub's private advisory system.

---

## 🚀 What's New in V1.4.0 — macOS Support

The captioner now runs natively on Macs, with **two** GPU backends:

### 🍎 Apple Silicon: Metal + MLX
- **llama.cpp Metal engine** — the same GGUF models (including the abliterated default) now run GPU-accelerated on M-series chips. Verified end-to-end on Apple Silicon: ~7s per caption on the Q2_K quant.
- **New MLX engine** — Apple's [MLX](https://github.com/ml-explore/mlx) framework via `mlx-vlm`, typically the fastest option on M-series chips. MLX models (4/6/8-bit) appear in the model dropdown on Apple Silicon and download with one click — no mmproj file needed, the vision tower is built in.
- **One-command setup** — `./setup.sh` installs everything: Python, the Metal wheel, and the MLX backend.
- The hardware pill shows **unified memory** pressure on Macs instead of CUDA VRAM.

### 🆕 2026 model refresh
The model dropdown is reorganized into groups (newest first):
- **Qwen3-VL 8B ABL v2** *(new recommended default)* — [prithivMLmods' v2 abliteration](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF), full quant range
- **Qwen3-VL 8B Caption-it** — an abliterated fine-tune [specialized for image captioning](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it-GGUF)
- **Huihui Qwen3-VL 8B ABL** — [huihui-ai's abliteration](https://huggingface.co/noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF) (quantized by noctrex)
- **Legacy v1** — kept for existing installs
- **MLX (Apple Silicon)** — several groups in the dropdown: **Abliterated v2** (Qwen3-VL 8B, 4/6/8-bit — the recommended Mac default), **Gliese Caption** (Qwen3.5, 0.8B–9B, captioning-tuned), **Qwen3-VL c_abliterated** (newest), plus **huihui abliterated**, **standard**, and an experimental **Qwen3.5** group

Downloading a model now also **auto-downloads its matching mmproj** (vision encoder) when none is present.

### ⚙️ Engine bump: llama-cpp-python 0.3.40
Both platforms now use JamePeng's v0.3.40 build (Windows: cu124–cu131 auto-matched; macOS: Metal), which adds support for the **Qwen3.5 / Qwen3.6-generation** GGUF models — including the larger abliterated 27B/35B-A3B releases for big-VRAM rigs.

---

## 🚀 What's New in V1.3.0

### 🔧 Installation that just works (fixes #8, #10)
- `setup.bat` now **detects your CUDA Toolkit version** (12.4 / 12.6 / 12.8 / 13.x) and installs the **matching** llama-cpp-python wheel — no more `ggml.dll` failures from wheel/toolkit mismatches.
- Setup **verifies the engine actually loads** before declaring success.
- New **`diagnose.bat`**: one double-click prints a full report (GPU, driver, CUDA Toolkit, wheel build, engine import) with specific fixes. Paste its output into any GitHub issue.
- If the engine fails, the app now shows **exactly what's wrong and how to fix it** in a dialog — not just a console traceback.

### 📁 Load any GGUF model (fixes #7)
- New **Browse button** next to the model dropdown — load any Qwen3-VL compatible GGUF from anywhere on disk (BF16, abliterated v2, other quants).
- Custom models are **remembered** between sessions, and any unknown GGUF files found in the model folder appear in the dropdown automatically.
- If no mmproj is found next to a custom model, the app asks whether to download the default or lets you **browse for the matching mmproj**.

### ✨ Quality of life
- **Real download progress** — model downloads show actual percentage and GB progress, **Cancel works mid-download**, and interrupted downloads **resume where they left off**.
- **VRAM-aware model list** — quants that won't fit your GPU are tinted red ("won't fit") or orange ("tight fit") with tooltips, and the default selection is the best quant for your card.
- **✓ markers** in the model dropdown show which models are already downloaded.
- **Auto-save captions** checkbox — saves every caption as a `.txt` sidecar instantly, no more popup per image.
- **Max tokens** raised to 2048.
- **Check for Updates** button in Settings (gear icon) — compares your version against the latest GitHub release.

---

## 🚀 What's New in V1.2.0

This release brings a major overhaul to how captions are generated, focusing on **accuracy, anatomy, and detail** over "storytelling" fluff.

### 🏥 Clinical Precision Mode
We've completely rewritten the prompts for all models (Flux, Stable Diffusion, Pony, etc.). Instead of "cinematic" or "moody" descriptions, the engine now focuses on:
- **Physical Reality**: Exact shapes, textures, and spatial relations.
- **Accurate Anatomy**: Detailed descriptions of bodies and poses without euphemisms.
- **Objective Detail**: "Horses through" the image content, listing exactly what is there.

### 🔞 Uncensored / Adult Detail Option
A new **"Uncensored / Adult Detail"** checkbox in the settings.
When enabled, this injects explicit instructions to describe **all** content (including nudity and adult themes) with full anatomical accuracy, bypassing standard safety refusals. Essential for high-quality dataset training.

---

## ✨ Key Features

- **Clinical Precision**: Using anatomically accurate, objective language instead of "creative writing" style. Designed for training, not storytelling.
- **Universal "Edit" Mode**: Full control via the Edit button to handle any prompt format (JSON, XML, Booru) without needing complex hardcoded "modes".
- **Lean Architecture**: Focused on speed and simplicity. No bloat, just tools that work.
- **Multi-Model Presets**: Pre-configured formats for **Flux 1 & 2**, **Stable Diffusion**, **Pony (SDXL)**, **Z-Image**, and more.
- **Drag & Drop**: Drop images or entire folders directly into the app.
- **Batch Processing**: Caption thousands of images automatically.
- **Smart Model Handling**: Native GGUF support with auto-downloading.
- **Hardware Monitoring**: Real-time GPU VRAM usage display.
- **Safety Controls**: Toggle between "PG" and fully "Uncensored" modes.
-   Drag & Drop Enabled

## 📦 Portable Release

-  This version is fully portable. Models are now detected in the application folder, making it easier to share and install.

---

## 📸 Screenshots

<p align="center">
  <img src="assets/screenshot.png" alt="Main Interface" width="800"/>
  <br/>
  <em>Main workspace: file browser, image viewer, caption editor, and model settings</em>
</p>

---

## 📋 Prerequisites (Windows)

`setup.bat` handles Python and all Python packages automatically — but these must be on your system first:

| Required | Notes |
|----------|-------|
| **Windows 10/11 (64-bit)** | Portable release is Windows-only |
| **NVIDIA GPU (8 GB+ VRAM)** | GTX 1070 minimum; RTX 3060+ recommended |
| **NVIDIA GPU driver (current)** | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| **NVIDIA CUDA Toolkit 12.4+** | **Required — the driver alone is not enough.** Install with `winget install Nvidia.CUDA` or from [CUDA Downloads](https://developer.nvidia.com/cuda-downloads) |
| **~10–15 GB free disk** | App, dependencies, and GGUF model files |

**How CUDA versions are handled:** `setup.bat` detects your installed Toolkit and installs the matching llama-cpp-python build automatically:

| Your CUDA Toolkit | Wheel installed |
|-------------------|-----------------|
| 13.1+ | `cu131` |
| 13.0 | `cu130` |
| 12.8 – 12.9 | `cu128` |
| 12.6 – 12.7 | `cu126` |
| 12.4 – 12.5 | `cu124` |

> ⚠️ If you install or upgrade the CUDA Toolkit **after** running setup, **re-run `setup.bat`** so the matching wheel is installed. Note: your GPU driver may report a newer CUDA *capability* (e.g. 13.3) than the Toolkit you have installed — what matters is the Toolkit on disk.

---

## ⚡ Quick Start

### 1. Install the CUDA Toolkit (once)
```powershell
winget install Nvidia.CUDA
```

### 2. Run Setup
Double-click `setup.bat` to install Python, all dependencies, and the CUDA-matched engine. Setup verifies the engine loads before finishing.

### 3. Get Models
Download models directly inside the app (✓ marks the ones you already have), or click the **📁 Browse** button to load any Qwen3-VL compatible `.gguf` from disk. The app **pre-selects the best model for your GPU** automatically — when in doubt, keep the default and click **Load Model**.

**Not sure which to pick?**

| Your setup | Recommended pick |
|---|---|
| 12 GB+ VRAM (RTX 3060+) | **Qwen3-VL 8B ABL v2 — Q6_K** *(typical pick; the app auto-selects the best quant for your VRAM)* |
| 8 GB VRAM | Qwen3-VL 8B ABL v2 — Q4_K_M |
| Low VRAM / older GPU | Qwen3-VL 8B ABL v2 — Q2_K |
| Best caption quality | Qwen3-VL 8B Caption-it — Q6_K *(captioning-tuned)* |
| Mac (Apple Silicon) | any **MLX** entry (4-bit = smallest, 8-bit = best quality) |

Models too big for your GPU are shown in **red** in the dropdown, so you can't accidentally pick one that won't fit.

### 4. Launch
Double-click `run.bat` to start the captioner.

### Something not working?
Double-click **`diagnose.bat`** — it prints a full report of what's wrong and how to fix it. If you open a GitHub issue, paste that report into it.

---

## 🍎 Quick Start (macOS)

```bash
git clone https://github.com/GitDonkeyHubbed/qwen3vl-captioner.git
cd qwen3vl-captioner
./setup.sh     # installs Python, Metal engine, and MLX backend
./run.sh       # launch the app
```

- **Apple Silicon (M1–M5):** GGUF models run GPU-accelerated via Metal, and MLX models (usually faster) appear in the model dropdown automatically. 16 GB+ unified memory recommended for the 8B models.
- **Intel Macs:** CPU-only (llama.cpp is built from source during setup; MLX is unavailable). Workable, but slow — Apple Silicon is strongly recommended.
- **Diagnostics:** `.venv/bin/python doctor.py` prints the same style of install report as `diagnose.bat` on Windows.

---

## 📁 Project Structure

```
qwen3vl-captioner/
├── app.py                  # Application entry point
├── doctor.py               # Install diagnostics (run via diagnose.bat)
├── run.bat                 # Launch script (Windows)
├── setup.bat               # Automated installer (Windows)
├── diagnose.bat            # One-click install diagnostics (Windows)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
│
├── setup.sh                # Automated installer (macOS)
├── run.sh                  # Launch script (macOS)
│
├── engine/                 # Inference backends
│   ├── __init__.py
│   ├── base.py             # Shared engine interface + caption cleanup
│   ├── cuda_setup.py       # CUDA toolkit detection, DLL loading, diagnostics
│   ├── inference.py        # Qwen3VLEngine — GGUF via llama.cpp (CUDA/Metal)
│   ├── mlx_engine.py       # MlxVlmEngine — MLX via mlx-vlm (Apple Silicon)
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
│   ├── version.py          # single source of truth for APP_VERSION
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

### Using Your Own Models

You're not limited to the built-in model list:

1. Click the **📁 Browse** button next to the model dropdown and pick any Qwen3-VL compatible `.gguf` (e.g. [Qwen3-VL-8B-Instruct-abliterated-v2-GGUF](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF), BF16 quants, etc.).
2. The app looks for an `mmproj` file **in the same folder as the model**. If none is found, it offers to download the default or lets you browse for the right one.
3. Custom models are remembered between sessions, and any `.gguf` files dropped into the app folder show up in the dropdown automatically.

> **Tip:** always pair a model with the mmproj published in its own HuggingFace repo. A mismatched vision encoder degrades caption quality.

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
# setup.bat auto-detects your CUDA version. For manual installs, download
# the wheel matching YOUR CUDA Toolkit from:
# https://github.com/JamePeng/llama-cpp-python/releases
#   13.1+ -> cu131 | 13.0 -> cu130 | 12.8+ -> cu128 | 12.6+ -> cu126 | 12.4+ -> cu124
pip install llama_cpp_python-0.3.40+cu124-cp312-cp312-win_amd64.whl

# 4. Run
python app.py
```

### Linux (Experimental)

The GUI is cross-platform (PyQt6); macOS has first-class support via `./setup.sh`. For Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For CUDA on Linux (JamePeng's fork also publishes linux cu1xx wheels):
CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama_cpp_python @ git+https://github.com/JamePeng/llama-cpp-python"

python app.py
```

> **Note:** Linux support is experimental. The CUDA DLL preloading in `engine/cuda_setup.py` is Windows-specific and safely skipped elsewhere.

---

## 🛠️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 64-bit | Windows 11 |
| **GPU** | NVIDIA GTX 1070 (8 GB) | NVIDIA RTX 3060+ (12 GB) |
| **VRAM** | 8 GB | 12+ GB |
| **RAM** | 16 GB | 32 GB |
| **Storage** | ~10 GB (model + app) | ~15 GB (both quants) |
| **CUDA Toolkit** | 12.4 | 12.8+ or 13.x |
| **Python** | 3.12 (installed by setup.bat) | 3.12 |

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

**First step for any install problem: double-click `diagnose.bat`** — it tells you exactly what's missing or mismatched and how to fix it.

| Issue | Solution |
|-------|----------|
| **"Failed to load ggml.dll"** | CUDA Toolkit missing, or the llama-cpp wheel doesn't match your CUDA version. Run `winget install Nvidia.CUDA`, then re-run `setup.bat`. Run `diagnose.bat` to confirm |
| **"Model not found"** | Use the 📁 Browse button, or place `.gguf` files in the app folder / its parent directory |
| **"CUDA not available"** | Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (not just GPU drivers), then re-run `setup.bat` |
| **Custom model gives bad output** | Make sure the mmproj next to the model matches it — when in doubt, use the mmproj published in the same HuggingFace repo as the model |
| **Blank image preview** | Fixed — Qt image allocation limit raised to handle large files |
| **Slow model loading** | Normal — first load takes 30-60s. Subsequent loads are faster |
| **Out of VRAM** | Use Q6_K instead of Q8_0, or reduce `max_tokens` |
| **"access violation"** | CUDA DLLs not found. Run via `run.bat` (sets PATH automatically) and ensure the CUDA Toolkit is installed |

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
