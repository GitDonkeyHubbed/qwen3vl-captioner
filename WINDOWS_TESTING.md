# 🔥 Windows GPU Burn Test

> The hands-on test plan for validating a release on a **real NVIDIA GPU**.
> Run it from `main` (or from the release branch you are about to ship).

## Why this test exists

CI proves the Windows install *logic* on a real Windows runner, but CI has
**no GPU**, so actual CUDA model loading and captioning is never exercised
there. This is that burn test — the only check that the CUDA-matched
llama-cpp-python wheel loads and that inference is GPU-fast rather than
silently falling back to the CPU.

Run it whenever any of these change: the pinned llama-cpp-python wheel, the
CUDA toolkit→wheel mapping in `engine/cuda_setup.py`, `setup.bat`, or the
default model in the registry.

---

## Prereqs on the Windows machine

If you already run ComfyUI you almost certainly have these:

- **NVIDIA GPU** + current driver
- **CUDA Toolkit installed** — *this is the whole point of the fix.* Confirm
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\` contains a `v12.x` or
  `v13.x` folder. If it's missing: `winget install Nvidia.CUDA`
  *(the driver alone is not enough — `ggml-cuda.dll` needs the Toolkit's runtime DLLs).*
- **Git** (to get the code). You do **not** need Python pre-installed —
  `setup.bat` installs Python 3.12 itself via `uv`.

---

## Steps

### 1. Get the code
Cloning and pulling are read-only and need no push credentials:

```bat
git clone https://github.com/GitDonkeyHubbed/qwen3vl-captioner.git
cd qwen3vl-captioner
```

If you already cloned it:

```bat
git checkout main
git pull
```

To test an unmerged release branch, substitute its name for `main` above.

### 2. Run `setup.bat` (double-click) — watch two lines
- **`[4/6] Detecting CUDA Toolkit`** → should print your version and pick a
  wheel tag. Toolkit → tag mapping (all five exist for the pinned `0.3.40` release):

  | CUDA Toolkit | Wheel tag |
  |--------------|-----------|
  | 13.1+ | `cu131` |
  | 13.0  | `cu130` |
  | 12.8 – 12.9 | `cu128` |
  | 12.6 – 12.7 | `cu126` |
  | 12.4 – 12.5 | `cu124` |

- **`[6/6] Verifying the engine loads`** → must print
  **`Engine OK - llama_cpp 0.3.40`**. (This is the single most important line —
  it proves the new wheel loaded.)

### 3. If setup fails → `diagnose.bat`
Double-click it, copy the full report. It tells us exactly what broke (missing
Toolkit, wheel/CUDA mismatch, import failure). Paste it into the chat / a GitHub
issue.

### 4. Launch the app
Double-click `run.bat`. (It adds the newest CUDA Toolkit `bin` to `PATH` so the
DLLs resolve — prevents the "access violation" failure mode.)

### 5. In-app walkthrough
- Dropdown default = **Qwen3-VL 8B ABL v2 — Q6_K (~6.26 GB)**
  (`prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF`). Click the
  **⬇ download** button → watch the **real % / GB progress bar**, and confirm it
  **auto-downloads the matching mmproj** (`...abliterated-v2.mmproj-f16.gguf`)
  right after.
- **Models land one level *above* the app folder.** If the repo is at
  `E:\qwen3vl-captioner\`, the `.gguf` files go to `E:\`.
- Click **Load Model** → watch the **GPU pill / VRAM** climb. This confirms CUDA
  engaged (GPU), not a silent CPU fallback (slow).
- Import a test image → **Regenerate Caption** → confirm tokens **stream out**.
- Run **Batch Caption All** on 2–3 images.
- *(Optional)* try **Qwen3-VL 8B Caption-it**
  (`prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it-GGUF`) and
  **Huihui Qwen3-VL 8B ABL** (`noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF`).

### 6. Specifically watch for
1. Does the pinned wheel **load cleanly**? (`Engine OK - llama_cpp 0.3.40`)
2. Does the default model **and its matching mmproj** download and load?
3. Is inference **GPU-fast**, not CPU-slow?

---

## After the test

### ✅ If it all works — cut the release
Make sure every in-repo version agrees (`gui/version.py`, `pyproject.toml`, the
README title and badge, the newest CHANGELOG entry — `tests/test_version_sync.py`
checks this), then push the tag:

```bat
git tag V1.4.3 && git push origin V1.4.3
```

The release workflow re-verifies the tag against those versions and refuses to
publish on any mismatch. It can also be run manually from the Actions tab.

### ⚠️ If the pinned wheel misbehaves (but everything else is fine)

Roll the wheel back by hand — there is no pre-staged rollback branch.

<details><summary>Applying a wheel rollback (two edits)</summary>

Roll the wheel back to the community-tested `0.3.24` while keeping the model
refresh. Two edits:

1. **`setup.bat`** — replace the v1.4.0 `WHEEL_URL` line with the v1.3.0 form.
   Note the URL *format* changed between releases (not just the version number),
   so replace the whole line:

   ```bat
   REM v1.4.0 (current):
   set "WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.40-!CUDA_WHEEL!-win-20260608/llama_cpp_python-0.3.40%%2B!CUDA_WHEEL!-cp312-cp312-win_amd64.whl"

   REM v1.3.0 rollback:
   set "WHEEL_URL=https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.24-!CUDA_WHEEL!-Basic-win-20260208/llama_cpp_python-0.3.24%%2B!CUDA_WHEEL!.basic-cp312-cp312-win_amd64.whl"
   ```

2. **`engine/cuda_setup.py`** — drop the `((13, 1), "cu131"),` row from
   `_WHEEL_TAGS` (the `0.3.24` release set has no `cu131` build, so a CUDA 13.1
   box must fall back to `cu130`). Leave the rest of the file alone — in
   particular keep the `pynvml` FutureWarning suppression in `diagnose()`.

> Edit both files by hand and re-run `setup.bat`. Send the `diagnose.bat`
> output with the report so the cause can be pinned down.

</details>

---

## Quick reference — what each script does
| Script | Role |
|--------|------|
| `setup.bat` | Installs `uv` → Python 3.12 → deps → the CUDA-matched `0.3.40` wheel, then verifies the engine imports. Re-run it if you install/upgrade CUDA. |
| `run.bat` | Adds newest CUDA Toolkit `bin` to `PATH`, then launches `app.py`. |
| `diagnose.bat` | One-click full report (GPU, driver, Toolkit, wheel build, engine import) with fixes. First stop for any problem. |
