# Contributing

Thanks for your interest in **QWEN 3 VL ABL Captioner**! This is a small,
focused desktop app for building captioned image datasets, and contributions
of all sizes are welcome — bug reports, model suggestions, docs, and code.

## Ways to contribute

### 🐛 Report a bug
Open an [issue](https://github.com/GitDonkeyHubbed/qwen3vl-captioner/issues) and,
**if it's an install or GPU problem, please include the diagnostics report**:

- **Windows:** double-click `diagnose.bat`
- **macOS / Linux:** `.venv/bin/python doctor.py`

Paste the full report into the issue — it shows your GPU, driver, CUDA Toolkit
(or Metal/MLX), the installed wheel, and whether the engine loads, which makes
problems much faster to pin down.

### 🧠 Suggest a model
Want a different Qwen3-VL model in the dropdown? Open an issue with the
HuggingFace repo link, or send a PR adding it to the registry (see
[Adding a model](#adding-a-model-to-the-registry) below).

### ✨ Request a feature / 💻 send code
Open an issue to discuss first for anything non-trivial, then send a PR.

## Development setup

Python is installed for you by the setup scripts (via [uv](https://astral.sh/uv)),
so you don't need a system Python.

```bash
# Windows
setup.bat

# macOS (Apple Silicon = Metal + MLX; Intel = CPU only)
./setup.sh
```

This creates `.venv/` with the app and the correct GPU engine. Launch with
`run.bat` / `./run.sh`.

### Running the tests

```bash
uv pip install -r requirements-dev.txt          # pytest + pyflakes
python -m pytest tests/ -q                        # core-logic unit tests
```

The tests cover the platform-independent logic (CUDA→wheel mapping, the model
registry, caption cleanup, config). They run headless and need no GPU.

### Before you open a PR

CI runs **Lint & Test** on every PR. Match it locally:

```bash
python -m compileall -q app.py doctor.py engine gui tests   # syntax
pyflakes app.py doctor.py engine/*.py gui/*.py tests/*.py    # dead/undefined names
python -m pytest tests/ -q                                    # tests
```

Keep the codebase **pyflakes-clean** (it does not honor `# noqa`, so remove
unused imports rather than suppressing them).

## Adding a model to the registry

Models live in [`gui/model_download_manager.py`](gui/model_download_manager.py).

**GGUF models** (Windows + macOS) are defined with the `_gguf_family` helper —
give it the display label, the HuggingFace repo id, the filename stem, the
matching `mmproj` (vision encoder) filename, and a `{quant: size_gb}` map:

```python
_MY_MODEL = _gguf_family(
    "My Model 8B",
    "someuser/My-Qwen3-VL-8B-GGUF",
    "My-Qwen3-VL-8B",
    "My-Qwen3-VL-8B.mmproj-f16.gguf",
    {"Q4_K_M": 4.68, "Q6_K": 6.26, "Q8_0": 8.11},
)
```

Then add it to `MODEL_REGISTRY` and to a group in `_GGUF_GROUPS` so it shows in
the dropdown. **Always pair a model with the `mmproj` published in its own repo**
— a mismatched vision encoder degrades captions or crashes the loader.

**MLX models** (Apple Silicon) are folders of safetensors with the vision tower
built in (no `mmproj`). Add an entry to `_MLX_ABL` / `_MLX_STD` / `_MLX_NEXTGEN`
with `repo_id`, `folder`, `size_gb`, and `"backend": "mlx"`.

After editing the registry, run the tests — `tests/test_model_registry.py` checks
registry integrity (every entry resolves, has a valid mmproj, etc.).

## Project layout

```
app.py / doctor.py        entry point + install diagnostics
engine/                   inference backends
  cuda_setup.py             CUDA detection, wheel mapping, DLL loading
  inference.py              Qwen3VLEngine (GGUF via llama.cpp; CUDA/Metal)
  mlx_engine.py             MlxVlmEngine (MLX via mlx-vlm; Apple Silicon)
  base.py                   shared engine interface + caption cleanup
  model_downloader.py       mmproj download helpers
gui/                      PyQt6 UI (main_window, settings_panel, registry, ...)
tests/                    headless core-logic tests
```

## Code of conduct

Please be respectful and constructive. See
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
