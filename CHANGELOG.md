# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions correspond to
git tags (`V1.x.x`).

## [1.4.2] — 2026-06-24

Security and dependency maintenance release. No new features.

### Security
- **Pillow floor bumped to >=12.2.0** — fixes 5 CVEs (2 HIGH, 3 MODERATE)
  including integer overflow / OOB writes (PSD, fonts), a FITS decompression
  bomb, and a PDF trailer DoS. Users on older Pillow builds were exposed when
  loading images.

### Changed
- `nvidia-ml-py>=12.0` replaces deprecated `pynvml` package (same module, no
  behaviour change — eliminates a FutureWarning on import).
- `huggingface-hub>=0.32` floor raised; `hf_xet>=1.0` added as explicit
  dependency (high-performance HuggingFace transfer was already used but not
  pinned).

### Infrastructure
- CI doctor step (`continue-on-error`) restored — Windows smoke test had
  regressed to always-fail after a PowerShell incompatibility.
- Added `SECURITY.md` with private vulnerability reporting instructions.

---

## [1.4.1] — 2026-06-16

Hardening release from a deep multi-agent code review (19 confirmed findings,
each adversarially verified against the source). No new features.

### Fixed
- **Batch: the last image now completes correctly.** The batch queue was popped
  one item too early, so the final image fell through to the single-image path
  (popping a per-image save dialog) and the batch never reported completion.
  Batch state is now tracked by an explicit flag.
- **Parallel download corruption guards (the big ones).**
  - A crashed parallel download left a full-size, hole-filled `.part` that the
    resume path could finalize as a "complete" but corrupt model. A `.parallel`
    marker now flags and discards that artifact.
  - Parallel segments now verify each response is HTTP 206; a `200` (Range
    ignored) is retried instead of silently corrupting the assembled file, and
    each segment is bounded to its own length.
- **Shutdown safety.** On window close the app no longer frees the model while a
  load/caption worker may still be using it (could crash llama.cpp); an active
  download is now actually cancelled; and the model load can no longer be
  started twice concurrently through the mmproj dialogs.
- **Cancelled captions** are no longer saved as truncated text.
- **MLX folder downloads** can now be cancelled (between files) and show real
  progress, instead of ignoring Stop until the multi-GB folder finished.
- **MLX backend now applies the system prompt** (engine parity with GGUF).
- Single-stream finalize uses `replace` (was `rename`, which raised on Windows
  if the target existed); resume validates the `.part` against the remote size
  before trusting it.
- PIL image file handle is released deterministically per caption (was leaked /
  could lock the file on Windows during batch runs).
- The **"Refer as {name}"** option now has a real name input (was always
  "the subject"); VRAM-fit hints and auto-select now apply to MLX models on
  Apple Silicon; "Already downloaded" checks all model search dirs.

## [1.4.0] — 2026-06-16

### Added
- **macOS support.** Apple Silicon GPU acceleration via llama.cpp **Metal**
  (Tier 1) and an **MLX** backend via `mlx-vlm` (Tier 2). New `setup.sh` /
  `run.sh` entry points. See [MAC_TESTING.md](MAC_TESTING.md).
- **Parallel model downloader.** Large GGUF files download over 8 concurrent
  HTTP range connections — roughly **3× faster** than the previous single
  stream (HuggingFace's CDN throttles each connection to ~25–30 MB/s). Includes
  per-segment **retry/resume** and a 60s socket timeout so a transient network
  blip no longer fails the whole download.
- **Stop button.** Cancel an in-progress model download directly from the
  status bar; a user-cancelled download **discards its partial file** so a
  different model can be selected immediately.
- **HuggingFace Xet transfer.** `hf_xet` high-performance transfer is enabled
  for the library download paths (`hf_hub_download`, `snapshot_download` — used
  by the vision encoder and MLX folder downloads).
- 2026 model refresh: new default **Qwen3-VL 8B Abliterated v2** (Q2_K…f16),
  plus Caption-it and Huihui families; MLX model entries on Apple Silicon.

### Changed
- Upgraded to **llama-cpp-python 0.3.40** (adds Qwen3.5 / 3.6 model-family
  support). The CUDA wheel is auto-matched to the installed Toolkit
  (`cu124`/`cu126`/`cu128`/`cu130`/`cu131`); the Metal wheel is used on Apple
  Silicon.
- Replaced the deprecated `pynvml` package with **`nvidia-ml-py`** (same NVML
  API, no deprecation warning on startup).

### Fixed
- **Vision-encoder mismatch crash.** The loader previously paired a model with
  the *first* `*mmproj*.gguf` found in the folder, so selecting a model whose
  own encoder wasn't downloaded could load a **mismatched** vision encoder and
  crash llama.cpp natively (no Python traceback). The loader now resolves each
  model's matching `mmproj_filename`, refuses to mispair, and offers to download
  the correct encoder when it's missing.
- Lint: detect `hf_xet` via `importlib.util.find_spec` instead of an unused
  import (pyflakes does not honor `# noqa`).

### Verified
- **Windows / CUDA** — RTX 4080: `0.3.40` cu124 wheel loads, parallel downloads,
  ~7 s/caption on GPU.
- **macOS / Apple Silicon** — M4: clean `./setup.sh` installs the Metal wheel
  and `mlx-vlm`; both backends caption end-to-end — Tier 1 llama.cpp **Metal**
  (GGUF + mmproj) ~5–6 s, Tier 2 **MLX** (std / abliterated / Qwen3.5 next-gen)
  ~4–6 s. See [MAC_TESTING.md](MAC_TESTING.md).

## [1.3.0]
- CUDA-matched installs, custom local models, diagnostics (`diagnose.bat` /
  `doctor.py`), and quality-of-life improvements.

## [1.2.0] / [1.1.0]
- Earlier releases — see the `V1.2.0` and `V1.1.0` git tags.
