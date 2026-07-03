# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions correspond to
git tags (`V1.x.x`).

## [1.4.3] — Unreleased

Maintenance release from a full repository health check, followed by a second
deep QC audit of the entire codebase (56 findings, each adversarially
verified against the source — several reproduced empirically against live
PyQt6 before fixing).

### Fixed (QC audit round 2)
- **Caption/save misattribution race (data corruption).** A finished caption
  was cached and auto-saved under whatever image was *selected at completion
  time*, not the image it was generated for — clicking thumbnail B while A's
  caption streamed silently overwrote `B.txt` with A's caption. Results are
  now pinned to the generating worker's own image, and batch items generate
  the popped queue entry even if the selection changes mid-run.
- **Sideways captions for phone photos.** The GGUF path never applied the
  EXIF Orientation tag, so rotated camera JPEGs were captioned as a sideways
  scene — invisibly, because the Qt preview auto-rotates. Orientation is now
  applied before encoding; extreme aspect ratios also no longer crash resize
  with a zero dimension (which aborted the rest of a batch).
- **Saved theme ignored at startup.** The app always launched dark; the
  persisted light-mode choice now applies on launch, translucent surfaces got
  light-theme palette entries (they were hardcoded dark rgba), invalid QSS
  (`::placeholder`, `letter-spacing`, `text-transform`, `line-height`) was
  removed, and placeholder text is colored via `QPalette` as Qt requires.
- **Resume regression from 1.4.3's own sidecar check.** Valid `.part` files
  from pre-1.4.3 versions (which never had a `.meta` sidecar) were silently
  discarded — multi-GB downloads restarted from zero on upgrade. Legacy
  partials are now grandfathered via the old size heuristic and stamped with
  a sidecar; the HTTP 416 finalize path also falls back to the sidecar's
  recorded size when the live probe fails, instead of dead-ending.
- **"Uncancellable" encoder dialog was dismissible with Esc**, silently
  dropping application modality while the nested event loop still ran. The
  dialog now genuinely ignores Esc/close until the download finishes.
- **CUDA 13.x DLL preload was a silent no-op** (only the first `bin` dir was
  globbed; CUDA 13 keeps its DLLs in `bin\x64`), and `doctor.py` reported a
  false "[OK] Wheel/CUDA match" for toolkits older than 12.4. Doctor also no
  longer fails a healthy Mac over the *optional* MLX backend, and exits 2 on
  an internal crash (CI normalizes only exit 1).
- **Batch state machine hardening.** Unload is refused during the between-item
  gap (previously stranded a zombie batch), batch start is re-entrancy-guarded,
  the batch button no longer re-enables while the last item is in flight,
  "Clear all" cancels an in-flight generation instead of orphaning it, and the
  stop-download confirm can no longer cancel a different (chained) download.
- **hf_token hardening.** `~/.vlcaptioner/` is created `0700` and config.json
  written `0600`; the download token is now also stripped on same-host
  HTTPS→HTTP redirect downgrades; the update-check result is HTML-escaped
  before rendering in a link-enabled label.
- `pip install .` was broken (setuptools flat-layout refusal) — explicit
  package discovery added, so the `qwen3vl-captioner` console script installs.
- Snapshot downloads reject path-traversal filenames (defense-in-depth); the
  status-bar RAM readout refreshes on the timer instead of only at startup.

### Added (QC audit round 2)
- **Download speed + ETA** in every download progress message, and a
  time-remaining estimate in the batch queue label.
- **Keyboard shortcuts**: Ctrl+S save caption, Ctrl+G generate,
  Ctrl+←/→ (and PgUp/PgDn) previous/next image.
- **Drag & drop anywhere** on the window (was: only the file-browser strip).
- "Model not downloaded" dialog now offers to download the selected registry
  model directly; the maximize button in the viewer toolbar actually works;
  import dialogs remember the last-used folder; model-load errors show the
  message with the traceback tucked into expandable details; a warning is
  raised when images share a stem (they'd share one `.txt` caption).
- **20 new tests** (113 total): EXIF/aspect-ratio image prep, `.part`
  identity-sidecar contract, config save-failure surfacing, last-import-dir,
  and a real truth-table for `mlx_backend_supported` (the old test re-derived
  the production expression as its own oracle).
- **CI hardening**: workflows get least-privilege `permissions` blocks; the
  Windows smoke test parses the wheel URL from `setup.bat` (single source)
  and a new job HEAD-checks all five pinned CUDA wheel URLs so a deleted
  release fails CI instead of user installs; `requirements-dev.txt` is now
  actually consumed by CI; the `uv` bootstrap installers are version-pinned.

### Fixed
- **Version string lag.** `gui/version.py` and `pyproject.toml` were still
  `1.4.1` after the 1.4.2 release, so the in-app "Check for Updates" reported a
  phantom update to every up-to-date user. Versions are back in sync.
- **GUI froze during a vision-encoder download.** When a model's `mmproj` was
  missing, the Load flow downloaded it synchronously on the Qt UI thread,
  freezing the window with no progress for the whole multi-hundred-MB transfer.
  It now runs off-thread behind a responsive modal dialog.
- **Shutdown safety.** Closing the window mid-download no longer risks
  destroying a still-running download thread — the `wait()` result is honored
  like the load/caption threads, and earlier shutting-down threads are joined
  before the engine is freed.
- **Download integrity.** A stale `.part` left under the same name by a
  different file can no longer be silently appended onto and finalized as a
  corrupt model — partials are validated against an identity sidecar before
  resume. Downloads now also pre-flight free disk space and fail fast with a
  clear message.
- **Atomic config writes.** `config.json` is written to a temp file and
  `os.replace`d into place, so an interrupted save can't truncate it (which
  load silently discarded, losing the hf_token / custom models / theme).
- **Caption robustness.** The non-streaming path no longer crashes when the
  model returns `null` content; the streaming path tolerates an empty
  `choices` list.
- **Presets.** Toggling a preset off now restores the user's own
  prefix/suffix instead of leaving the preset's tokens applied; the
  "Refer-as name" field now emits `settings_changed` like other inputs.
- **Deterministic encoder pick.** `find_mmproj_file` sorts (preferring `f16`)
  when a folder holds more than one encoder, instead of relying on filesystem
  order.
- **Misc.** `diagnose.bat` now evaluates `where python` correctly (delayed
  expansion); removed two stray `.gitignore` patterns; the `mlx` extra floor is
  raised to `mlx-vlm>=0.6` to match the shipped next-gen MLX models.

### Docs
- README MLX lineup, "default quant" wording, version header, and
  project-structure tree corrected; `CONTRIBUTING.md` now points at the real
  registry test (`tests/test_model_registry.py`).

### Known limitations
- Cancelling an MLX folder (snapshot) download still only takes effect between
  files — a large shard already in flight must finish first — but the UI now
  says so instead of looking hung.

---

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
