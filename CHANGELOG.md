# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions correspond to
git tags (`V1.x.x`).

## [Unreleased] — video captioning engine

**Not a release, and not a user-facing feature yet.** This section covers the
*engine* half of issue #26: video captioning exists in `engine/` and in the
test suite, but no GUI code calls it — `caption_video()`, `first_frame()` and
`is_video_file()` have no caller outside `engine/` and `tests/`. The app
still captions images only, `APP_VERSION` stays at the released **1.4.3**,
and the GUI wiring (video files in the picker, a "Frames per video" control,
video thumbnails, video-aware batch runs) is part 2 of #26.

Validated by hand on an RTX 4080: frames arrive in temporal order, 16 frames
take 9.1 s at 9.4 GB peak VRAM, the context preflight refuses cleanly and
cancel leaves nothing running. The Qwen3-VL handler switch was A/B'd over 7
images (OCR, chart-reading and fine-detail cases) with no caption regression.

### Fixed (review findings on the engine work)
- **A reasoning-only response came back as the caption.** `clean_caption`
  falls back to the original text when cleaning empties it, which protects a
  prefix-only caption like `"Caption:"`. A response that is *nothing but* a
  closed `<think>` block also cleans to empty — so the fallback handed the
  trace straight back, the one outcome `strip_reasoning` exists to prevent.
  The fallback now restores the post-reasoning text, so a reasoning-only
  response cleans to `""` (the GUI shows "Nothing to save") while a
  prefix-only one is still preserved.
- **A configured prefix manufactured a caption out of nothing.** Emptying a
  reasoning-only response only helps if the emptiness survives: with a prefix
  set, `apply_prefix_suffix("")` returned `"photo of "`, a truthy string that
  batch auto-save writes to a sidecar. The caption a user got was their own
  prefix with nothing from the image. Affixes now leave an empty caption
  empty, guarded once for both backends and both caption methods.
- **An unclosed reasoning block still reached sidecars.** It was returned
  unchanged on the reasoning that a visible monologue beats an empty caption
  box — true only while a human is looking at the box. A batch run with
  auto-save writes every non-empty result unwatched, so the trace landed in
  the dataset. A response that is entirely reasoning, closed or not, now
  yields no caption.
- **Sampled frames were held at native resolution for the whole inference.**
  Both backends downscaled to 640 px only *after* `sample_frames()` returned,
  but the sampler builds the entire list first — so the peak was already
  paid: 16 frames of 4K RGB is 400 MB, of 8K is 1.6 GB, alive until the
  caption finished. `sample_frames` now takes `max_dim` and clamps inside
  `_to_pil`, the one choke point both decode paths share, bringing the same
  16 frames to 11 MB. `first_frame` passes no `max_dim`, so thumbnails still
  get the real image.
- **The context preflight costed every model with Qwen3-VL's tiling.** Each
  family's real cost now comes from its own published config: Qwen3-VL and
  Qwen3.5 tile at 32 px (`patch_size` 16 x `merge_size` 2), Qwen2.5-VL at
  28 px (14 x 2), while Gemma-4 and Gemma-3 resample every image to a fixed
  280 and 256 soft tokens and do not vary with resolution at all. The old
  single formula over-counted a 640x640 Gemma-4 frame by 43% (400 vs 280),
  refusing 16-frame clips that fit in 8192 with room to spare — and
  *under*-counted Qwen2.5-VL by 32% (400 vs 529), the direction where the
  preflight passes and the caption then fails inside llama.cpp. An
  unrecognised family falls back to the most expensive tiling, so a new
  model errs toward refusing early rather than toward a failed caption.
- **`n_ctx=0` made every video refusable.** llama.cpp reads 0 as "use the
  model's native context", but the engine recorded the raw argument — so the
  preflight compared a positive budget against a zero-token window and
  refused every clip on a model with plenty of room. It now records
  `Llama.n_ctx()`, and an unknown window skips the check rather than always
  failing it.
- **The budget preflight ran after every frame was encoded.** Its whole point
  is to fail cheaply, but a clip it was about to reject had already paid for
  a JPEG encode and base64 per frame. Frames are now clamped and measured
  first, and encoded only once the request fits.
- **A missing Gemma handler fell back to a Qwen one.** The handler supplies
  the chat template and image-token protocol; Gemma's uses
  `<start_of_turn>`/`<end_of_turn>` where Qwen's uses `<|im_start|>`. That
  substitution is not a degraded mode, it is a wrong one, and it would have
  surfaced as nonsense output or an opaque media-evaluation failure. Qwen
  families still fall back along their shared ChatML lineage; a Gemma family
  now raises an error naming the missing handler. (Unreachable on the pinned
  wheel, which ships all of them.)
- **The opaque-constructor retry chain treated any `TypeError` as an
  unsupported keyword.** A handler rejecting the flag *combination* in its
  own body would be retried until some narrower set got past the raise,
  silently accepting a construction it meant to refuse. Only a
  "unexpected keyword argument" error now drops a flag.
- **Asking for every frame of a short clip silently dropped most of them.**
  `_midpoint_indices` rounded `(i+0.5)*total/n`, and Python's `round()` breaks
  ties to even — so whenever `total == num_frames` the midpoints 0.5, 1.5,
  2.5 ... collapsed in pairs and dedup discarded them for good. An 8-frame
  clip asked for 8 frames returned 5 (`[0, 2, 4, 6, 7]`); a 3-frame clip asked
  for 3 returned 2. Truncating instead has no ties and is the more correct
  reading anyway — a frame index is a bin, so the frame *containing* the
  midpoint is the one wanted. Now exactly `min(total, num_frames)` distinct
  indices for every input, with the mid-span placement unchanged.
- **`cancel_check` alone was ignored once GGUF generation started.** The
  shared `_generate` streamed only when a `stream_callback` was supplied; the
  blocking branch had no point at which to poll, so a caller that passed
  `cancel_check` without wanting tokens ran to completion and was handed the
  caption it had asked to abandon. It now streams whenever either is present,
  invoking the callback only when there is one — matching the MLX backend,
  which cancels regardless of streaming.
- **Cancelling was ignored until generation started.** Frame extraction can
  run for seconds before any token loop exists to notice a cancel: a clip
  whose header reports no frame count is scanned twice end to end, and both
  backends then encode or stage every sampled frame. `sample_frames` now
  takes an optional `cancel_check`, polled every `CANCEL_POLL_INTERVAL`
  frames in both scans and between frames while encoding; it raises
  `VideoCancelled`, which each engine turns into the same `""` a cancel
  during generation produces.

### Added (engine only — no GUI entry point)
- **Video captioning in the engine.** `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`
  and `.m4v` are captioned by sampling evenly-spaced frames and sending them
  as one multi-image chat turn — the only mechanism `mtmd` actually supports
  today (the Qwen3-VL chat template still carries "# Video not supported
  yet"). Implemented once in `engine/video.py` and driven identically by the
  GGUF and MLX engines; frame count (8 by default, 16 max) and frame size
  (640 px longest side) are shared constants so the backends cannot drift
  apart. Callable from Python only until the GUI lands.
- **Two new model families in the download list**, with their own chat
  templates selected at load time: HauhauCS Qwen3.5 9B Uncensored
  (`qwen35`) and HauhauCS Gemma-4 E4B Uncensored (`gemma4`). The chat family
  is inferred from the filename and can be overridden explicitly.
- **Up-front context-budget preflight for video.** Qwen3-VL's M-RoPE
  disables llama.cpp's context shift, so an `n_ctx` overflow cannot be
  recovered from mid-generation. It is not a crash, though: measured without
  the preflight on an RTX 4080, llama.cpp logs `decode: failed to find a
  memory slot for batch of size 300` and the pinned wheel raises a catchable
  `ValueError` ("Media evaluation failed with error code 1") after 2.56 s —
  the process survives, but the user has paid for a doomed encode and the
  message points at nothing actionable. The frame count, the measured prompt
  and the generation budget are now checked against `n_ctx` before anything
  is encoded: the check fails in 0.05 s, does no GPU work, and names the
  argument to lower (`num_frames`).

### Fixed
- **Reasoning traces could become the caption.** The pinned wheel builds
  Qwen3.5 and Gemma-4 handlers with `enable_thinking=True`, which ends the
  generation prompt inside an open `<think>` block — the model's monologue
  then *is* the caption, usually exhausting the token budget before any
  description appears. Thinking is now disabled at construction, and a
  leading `<think>`/thought-channel block is stripped from the result if one
  appears anyway. An unclosed block is left intact rather than silently
  saving an empty sidecar.
- **Six of the eleven Gemma-4 quants downloaded without a vision encoder.**
  The `P` of a "pure" quant (`Q6_K_P`) was not recognised as part of the
  quant tag, so those builds' identity key ended in a stray `p` and never
  matched the `mmproj` published beside them. Gemma's `E4B`
  effective-parameter size is now read as a size too, and kept distinct from
  a dense 4B.
- **Downloading a second model family skipped its vision encoder.** The
  "already have one?" check asked `find_mmproj_file(target_dir)` with no
  model, which returns whichever encoder sorts first — so a Gemma-4 download
  into a folder holding a Qwen3-VL encoder queued nothing. Both the queue and
  the post-download re-check now match the registry's exact `mmproj_filename`.
  (Pairing a model with another model's `mmproj` does not fail cleanly; it
  crashes llama.cpp natively.)
- **A frame count that *over*-reports silently shrank the sample.** The
  sampler trusted
  `CAP_PROP_FRAME_COUNT` and seeked to evenly-spaced indices from it. Seeking
  past the real end of a truncated file *succeeds*, so the read that follows
  simply failed and was skipped: on an AVI whose header claims 50 frames over
  32 that decode, `num_frames=16` returned 10 frames, 8 returned 5 and 4
  returned 3 — no error, and the surviving frames bunched into the part of
  the index space that still existed. The indexed pass now reports how many
  of the indices it was asked for actually decoded, and a **short** result
  (not just an empty one, which was the old trigger) falls back to the
  sequential pass, whose `grab()` count spans the real clip. Measured on that
  same file the fixed sampler returns the full 4, 8 and 16 frames. A
  genuinely short clip is unaffected: its index list is already clamped to
  its own length, so it is never "short" and pays for no second pass. Only
  the over-reporting direction is handled: a header that *under*-reports is
  still trusted, every index built from it decodes, nothing looks short, and
  the tail of the clip past the claimed count is never sampled. That gap is
  documented in `sample_frames` and left for later.
- **The sequential frame sampler under-delivered and skewed early.** For
  files whose header reports no frame count, or whose container cannot seek,
  the old rolling-halving pass kept whichever frames it happened to survive
  with — fewer than requested, and weighted toward the intro. It now counts
  frames with a decode-free `grab()` pass and then decodes only the midpoint
  indices, giving the same coverage as the seeking path. The unseekable
  fallback had no test at all and now has one, and it gets its own warning
  text: a container that refuses to seek usually has a perfectly truthful
  header, and the shared message used to accuse that header of lying.

### Changed
- **Single-image prompts change — deliberately.** Rendering the pinned
  wheel's own jinja templates over the real message list shows the old path
  (`Qwen25VLChatHandler`) emitting
  `<|im_start|>user\nPicture 1: <|vision_start|> <img> <|vision_end|>PROMPT…`
  — a hardcoded `Picture 1: ` prefix and a space either side of the
  placeholder, with no flag to turn either off. The new `Qwen3VLChatHandler`
  with `add_vision_id=False` emits
  `<|im_start|>user\n<|vision_start|><img><|vision_end|>PROMPT…` and the
  trailing newline after the assistant turn that the Qwen3-VL template
  specifies. So this is a prompt change for every existing single-image user,
  and a correctness fix rather than a no-op: an A/B over the same 7 images
  scored 11/11 on OCR and 9/9 on chart reading on both sides, with no factual
  regressions. `add_vision_id=False` is what keeps the *new* handlers
  (`Qwen3VLChatHandler`, `Qwen35ChatHandler`, both defaulting to `True`) from
  re-introducing `Picture N:`; video states its frame ordering in the text
  prompt instead, which also works for Gemma-4, whose handler has no such
  flag. Unknown keywords are dropped on a `TypeError` retry so other
  llama-cpp-python builds still load.
- `opencv-python-headless` is capped below the next major (`>=4.9,<6`).
  OpenCV 5.0 shipped during development and an unbounded range would upgrade
  every fresh install into it unannounced; the suite passes on 5.x.
- Video frames are encoded as JPEG q95 rather than PNG, matching the
  single-image path — the chat handler re-encodes to JPEG regardless, so PNG
  only cost a slow compression pass and a much larger base64 payload, N times
  over per clip.

## [1.4.3] — 2026-07-30

Maintenance release from a full repository health check, followed by a second
deep QC audit of the entire codebase (56 findings, each adversarially
verified against the source — several reproduced empirically against live
PyQt6 before fixing), plus a rework of the Windows install diagnostics
driven by a live support case (issue #22).

### Fixed (install diagnostics — issue #22)
- **False "CPU build detected" warning on healthy CUDA installs.** The
  v0.3.40 wheels carry the `+cuNNN` CUDA tag only in the wheel filename —
  the installed metadata reports plain `0.3.40` — so `doctor.py` warned
  "CPU build" on every correct GPU install and the wheel/toolkit match
  check silently disabled itself. CUDA builds are now detected by the
  `ggml-cuda.dll` they ship, and the `+cuNNN` tag is recovered from the
  PEP 610 `direct_url.json` that pip/uv record, restoring the match check.
- **`WinError 127` engine-load failures are now diagnosed, not shrugged
  at.** `diagnose.bat` scans the DLL search locations that outrank the
  app's own folders (python dir, System32, Windows dir, CWD) plus PATH,
  and prints every conflicting llama.cpp-family DLL ranked by whether it
  can actually shadow the wheel's copy. Remediation always leads with the
  MSVC-runtime update and warns against deleting System32 files (rename
  reversibly instead). Confirmed live on issue #22: a stale
  `libomp140.x86_64.dll` in System32.
- `llama_cpp/bin` (new in the v0.3.40 wheel layout) is registered on the
  DLL search path alongside `lib`, and all diagnostic stat/scan calls
  degrade gracefully on unreadable directories instead of crashing app
  startup or the doctor.

### Added (release process)
- **Version-sync guard test**: `gui/version.py`, `pyproject.toml`, the
  README title/badge, and the newest CHANGELOG entry must all agree on the
  version — any drift fails CI on every PR.
- **Tag-verified automatic releases**: pushing a `V*` tag now runs a
  workflow that verifies the tag matches every in-repo version and only
  then creates the GitHub release with notes extracted from this file —
  a tag/file mismatch (the v1.4.2 ZIP shipped showing "1.4.1" in-app)
  can no longer become a release.

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
