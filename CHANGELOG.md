# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions correspond to
git tags (`V1.x.x`).

## [Unreleased] — video engine (targets 1.5.0)

**Not a release, and not a user-facing feature yet.** This section covers the
*engine* half of issue #26: video captioning exists in `engine/` and in the
test suite, but no GUI code calls it — `caption_video()`, `first_frame()` and
`is_video_file()` have no caller outside `engine/` and `tests/`. The app
still captions images only, `APP_VERSION` stays at the released **1.4.4**,
and the GUI wiring (video files in the picker, a "Frames per video" control,
video thumbnails, video-aware batch runs) is part 2 of #26.

Two fixes first written for this work shipped in 1.4.4 and are described
there, not repeated here: the empty-caption guard in `apply_prefix_suffix`
(which `caption_video()` passes through too) and the exact-name check that
queues each model's own vision encoder.

Validated by hand on an RTX 4080: frames arrive in temporal order, 16 frames
take 9.1 s at 9.4 GB peak VRAM, the context preflight refuses cleanly and
cancel leaves nothing running. The Qwen3-VL handler switch was A/B'd over 7
images (OCR, chart-reading and fine-detail cases) with no caption regression.

### Fixed (review findings on the engine work)
- **Model loading rejected flags forwarded through `**kwargs`.** Optional
  captioning flags now require an explicit keyword parameter in the resolved
  handler signature, including inherited constructors. Qwen3-VL and Gemma
  handlers no longer forward unsupported options to their strict base class.
- **Gemma thought channels survived caption cleanup.** Both complete and
  unclosed `<|channel>thought` blocks are recognized using Gemma's actual
  `<channel|>` closing token. Regression tests cover both image backends,
  streaming GGUF output, affixes, and protection of existing caption files.
- **A reasoning-only response came back as the caption.** `clean_caption`
  falls back to the original text when cleaning empties it, which protects a
  prefix-only caption like `"Caption:"`. A response that is *nothing but* a
  closed `<think>` block also cleans to empty — so the fallback handed the
  trace straight back, the one outcome `strip_reasoning` exists to prevent.
  The fallback now restores the post-reasoning text, so a reasoning-only
  response cleans to `""` (reported as "The model returned no caption" and
  never saved) while a prefix-only one is still preserved.
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
- **An unrecognised `chat_family` fell through to a Qwen handler.** A key
  missing from the handler map (a registry typo, or a value from a newer
  build) skipped the Gemma guard and reached the ChatML fallback, applying a
  Qwen template to whatever the file actually is. It now raises a
  `ValueError` naming the known families. Filename inference only ever
  produces a known key, so nothing that loaded before is refused.
- **The opaque-constructor retry chain treated any `TypeError` as an
  unsupported keyword.** A handler rejecting the flag *combination* in its
  own body would be retried until some narrower set got past the raise,
  silently accepting a construction it meant to refuse. Only an error that
  rejects a keyword now drops a flag, recognised in every spelling CPython
  uses for it. A Python callable says "unexpected keyword argument", but a C
  extension type says "invalid keyword argument" or "takes no keyword
  arguments", and extension types are the opaque constructors this chain
  exists for. Matching only the first spelling had left it dead there: the
  first attempt raised, the narrower flag sets were never tried, and a build
  that could have loaded the model did not.
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
- **A seek that reported success without moving was trusted.**
  `cap.set(CAP_PROP_POS_FRAMES, idx)` returning True does not prove the
  decoder moved; some backends and containers accept the seek and stay put.
  The reads that followed decoded consecutive frames from wherever the
  decoder sat, and because the count came out right neither the
  short-sample check nor the refused-seek check fired: the model was handed
  a cluster of near-identical frames described as spanning the whole clip.
  The position is now read back after every seek and must land within half
  the tightest gap between the sampled indices, with no fixed floor (one
  wide enough for a keyframe snap on a film would be wider than a short clip
  itself). A miss skips that index, so the sample comes up short and the
  sequential pass re-takes it; when no index is reachable at all, the
  container is treated as one that cannot seek.

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
  appears anyway. A response that is nothing but reasoning, closed or not,
  yields no caption and is never saved (see the review findings above).
- **Six of the eleven Gemma-4 quants downloaded without a vision encoder.**
  The `P` of a "pure" quant (`Q6_K_P`) was not recognised as part of the
  quant tag, so those builds' identity key ended in a stray `p` and never
  matched the `mmproj` published beside them. Gemma's `E4B`
  effective-parameter size is now read as a size too, and kept distinct from
  a dense 4B.
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

## [1.4.4] — 2026-09-29

The first release since 1.4.3, and it carries considerably more than a patch
number suggests. The full-repository audit fixes that landed on `main` after
1.4.3 was tagged (#34, which fixed most of the 66 verified findings in #33;
the ones still open on `main` are listed under Known gaps) were never given
a changelog entry; they ship here, documented, alongside two defects found
while developing the next feature release and raised dependency floors that
clear 13 Pillow advisories. None of it is in the 1.4.3 release build: users
of that build get all of it at once, because `main` has been carrying it
untagged.

### Fixed — a fake caption and a skipped encoder download

Both found while developing the video engine. The first put wrong data in a
training set without raising an error, which is why it did not show up in
normal use. The second was a missed download: Load Model had to ask for an
encoder that should have arrived with the model.

- **Prefix/suffix manufactured a caption out of a failed generation.**
  `apply_prefix_suffix("")` returned the affixes joined around nothing —
  `"photo of  high quality"` — which is truthy, so `_auto_save_caption` wrote
  it to the `.txt` sidecar, cached it as saved and marked the image done. A
  single image shows it in the caption box, but a batch run with auto-save is
  unattended, so every image the model returned nothing for received the same
  stock string and was counted in the "saved" total. An empty caption now
  stays empty at the single point both engines funnel through, so the
  existing falsy-caption guard refuses it, the batch summary counts it as
  failed, and the notification bell says the model returned nothing for that
  image. In 1.4.3 an empty result with no prefix or suffix was also cached as
  a caption, and Export then wrote it out as an empty `.txt`, over a good one
  too. Now nothing is cached for it: the image keeps the caption and badge it
  had, Export writes nothing for it, and closing the app does not warn about
  it.
- **Downloading a second model skipped its vision encoder.** Both gates
  that decide whether to queue an `mmproj` asked
  `find_mmproj_file(target_dir)` with no model, which accepts whichever
  encoder is already in the folder. The registry ships four GGUF models,
  each with its own encoder file, so a user who downloaded a second one into
  the same folder did not get its encoder, and Load Model had to stop and
  ask for it. Both gates now match the registry's exact `mmproj_filename`;
  an encoder genuinely already present is still skipped.

### Fixed — caption safety (#34)

- Dirty-state tracking with Save/Discard/Cancel on every overwrite path, so a
  hand-edited caption is no longer destroyed by changing the selection,
  regenerating, or Clear All.
- Streamed tokens are pinned to the image that requested them, so selecting a
  different image mid-generation can no longer write one image's caption into
  another's sidecar.
- Atomic sidecar writes, with a retry for the Windows sharing lock.
- Unreadable sidecars are treated as captioned by batch, export and delete
  rather than silently re-captioned and overwritten.
- Clearing a caption with the trash button and then saving now deletes its
  `.txt` sidecar; in 1.4.3 an emptied caption could not be saved, so the old
  file stayed.
- The caption editor is read-only while generating.
- The caption cache and thumbnails follow external sidecar changes instead of
  serving a stale first read.

### Fixed — engine and model pairing (#34)

- Shared image preprocessing across backends: the MLX engine now gets the
  same EXIF-corrected image, clamped to 1280 px, as the GGUF engine (it was
  handed the original file at full resolution), JPEG sources are downscaled
  while decoding, and the GGUF engine sends JPEG q95 instead of PNG.
- Model-aware, family-aware `mmproj` pairing that refuses a foreign encoder
  instead of guessing — the mismatch that previously crashed natively on the
  first caption.

### Fixed — setup and install (#34)

- `setup.bat` no longer aborts with "Failed to install uv" immediately after
  installing uv successfully.
- The venv is re-creatable, so "re-run setup" — the remedy the app, the
  doctor and the README all prescribe — actually works.
- uv installs correctly when launched from PowerShell 7.
- Verified on fresh `windows-latest` runners.

### Fixed — interface (#34)

- Light theme contrast, including controls that previously rendered
  white-on-white, and a runtime theme switch that now repaints rather than
  freezing colours from whichever palette was loaded at import.
- Batch busy state no longer lets a finishing caption re-enable a button
  mid-download.
- Thumbnails decode asynchronously.
- Zoom goes through a `QGraphicsView` transform instead of re-scaling the
  full-resolution pixmap on every wheel notch.
- Windows download pre-allocation no longer physically writes gigabytes of
  zeros before the first byte arrives.

### Fixed — batch, import and diagnostics (#34)

- Batch Caption All asks before overwriting existing captions (Skip
  already-captioned / Overwrite all / Cancel, defaulting to Skip), and Export
  writes only captions with no file yet unless you choose to overwrite ones
  that differ.
- A failed caption write is reported and counted; the green check now appears
  only once the caption is on disk.
- The Caption Length setting works for the Stable Diffusion and Pony tag
  presets (all three lengths used to ask for 15-30 tags).
- Folder import skips macOS `._` files, a folder that cannot be read no longer
  aborts the app, and dropping an image from a browser tab no longer imports
  every image in the working directory.
- Cancel no longer stops a running model download without asking.
- Qt's image allocation limit is restored (2048 MB) instead of removed.
- CUDA detection requires `cudart64_*.dll`, so a leftover version folder no
  longer wins; a missing `nvidia-ml-py` is no longer reported as a missing
  driver; the Intel-Mac build is pinned to the release tag.

### Changed — dependencies

Re-run `setup.bat` / `./setup.sh` to pick these up: setup recreates the venv
and installs from `requirements.txt`, so an existing install keeps its old
Pillow until then. Manual (Linux) installs: run `pip install -r
requirements.txt` again inside the venv.

- **Pillow floor raised from 12.2.0 to 12.3.0.** 12.2.0 is affected by 13
  advisories (10 HIGH, 3 MODERATE, each with its own CVE), all fixed in
  12.3.0 and none known to affect it: decompression-bomb checks skipped when
  loading BDF, PCF and GD files; heap out-of-bounds writes in
  `Image.paste()`/`Image.crop()`, `ImageFilter.RankFilter` and
  `ImageCmsTransform.apply()`; an out-of-bounds read on McIdas AREA files;
  denial of service through PDF streams, EPS files and tiled JPEG 2000; heap
  data copied into TGA RLE output; and command injection in the Windows
  image-viewer helper. 12.3.0 no longer publishes manylinux2014 wheels (its
  oldest Linux wheels need glibc 2.27); that does not narrow the experimental
  Linux path, which already needed glibc 2.28 for PyQt6's Qt wheels.
- **The install doctor checks Pillow.** `diagnose.bat` (or
  `.venv/bin/python doctor.py`) reports a Pillow older than 12.3.0 as a
  problem, with the command that updates it. Unzipping a new version over an
  old folder keeps the old venv, and the app starts normally on the old
  Pillow, so nothing else would say so.
- **`huggingface-hub` is now `>=0.32,<3`.** 2.0.0 is the current release;
  the app's three hub calls (`hf_hub_download`, `HfApi.list_repo_files`,
  `hf_hub_url`) were checked on both 1.33.0 and 2.0.0, and the range is
  capped below the next major until it can be checked too. On Apple Silicon
  the MLX backend's `transformers` currently requires `huggingface-hub<2.0`,
  so Macs stay on 1.x; both work.
- **Build requirement raised from `setuptools>=75.0` to `>=84.0.0`**,
  clearing CVE-2025-47273 (HIGH, path traversal in
  `PackageIndex.download`, fixed in 78.1.1) and CVE-2026-59890 (MODERATE,
  sdist `MANIFEST.in` exclusion bypass, fixed in 83.0.0). This affects
  `pip install .` and wheel builds only; setup does not build the package.
- The Linux manual-install command in the README pins the llama-cpp-python
  fork to commit `12861b91` (tag `v0.3.40-Metal-macos-20260607`, the source
  `setup.sh` builds) instead of the fork's default branch.

### Changed — CI and supply chain (#28, #34, #36, #37, #38)

- Ruff replaces pyflakes; PR checks, GitGuardian secret scanning and
  Dependabot added (#28).
- The two unvetted third-party pull-request actions added in #28 were
  removed; "Require test plan" is now an inline step that runs with no
  token (#37).
- Third-party actions that receive secrets are pinned to commit SHAs, so a
  moved tag cannot change what runs (#38). The GitGuardian scan is now the
  only third-party action left.
- Dependabot pull requests are exempt from the "Require test plan" check (#36).
- The unused TestingBot Maestro workflow, a mobile-UI test scaffold with no
  test flows in this repository, was removed (#36).
- The version-sync guard added in 1.4.3 now also fails when the README's
  test count differs from the suite, and the release job reads
  `gui/version.py` with the same pattern as the guard, so a quoting change
  can no longer pass CI and then fail the release (#34).

### Changed — release and CI (this release)

- The Linux test job installs `requirements.txt` instead of repeating its
  pins, so CI resolves the same ranges users install from: the newest version
  each range allows, which is what a fresh setup gets.
- Every release now carries `QWEN3-VL-Captioner.zip`, built from the tagged
  commit, under a name that never changes, so
  `releases/latest/download/QWEN3-VL-Captioner.zip` always serves the newest
  published release. The release job refuses to publish if the ZIP lacks
  the setup or run scripts or `app.py`, and checks afterwards that the link
  resolves to the new release. The README's download button still links to
  `main` in this release.

### Tests

- 371 collected, up from 143 at 1.4.3. CI runs the suite on Linux; it also
  passed natively on Windows when #34 was validated (311 passed, 2 POSIX-only
  skips).
- Both defects in the first section were reproduced before being changed.
  Ten of the thirteen tests added for them fail against the pre-fix code; the
  other three check the opposite direction (an encoder already on disk is not
  downloaded again, a real caption is left alone) and pass either way.
- Caption-workflow tests check that an empty result names its image in the
  notification bell and leaves the image's caption, badge and sidecar as
  they were, including through Export and the unsaved-captions check on
  close. `tests/test_doctor.py` (new) checks the doctor's Pillow warning and
  that its floor matches `requirements.txt` and `pyproject.toml`.
- `tests/test_mmproj_queue.py` (new) drives the real `MainWindow` offscreen
  with the download stubbed, covering both encoder-queue gates in both
  directions. The two models are selected out of `MODEL_REGISTRY` rather
  than hardcoded, so renaming a model cannot turn the test into a no-op.

### Known gaps

- Two performance findings from #33 remain open: clearing or widening the
  list filter and Clear All are still O(n²) on large imports, and the file
  browser still builds a widget tree per image rather than virtualizing.
  Both are visible only on datasets of a few thousand images.
- On mlx-vlm 0.3.4 through 0.3.11 the MLX engine ignores the Temperature
  setting and decodes greedily (always the most likely token). Those
  versions accept the `sampler` argument the app passes to `stream_generate`
  without error and never use it, and the fallback to the older
  `temperature`/`top_p` arguments waits for a `TypeError` that no mlx-vlm
  release since 0.1.0 raises. mlx-vlm 0.3.12 and later honour the setting.
  `setup.sh` installs mlx-vlm unpinned; on macOS 13 the newest release that
  installs is 0.3.9, because mlx 0.29.4 and later publish no macOS 13 wheels.

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
