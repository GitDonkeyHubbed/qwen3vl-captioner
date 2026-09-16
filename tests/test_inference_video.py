"""Tests for the GGUF engine's video captioning path (engine.inference).

``caption_video`` sends N temporally ordered frames as one multi-image chat
turn. No real model, video file, or cv2 is needed: a fake
``create_chat_completion`` captures its kwargs and returns canned
llama-cpp-shaped responses, and ``engine.video.sample_frames`` is replaced by
a stub yielding solid-color PIL frames (the color encodes temporal order).
"""

import base64
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import engine
from engine import inference
from engine.base import DEFAULT_SYSTEM_PROMPT, VideoCancelled
from engine.inference import MAX_VIDEO_FRAMES, Qwen3VLEngine, infer_chat_family


# Distinct per-frame colors so the encoded data URIs reveal temporal order.
_FRAME_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
]

_CANNED_RESPONSE = {"choices": [{"message": {"content": "a cat walks by"}}]}

_STREAM_CHUNKS = [
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"content": "A dog"}}]},
    {"choices": [{"delta": {"content": " runs."}}]},
    {"choices": [{"delta": {}}]},
]


class FakeModel:
    """Stands in for llama_cpp.Llama: records kwargs, returns canned output."""

    def __init__(self, response=None, stream_chunks=None):
        self.response = response
        self.stream_chunks = stream_chunks or []
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.stream_chunks)
        return self.response


def _install_fake_video(monkeypatch, frame_size=(64, 48)):
    """Inject a stub engine.video module; returns a dict recording the call."""
    seen = {}

    def sample_frames(video_path, num_frames=8, cancel_check=None, max_dim=None):
        seen["video_path"] = video_path
        seen["num_frames"] = num_frames
        seen["cancel_check"] = cancel_check
        if cancel_check and cancel_check():
            # Mirrors the real extractor, which raises rather than returning
            # a short sample when cancelled mid-scan.
            raise VideoCancelled("cancelled during video frame extraction")
        return [
            Image.new("RGB", frame_size, _FRAME_COLORS[i % len(_FRAME_COLORS)])
            for i in range(num_frames)
        ]

    mod = types.ModuleType("engine.video")
    mod.sample_frames = sample_frames
    monkeypatch.setitem(sys.modules, "engine.video", mod)
    monkeypatch.setattr(engine, "video", mod, raising=False)
    return seen


def _make_engine(response=None, stream_chunks=None, n_ctx=8192):
    eng = Qwen3VLEngine()
    eng._is_loaded = True
    eng._n_ctx = n_ctx
    eng.model = FakeModel(response=response, stream_chunks=stream_chunks)
    return eng


def _video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")  # existence is all caption_video checks
    return path


def _decode_data_uri(uri: str) -> Image.Image:
    # JPEG q95, matching the single-image path: the chat handler re-encodes
    # whatever it receives to JPEG anyway, and N frames per turn make the
    # payload size difference N times worse than it is for one image.
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


def _decoded_frame_index(uri: str) -> int:
    """Recover which _FRAME_COLORS entry a decoded frame came from.

    JPEG is lossy, so the corner pixel is compared by nearest colour rather
    than equality — the frames are solid and far apart in RGB, so the nearest
    match is unambiguous.
    """
    got = _decode_data_uri(uri).convert("RGB").getpixel((0, 0))
    distances = [
        sum((a - b) ** 2 for a, b in zip(got, color, strict=True))
        for color in _FRAME_COLORS
    ]
    return distances.index(min(distances))


def test_message_structure_and_temporal_order(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    caption = eng.caption_video(_video_file(tmp_path), "Describe the video.", num_frames=4)

    assert caption == "a cat walks by"
    assert len(eng.model.calls) == 1
    call = eng.model.calls[0]
    assert call["stream"] is False
    assert call["temperature"] == 0.6
    assert call["top_p"] == 0.9
    assert call["max_tokens"] == 1024

    messages = call["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT

    parts = messages[1]["content"]
    assert [p["type"] for p in parts] == ["image_url"] * 4 + ["text"]
    # The frames are framed as one clip in the text, because add_vision_id
    # ("Picture N:") is switched off and Gemma-4's handler has no such flag
    # at all. The user's prompt is appended verbatim.
    assert parts[-1]["text"] == (
        "The 4 images above are frames sampled in order from a single video "
        "clip. Describe the video."
    )

    # The image parts must preserve temporal order (frame i has color i).
    assert [_decoded_frame_index(p["image_url"]["url"]) for p in parts[:4]] == [
        0, 1, 2, 3
    ]


def test_frames_resized_to_video_max_dim(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch, frame_size=(1280, 720))
    eng = _make_engine(response=_CANNED_RESPONSE)

    eng.caption_video(_video_file(tmp_path), "p", num_frames=2)

    parts = eng.model.calls[0]["messages"][1]["content"]
    img = _decode_data_uri(parts[0]["image_url"]["url"])
    assert img.size == (640, 360)


@pytest.mark.parametrize("requested,effective", [(1, 2), (99, MAX_VIDEO_FRAMES)])
def test_num_frames_clamped(monkeypatch, tmp_path, requested, effective):
    seen = _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    eng.caption_video(_video_file(tmp_path), "p", num_frames=requested)

    assert seen["num_frames"] == effective
    parts = eng.model.calls[0]["messages"][1]["content"]
    assert len(parts) == effective + 1  # N frames plus the text part


def test_context_budget_overflow_raises(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=512)

    with pytest.raises(RuntimeError) as excinfo:
        eng.caption_video(_video_file(tmp_path), "p")

    assert "frames" in str(excinfo.value).lower()
    assert "512" in str(excinfo.value)
    assert eng.model.calls == []  # refused before touching the model


def test_context_budget_counts_vision_tokens(monkeypatch, tmp_path):
    # 8 frames at 640x360 (already at VIDEO_FRAME_MAX_DIM, so no resize):
    # estimate_vision_tokens gives 20*12 = 240 each -> 1920 total, and
    # ceil(1.15 * 1920) = 2208. With max_tokens=1024, ~41 fallback prompt
    # tokens (the fake model has no tokenize) and the 128-token scaffold,
    # needed ~= 3401 > 2500 -> refuse. Were the vision tokens dropped from
    # the sum, the same call would fit (~1193 < 2500) and reach the model.
    _install_fake_video(monkeypatch, frame_size=(640, 360))
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=2500)

    with pytest.raises(RuntimeError) as excinfo:
        eng.caption_video(_video_file(tmp_path), "p", num_frames=8)

    assert "8 video frames" in str(excinfo.value)
    assert "2500" in str(excinfo.value)
    # The remedy names the engine argument, not a GUI control: this PR ships
    # the engine only, so there is no "frames per video" widget to point at.
    assert "num_frames" in str(excinfo.value)
    assert eng.model.calls == []  # refused before touching the model


def test_context_budget_allows_vision_tokens_that_fit(monkeypatch, tmp_path):
    # Same 8x 640x360 frames (~3401 tokens needed) with room to spare.
    _install_fake_video(monkeypatch, frame_size=(640, 360))
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=4096)

    caption = eng.caption_video(_video_file(tmp_path), "p", num_frames=8)

    assert caption == "a cat walks by"
    assert len(eng.model.calls) == 1


def test_context_budget_counts_prompt_tokens(monkeypatch, tmp_path):
    # 2 tiny 64x64 frames are ~10 vision tokens after overhead — negligible.
    # A ~5800-char prompt hits the fallback estimate (len//3 + 16 ~= 1974),
    # so 10 + 512 + ~2015 + 128 overflows n_ctx=2048. A flat text allowance
    # instead of measuring the prompt would let this call through.
    _install_fake_video(monkeypatch, frame_size=(64, 64))
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=2048)
    long_prompt = "describe the scene in detail " * 200

    with pytest.raises(RuntimeError) as excinfo:
        eng.caption_video(
            _video_file(tmp_path), long_prompt, max_tokens=512, num_frames=2
        )

    assert "2048" in str(excinfo.value)
    assert eng.model.calls == []

    # The identical call with a short prompt fits comfortably.
    caption = eng.caption_video(
        _video_file(tmp_path), "p", max_tokens=512, num_frames=2
    )
    assert caption == "a cat walks by"
    assert len(eng.model.calls) == 1


def test_streaming_concatenates_tokens(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch)
    eng = _make_engine(stream_chunks=_STREAM_CHUNKS)
    received = []

    caption = eng.caption_video(
        _video_file(tmp_path), "p", stream_callback=received.append
    )

    assert eng.model.calls[0]["stream"] is True
    assert received == ["A dog", " runs."]
    assert caption == "A dog runs."


def test_streaming_honors_cancel_check(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch)
    eng = _make_engine(stream_chunks=_STREAM_CHUNKS)
    received = []

    caption = eng.caption_video(
        _video_file(tmp_path), "p",
        stream_callback=received.append,
        cancel_check=lambda: len(received) >= 1,
    )

    assert received == ["A dog"]
    assert caption == "A dog"


def test_prefix_suffix_applied(monkeypatch, tmp_path):
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    caption = eng.caption_video(
        _video_file(tmp_path), "p", prefix="anime,", suffix="masterpiece"
    )

    assert caption == "anime, a cat walks by masterpiece"


def test_not_loaded_raises(tmp_path):
    eng = Qwen3VLEngine()
    with pytest.raises(RuntimeError, match="not loaded"):
        eng.caption_video(tmp_path / "clip.mp4", "p")


def test_missing_file_raises(tmp_path):
    eng = _make_engine(response=_CANNED_RESPONSE)
    with pytest.raises(FileNotFoundError):
        eng.caption_video(tmp_path / "missing.mp4", "p")


@pytest.mark.parametrize("filename,family", [
    ("Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf", "qwen3vl"),
    ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", "qwen35"),
    ("Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", "gemma4"),
    ("qwen3_5-instruct.Q8_0.gguf", "qwen35"),
    ("Qwen3.6-4B.Q4_K_M.gguf", "qwen35"),
    ("qwen3_6-preview.gguf", "qwen35"),
    ("qwen35-merge.gguf", "qwen35"),
    ("gemma-3-27b-it.Q4_K_M.gguf", "gemma3"),
    ("gemma3-4b.gguf", "gemma3"),
    ("gemma4-experimental.gguf", "gemma4"),
    # Legacy Qwen2-VL / Qwen2.5-VL files a user may browse to keep the
    # Qwen2.5-VL handler instead of falling through to the Qwen3-VL template.
    ("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "qwen25vl"),
    ("qwen2_5-vl-3b.gguf", "qwen25vl"),
    ("Qwen2-VL-7B-Instruct-Q4_K_M.gguf", "qwen25vl"),
    ("qwen2vl-2b-instruct.Q8_0.gguf", "qwen25vl"),
    ("qwen2_vl-7b.gguf", "qwen25vl"),
    # The plain-Qwen2 tags must not swallow the newer lines.
    ("Qwen3-VL-2B-Instruct.gguf", "qwen3vl"),
    ("Qwen3.5-VL-4B.gguf", "qwen35"),
    ("SomeOther-Model.Q4_K_M.gguf", "qwen3vl"),
])
def test_infer_chat_family(filename, family):
    assert infer_chat_family(Path("/models") / filename) == family


class _FamilyHandler:
    pass


class _Qwen3VLHandler:
    pass


class _Qwen25Handler:
    pass


def test_handler_resolution_prefers_family_handler(monkeypatch):
    ns = types.SimpleNamespace(
        Gemma4ChatHandler=_FamilyHandler,
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
    )
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    assert inference._resolve_chat_handler_cls("gemma4") is _FamilyHandler


def test_handler_resolution_falls_back_in_order(monkeypatch):
    # Family handler missing -> Qwen3VLChatHandler.
    ns = types.SimpleNamespace(
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
    )
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    assert inference._resolve_chat_handler_cls("qwen35") is _Qwen3VLHandler

    # Only the legacy handler present -> Qwen25VLChatHandler.
    ns = types.SimpleNamespace(Qwen25VLChatHandler=_Qwen25Handler)
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    assert inference._resolve_chat_handler_cls("qwen3vl") is _Qwen25Handler


class _RecordingLlama:
    """Stands in for llama_cpp.Llama during load_model tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_llama_cpp(monkeypatch, instantiated):
    """Fake out llama-cpp-python so load_model runs without the real wheel.

    Each fake handler class records ``(class name, clip_model_path)`` into
    ``instantiated`` on construction, so tests can assert which family's
    handler load_model actually built.
    """

    class _RecordingHandler:
        def __init__(self, clip_model_path, verbose=False):
            instantiated.append((type(self).__name__, clip_model_path))

    class Gemma4ChatHandler(_RecordingHandler):
        pass

    class Qwen3VLChatHandler(_RecordingHandler):
        pass

    class Qwen25VLChatHandler(_RecordingHandler):
        pass

    ns = types.SimpleNamespace(
        Gemma4ChatHandler=Gemma4ChatHandler,
        Qwen3VLChatHandler=Qwen3VLChatHandler,
        Qwen25VLChatHandler=Qwen25VLChatHandler,
    )
    monkeypatch.setattr(inference, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(inference, "Llama", _RecordingLlama, raising=False)
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    monkeypatch.setattr(
        inference, "Qwen25VLChatHandler", Qwen25VLChatHandler, raising=False
    )


def _model_files(tmp_path):
    # Filename-inference on this model name yields 'qwen3vl'.
    model = tmp_path / "Qwen3-VL-8B-Instruct.Q4_K_M.gguf"
    mmproj = tmp_path / "mmproj-F16.gguf"
    model.write_bytes(b"\x00")
    mmproj.write_bytes(b"\x00")
    return model, mmproj


def test_load_model_explicit_chat_family_wins_over_filename(monkeypatch, tmp_path):
    instantiated = []
    _install_fake_llama_cpp(monkeypatch, instantiated)
    model, mmproj = _model_files(tmp_path)

    eng = Qwen3VLEngine()
    eng.load_model(model, mmproj, chat_family="gemma4")

    # The explicit family must be used, not the filename-inferred 'qwen3vl'.
    assert eng.chat_family == "gemma4"
    assert instantiated == [("Gemma4ChatHandler", str(mmproj))]
    assert eng.model.kwargs["model_path"] == str(model)


def test_load_model_infers_chat_family_when_omitted(monkeypatch, tmp_path):
    instantiated = []
    _install_fake_llama_cpp(monkeypatch, instantiated)
    model, mmproj = _model_files(tmp_path)

    eng = Qwen3VLEngine()
    eng.load_model(model, mmproj)

    assert eng.chat_family == "qwen3vl"
    assert instantiated == [("Qwen3VLChatHandler", str(mmproj))]


# ── Caption-safe chat-handler construction ───────────────────────────────
#
# The pinned wheel (JamePeng llama-cpp-python 0.3.40) defaults
# enable_thinking=True and add_vision_id=True. Both are wrong for captioning:
# thinking mode ends the generation prompt inside an open <think> block so the
# reasoning trace becomes the caption, and add_vision_id prefixes every image
# with "Picture N:". Handlers differ in which flags they accept, so the
# construction degrades by dropping unknown keywords.


class _FlagRecordingHandler:
    """Base for fakes that record the kwargs they were constructed with."""

    seen: dict

    def __init__(self, clip_model_path, verbose=False, **kwargs):
        type(self).seen = {"clip_model_path": clip_model_path, **kwargs}


def test_construct_chat_handler_disables_both_flags():
    class Handler(_FlagRecordingHandler):
        def __init__(self, clip_model_path, verbose=False,
                     enable_thinking=True, add_vision_id=True):
            super().__init__(
                clip_model_path, verbose,
                enable_thinking=enable_thinking, add_vision_id=add_vision_id,
            )

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert Handler.seen["enable_thinking"] is False
    assert Handler.seen["add_vision_id"] is False
    assert Handler.seen["clip_model_path"] == str(Path("/m/mmproj.gguf"))


def test_construct_chat_handler_drops_add_vision_id_when_unsupported():
    """Gemma-4's handler reasons but has no vision-id flag."""
    class Handler(_FlagRecordingHandler):
        def __init__(self, clip_model_path, verbose=False, enable_thinking=True):
            super().__init__(
                clip_model_path, verbose, enable_thinking=enable_thinking
            )

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert Handler.seen["enable_thinking"] is False
    assert "add_vision_id" not in Handler.seen


def test_construct_chat_handler_drops_enable_thinking_when_unsupported():
    """Qwen3-VL's handler labels images but does not reason."""
    class Handler(_FlagRecordingHandler):
        def __init__(self, clip_model_path, verbose=False, add_vision_id=True):
            super().__init__(
                clip_model_path, verbose, add_vision_id=add_vision_id
            )

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert Handler.seen["add_vision_id"] is False
    assert "enable_thinking" not in Handler.seen


def test_construct_chat_handler_falls_back_to_plain_construction():
    """An older wheel whose handler takes neither flag must still load."""
    class Handler(_FlagRecordingHandler):
        def __init__(self, clip_model_path, verbose=False):
            super().__init__(clip_model_path, verbose)

    handler = inference._construct_chat_handler(
        Handler, Path("/m/mmproj.gguf"), False
    )
    assert isinstance(handler, Handler)
    assert Handler.seen == {"clip_model_path": str(Path("/m/mmproj.gguf"))}


def test_construct_chat_handler_surfaces_a_real_typeerror():
    """A TypeError from the constructor *body* must not be swallowed.

    The retry chain exists to drop unknown keywords, not to hide a broken
    handler — a silently-None handler would crash later inside llama.cpp with
    no hint of where it came from.
    """
    class Handler:
        def __init__(self, clip_model_path, verbose=False, **kwargs):
            raise TypeError("mmproj is not a vision encoder")

    with pytest.raises(TypeError, match="not a vision encoder"):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)


def test_construct_chat_handler_runs_the_constructor_only_once_on_a_body_error():
    """A body TypeError must surface on the first attempt, not after retries.

    Filtering kwargs by signature (rather than retrying on TypeError) means a
    compatible constructor that raises internally is called exactly once, so
    any side effects before the raise are not repeated.
    """
    calls = []

    class Handler:
        def __init__(self, clip_model_path, verbose=False,
                     enable_thinking=True, add_vision_id=True):
            calls.append(1)
            raise TypeError("mmproj is not a vision encoder")

    with pytest.raises(TypeError, match="not a vision encoder"):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert calls == [1]


# ── Cancelling during extraction, before any token loop exists ───────────

def test_caption_video_returns_empty_when_cancelled_during_extraction(
    monkeypatch, tmp_path
):
    """A cancel mid-scan must look like a cancel mid-generation: "".

    Extraction of a header-less clip scans the file twice end to end. Before
    this, cancel_check was not consulted until the token loop, so both scans
    and every frame encode ran to completion after the user cancelled.
    """
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4, cancel_check=lambda: True
    )

    assert caption == ""
    # Cancelled before the model was ever called.
    assert eng.model.calls == []


def test_caption_video_passes_cancel_check_down_to_the_sampler(
    monkeypatch, tmp_path
):
    """The predicate must reach the sampler, not just be checked after it."""
    seen = _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)
    sentinel = lambda: False  # noqa: E731

    eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4, cancel_check=sentinel
    )

    assert seen["cancel_check"] is sentinel


def test_caption_video_stops_encoding_when_cancelled_between_frames(
    monkeypatch, tmp_path
):
    """Each frame costs a JPEG encode; a cancel must not pay for the rest."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)
    encoded = []
    real_encode = inference._encode_data_uri

    def counting_encode(img):
        encoded.append(1)
        return real_encode(img)

    monkeypatch.setattr(inference, "_encode_data_uri", counting_encode)
    # False for the sampler's own polling, then true once encoding starts.
    calls = []

    def cancel_after_first_frame():
        calls.append(1)
        return len(encoded) >= 1

    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=8,
        cancel_check=cancel_after_first_frame,
    )

    assert caption == ""
    assert len(encoded) == 1, "kept encoding frames after the cancel"
    assert eng.model.calls == []


def test_caption_video_uncancelled_run_returns_the_same_caption(
    monkeypatch, tmp_path
):
    """A live-but-false predicate must not change what the caller gets.

    It does change HOW: a cancel-only caller now takes the streaming
    completion, because a blocking one has no point at which to poll. The
    tokens are the same either way, so the caption must be identical.
    """
    _install_fake_video(monkeypatch)
    eng = _make_engine(stream_chunks=_STREAM_CHUNKS)

    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4, cancel_check=lambda: False
    )

    assert caption == "A dog runs."  # == the joined _STREAM_CHUNKS
    assert len(eng.model.calls) == 1
    assert eng.model.calls[0]["stream"] is True


def test_cancel_only_caller_gets_no_stream_callback_invocations(
    monkeypatch, tmp_path
):
    """Streaming for cancellation must not push tokens at a caller who
    never asked for them."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(stream_chunks=_STREAM_CHUNKS)

    # No stream_callback supplied; if the engine called one it would raise.
    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4, cancel_check=lambda: False
    )
    assert caption == "A dog runs."


def test_plain_call_still_uses_the_blocking_completion(monkeypatch, tmp_path):
    """No callback and no predicate: nothing to poll for, so don't stream."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    caption = eng.caption_video(_video_file(tmp_path), "p", num_frames=4)

    assert caption == "a cat walks by"
    assert eng.model.calls[0]["stream"] is False


def test_cancel_during_generation_without_a_stream_callback(
    monkeypatch, tmp_path
):
    """The gap this closed: cancel_check alone used to be ignored once
    generation began, so the call ran to completion and returned the very
    caption the caller had asked to abandon."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(stream_chunks=_STREAM_CHUNKS)

    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4,
        # False through extraction, true once generation starts.
        cancel_check=lambda: bool(eng.model.calls),
    )

    assert caption == ""
    assert eng.model.calls[0]["stream"] is True


# ── Handlers whose signature cannot be inspected at all ──────────────────
#
# An extension type with no __text_signature__ makes inspect.signature raise,
# so there is nothing to filter against. Simulated by making signature() raise
# ValueError, which is precisely what CPython does for such a type.

def _make_signature_opaque(monkeypatch):
    def raising_signature(obj, *a, **k):
        raise ValueError("no signature found for builtin type")

    monkeypatch.setattr(inference.inspect, "signature", raising_signature)


def test_opaque_handler_rejecting_both_flags_still_loads(monkeypatch):
    """Passing both flags blindly aborted the load; now it degrades."""
    _make_signature_opaque(monkeypatch)
    seen = {}

    class Handler:
        def __init__(self, clip_model_path, verbose=False):
            seen["built"] = clip_model_path

    handler = inference._construct_chat_handler(
        Handler, Path("/m/mmproj.gguf"), False
    )
    assert isinstance(handler, Handler)
    assert seen["built"] == str(Path("/m/mmproj.gguf"))


def test_opaque_handler_keeps_the_flags_it_does_accept(monkeypatch):
    """Degrading must not overshoot — a supported flag is still applied."""
    _make_signature_opaque(monkeypatch)
    seen = {}

    class Handler:
        def __init__(self, clip_model_path, verbose=False, enable_thinking=True):
            seen["enable_thinking"] = enable_thinking

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert seen["enable_thinking"] is False


def test_opaque_handler_body_typeerror_is_not_swallowed(monkeypatch):
    """With no flags left to drop, the error is the handler's own."""
    _make_signature_opaque(monkeypatch)

    class Handler:
        def __init__(self, clip_model_path, verbose=False, **kwargs):
            raise TypeError("mmproj is not a vision encoder")

    with pytest.raises(TypeError, match="not a vision encoder"):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)


def test_inspectable_handler_is_constructed_exactly_once(monkeypatch):
    """The guarantee the inspection path exists for, still held."""
    calls = []

    class Handler:
        def __init__(self, clip_model_path, verbose=False,
                     enable_thinking=True, add_vision_id=True):
            calls.append(1)
            raise TypeError("mmproj is not a vision encoder")

    with pytest.raises(TypeError):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert len(calls) == 1


# ── The recorded context window is llama.cpp's, not the caller's argument ──

def test_load_model_records_the_effective_context_not_the_argument(
    monkeypatch, tmp_path
):
    """n_ctx=0 means "use the model's native context", not "no context".

    Storing the raw 0 made caption_video compare a positive budget against a
    zero-token window, so every clip was refused on a model with plenty of
    room.
    """
    instantiated = []
    _install_fake_llama_cpp(monkeypatch, instantiated)

    class NativeCtxLlama(_RecordingLlama):
        def n_ctx(self):
            return 32768

    monkeypatch.setattr(inference, "Llama", NativeCtxLlama, raising=False)
    model, mmproj = _model_files(tmp_path)

    eng = Qwen3VLEngine()
    eng.load_model(model, mmproj, n_ctx=0)

    assert eng._n_ctx == 32768


def test_load_model_falls_back_when_n_ctx_is_unavailable(monkeypatch, tmp_path):
    """A build without the accessor keeps the argument."""
    instantiated = []
    _install_fake_llama_cpp(monkeypatch, instantiated)
    model, mmproj = _model_files(tmp_path)

    eng = Qwen3VLEngine()
    eng.load_model(model, mmproj, n_ctx=4096)
    assert eng._n_ctx == 4096


def test_preflight_is_skipped_when_the_window_size_is_unknown(
    monkeypatch, tmp_path
):
    """Unknown window (0) must degrade to not-checking, not always-failing."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=0)

    caption = eng.caption_video(_video_file(tmp_path), "p", num_frames=4)
    assert caption == "a cat walks by"


# ── The budget check must come before the frames are encoded ─────────────

def test_over_budget_clip_is_refused_before_any_frame_is_encoded(
    monkeypatch, tmp_path
):
    """The preflight exists to fail cheaply; encoding first defeats that.

    Every frame was JPEG-encoded and base64'd before the budget was checked,
    so a request the engine was about to reject still paid the full CPU cost
    of preparing it.
    """
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE, n_ctx=256)  # far too small

    encoded = []
    real_encode = inference._encode_data_uri
    monkeypatch.setattr(
        inference, "_encode_data_uri",
        lambda img: (encoded.append(1), real_encode(img))[1],
    )

    with pytest.raises(RuntimeError, match="context"):
        eng.caption_video(_video_file(tmp_path), "p", num_frames=8)

    assert encoded == [], "frames were encoded before the budget was checked"


def test_within_budget_clip_still_encodes_every_frame(monkeypatch, tmp_path):
    """Splitting the loop must not drop frames from the request."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    eng.caption_video(_video_file(tmp_path), "p", num_frames=6)

    parts = eng.model.calls[0]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url"] * 6 + ["text"]
    assert [_decoded_frame_index(p["image_url"]["url"]) for p in parts[:6]] == [
        0, 1, 2, 3, 4, 5
    ]


# ── A missing Gemma handler must not fall back to a Qwen template ────────

def test_missing_gemma_handler_raises_instead_of_using_a_qwen_one(monkeypatch):
    """Cross-template substitution is wrong, not degraded.

    Gemma's template uses <start_of_turn>/<end_of_turn> and its own image
    protocol; a Qwen handler would emit the wrong control tokens and fail
    somewhere with nothing pointing at the cause.
    """
    ns = types.SimpleNamespace(
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
    )
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)

    for family in ("gemma4", "gemma3"):
        with pytest.raises(RuntimeError, match="chat template requires"):
            inference._resolve_chat_handler_cls(family)


def test_present_gemma_handler_is_still_used(monkeypatch):
    ns = types.SimpleNamespace(
        Gemma4ChatHandler=_FamilyHandler,
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
    )
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    assert inference._resolve_chat_handler_cls("gemma4") is _FamilyHandler


def test_qwen_families_still_fall_back_along_the_chatml_lineage(monkeypatch):
    """Qwen handlers share <|im_start|> turns, so substitution is defensible."""
    ns = types.SimpleNamespace(
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
    )
    monkeypatch.setattr(inference, "llama_chat_format", ns, raising=False)
    assert inference._resolve_chat_handler_cls("qwen35") is _Qwen3VLHandler


def test_opaque_handler_body_typeerror_stops_the_retry_chain(monkeypatch):
    """A body TypeError must not be retried until some narrower set passes.

    A handler that rejects the flag COMBINATION in its own body would
    otherwise be called again with fewer flags, and one of those calls could
    get past the raise — silently accepting a construction the handler meant
    to refuse. Only "unexpected keyword argument" means "drop this flag".
    """
    _make_signature_opaque(monkeypatch)
    calls = []

    class Handler:
        def __init__(self, clip_model_path, verbose=False, **kwargs):
            calls.append(dict(kwargs))
            if len(kwargs) == 2:
                raise TypeError("both flags together are not supported here")

    with pytest.raises(TypeError, match="both flags together"):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert len(calls) == 1, f"retried past a body error: {calls}"


def test_opaque_handler_unsupported_keyword_is_still_retried(monkeypatch):
    """The retry that the chain exists for must still happen."""
    _make_signature_opaque(monkeypatch)
    seen = {}

    class Handler:
        def __init__(self, clip_model_path, verbose=False, enable_thinking=True):
            seen["enable_thinking"] = enable_thinking

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)
    assert seen == {"enable_thinking": False}


# ── Vision-token cost is per-family, from each model's published config ──

@pytest.mark.parametrize("family,expected", [
    # Qwen tilings: patch_size x merge_size from preprocessor_config.json.
    ("qwen3vl", 20 * 20),    # 16 x 2 = 32 px blocks
    ("qwen35", 20 * 20),     # 16 x 2 = 32 px blocks
    ("qwen25vl", 23 * 23),   # 14 x 2 = 28 px blocks -> ceil(640/28) = 23
    # Gemma resamples to a fixed soft-token budget, so resolution is moot.
    ("gemma4", 280),         # config.json vision_soft_tokens_per_image
    ("gemma3", 256),         # config.json mm_tokens_per_image
])
def test_vision_tokens_match_each_family_published_cost(family, expected):
    assert inference.estimate_vision_tokens(640, 640, family) == expected


@pytest.mark.parametrize("family", ["gemma4", "gemma3"])
@pytest.mark.parametrize("size", [(64, 48), (640, 640), (1280, 720)])
def test_gemma_vision_cost_does_not_vary_with_resolution(family, size):
    """A fixed soft-token budget is the whole point of the Gemma split.

    Tiling maths over-counted a 640x640 Gemma frame by 43% (400 vs 280), so
    the preflight refused 16-frame clips that fit in 8192 with room to spare.
    """
    assert inference.estimate_vision_tokens(
        *size, family
    ) == inference.estimate_vision_tokens(1, 1, family)


def test_qwen25vl_is_not_counted_with_the_qwen3_grid():
    """Qwen2.5-VL uses 14 px patches, so 32 px blocks UNDER-count it.

    This is the dangerous direction: the preflight passes and the caption
    then fails inside llama.cpp. 640x640 is 529 tokens, not 400.
    """
    assert inference.estimate_vision_tokens(640, 640, "qwen25vl") == 529
    assert inference.estimate_vision_tokens(
        640, 640, "qwen25vl"
    ) > inference.estimate_vision_tokens(640, 640, "qwen3vl")


def test_unknown_family_over_counts_rather_than_under_counts():
    """An unrecognised model must err toward refusing, not toward failing."""
    est = inference.estimate_vision_tokens
    unknown = est(640, 640, "something-new")
    assert unknown == est(640, 640, "qwen3vl")
    assert unknown >= est(640, 640, "gemma4")


def test_caption_video_uses_the_loaded_family_for_the_budget(
    monkeypatch, tmp_path
):
    """The preflight must cost frames with the family actually loaded.

    A Gemma load budgeted with the Qwen grid refuses clips that fit; the
    engine now passes self.chat_family through.
    """
    _install_fake_video(monkeypatch, frame_size=(640, 640))
    seen = []
    real = inference.estimate_vision_tokens
    monkeypatch.setattr(
        inference, "estimate_vision_tokens",
        lambda w, h, fam="qwen3vl": (seen.append(fam), real(w, h, fam))[1],
    )

    eng = _make_engine(response=_CANNED_RESPONSE)
    eng.chat_family = "gemma4"
    eng.caption_video(_video_file(tmp_path), "p", num_frames=4)

    assert seen == ["gemma4"] * 4


# ── The opaque-constructor retry must know every spelling CPython uses
#    to reject a keyword ────────────────────────────────────────────────
#
# This path exists only for constructors whose signature cannot be read —
# in practice, extension types. Those raise "'x' is an invalid keyword
# argument for f()" or "f() takes no keyword arguments"; only a pure-Python
# callable says "got an unexpected keyword argument", and an inspectable one
# never reaches here. Matching just that spelling left the chain dead exactly
# where it exists to work: the first attempt raised, the error propagated,
# and the narrower flag sets were never tried.

# Each spelling below was copied from a live interpreter, not invented:
#   datetime.datetime(2020, 1, 1, nope=2) -> "'nope' is an invalid keyword
#                                             argument for this function"
#   (1).to_bytes(nope=2)                  -> "... for to_bytes()"
#   object(nope=2)                        -> "object() takes no arguments"
KWARG_REJECTIONS = {
    "c-type-named": "'{flag}' is an invalid keyword argument for Handler()",
    "c-type-generic": "'{flag}' is an invalid keyword argument for this function",
    "c-type-no-kwargs": "Handler() takes no keyword arguments",
    "c-type-no-args": "Handler() takes no arguments",
    "pure-python": "__init__() got an unexpected keyword argument '{flag}'",
}


def _handler_rejecting(spelling, accepted=()):
    """A handler that refuses every flag except those in *accepted*."""
    template = KWARG_REJECTIONS[spelling]
    seen = {}

    class Handler:
        def __init__(self, clip_model_path, verbose=False, **flags):
            for flag in flags:
                if flag not in accepted:
                    raise TypeError(template.format(flag=flag))
            seen.update(flags)
            seen["built"] = True

    return Handler, seen


@pytest.mark.parametrize("spelling", sorted(KWARG_REJECTIONS))
def test_opaque_handler_retries_on_every_rejection_spelling(spelling, monkeypatch):
    """A handler accepting neither flag must still be constructed."""
    _make_signature_opaque(monkeypatch)
    Handler, seen = _handler_rejecting(spelling)

    handler = inference._construct_chat_handler(
        Handler, Path("/m/mmproj.gguf"), False
    )

    assert isinstance(handler, Handler)
    assert seen["built"] is True
    # It narrowed all the way to the no-flags attempt.
    assert "enable_thinking" not in seen
    assert "add_vision_id" not in seen


@pytest.mark.parametrize("spelling", sorted(KWARG_REJECTIONS))
def test_opaque_handler_stops_narrowing_at_a_flag_it_accepts(spelling, monkeypatch):
    """Degrading must not overshoot, whichever way the refusal is worded."""
    _make_signature_opaque(monkeypatch)
    Handler, seen = _handler_rejecting(spelling, accepted=("enable_thinking",))

    inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)

    assert seen.get("enable_thinking") is False
    assert "add_vision_id" not in seen


def test_broader_matching_did_not_cost_the_body_error_guarantee(monkeypatch):
    """A TypeError from inside the body still stops the chain at one call.

    This is what the narrow match was protecting. Widening it must not let a
    handler that refuses a construction on purpose be retried until some
    narrower flag set slips past the raise.
    """
    _make_signature_opaque(monkeypatch)
    calls = []

    class Handler:
        def __init__(self, clip_model_path, verbose=False, **flags):
            calls.append(flags)
            raise TypeError("mmproj is not a vision encoder")

    with pytest.raises(TypeError, match="not a vision encoder"):
        inference._construct_chat_handler(Handler, Path("/m/mmproj.gguf"), False)

    assert len(calls) == 1, "a body error must not be retried"


# ── An unknown chat family is rejected, not guessed at ──────────────────

def _full_namespace():
    return types.SimpleNamespace(
        Qwen3VLChatHandler=_Qwen3VLHandler,
        Qwen25VLChatHandler=_Qwen25Handler,
        Qwen35ChatHandler=_FamilyHandler,
        Gemma4ChatHandler=_FamilyHandler,
        Gemma3ChatHandler=_FamilyHandler,
    )


@pytest.mark.parametrize("bogus", ["gemm4", "qwen4vl", "", "GEMMA4"])
def test_an_unknown_chat_family_raises_instead_of_falling_back(bogus, monkeypatch):
    """A typo must not quietly drive a Gemma model with a Qwen template.

    The Gemma guard covered a *known* family whose handler is missing from
    the build. An unrecognised key skipped that guard and reached the Qwen
    fallback loop — the same silent mis-templating, by a different door.
    """
    monkeypatch.setattr(inference, "llama_chat_format", _full_namespace(),
                        raising=False)
    with pytest.raises(ValueError, match="Unknown chat_family"):
        inference._resolve_chat_handler_cls(bogus)


def test_the_unknown_family_error_names_the_families_that_exist(monkeypatch):
    monkeypatch.setattr(inference, "llama_chat_format", _full_namespace(),
                        raising=False)
    with pytest.raises(ValueError) as excinfo:
        inference._resolve_chat_handler_cls("gemm4")
    for known in inference.CHAT_FAMILY_HANDLERS:
        assert known in str(excinfo.value)


@pytest.mark.parametrize("family", sorted(inference.CHAT_FAMILY_HANDLERS))
def test_every_known_family_still_resolves(family, monkeypatch):
    """The validation must not reject anything that used to work."""
    monkeypatch.setattr(inference, "llama_chat_format", _full_namespace(),
                        raising=False)
    assert inference._resolve_chat_handler_cls(family) is not None


def test_infer_chat_family_can_never_produce_an_unknown_key():
    """The guard is reachable only by an explicit caller, by construction.

    Filename inference is the other source of a family string; if it could
    return something outside the map, this change would turn a working load
    into a hard failure.
    """
    names = [
        "Qwen3-VL-8B.gguf", "gemma-4-e4b.gguf", "gemma3-it.gguf",
        "Qwen3.5-9B.gguf", "Qwen2.5-VL-7B.gguf", "qwen2-vl-2b.gguf",
        "something-entirely-unknown.gguf", "", "....gguf",
    ]
    for name in names:
        assert inference.infer_chat_family(name) in inference.CHAT_FAMILY_HANDLERS
