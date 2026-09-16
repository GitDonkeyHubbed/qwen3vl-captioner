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

    def sample_frames(video_path, num_frames=8, cancel_check=None):
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


def test_caption_video_uncancelled_run_is_unchanged(monkeypatch, tmp_path):
    """A live-but-false predicate must not alter a normal run."""
    _install_fake_video(monkeypatch)
    eng = _make_engine(response=_CANNED_RESPONSE)

    caption = eng.caption_video(
        _video_file(tmp_path), "p", num_frames=4, cancel_check=lambda: False
    )

    assert caption == "a cat walks by"
    assert len(eng.model.calls) == 1
