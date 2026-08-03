"""Tests for MLX model-folder detection and video captioning (engine.mlx_engine).

An MLX model is a directory holding ``config.json`` plus at least one
``*.safetensors`` shard (the vision tower is embedded — no mmproj file).

The caption_video tests fake the ``mlx_vlm`` modules in ``sys.modules`` and
monkeypatch ``engine.video.sample_frames``, so they run without mlx or cv2.
"""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import engine.video
from engine.base import DEFAULT_SYSTEM_PROMPT
from engine.mlx_engine import (
    MLX_SUPPORTED,
    MlxVlmEngine,
    _load_make_sampler,
    _stream_with_sampling,
    is_mlx_model_dir,
)


def _make_mlx_folder(path):
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\x00")


def test_is_mlx_model_dir_true_for_complete_folder(tmp_path):
    folder = tmp_path / "model"
    _make_mlx_folder(folder)
    assert is_mlx_model_dir(folder) is True


def test_is_mlx_model_dir_accepts_string_path(tmp_path):
    folder = tmp_path / "model"
    _make_mlx_folder(folder)
    assert is_mlx_model_dir(str(folder)) is True


def test_is_mlx_model_dir_false_when_missing(tmp_path):
    assert is_mlx_model_dir(tmp_path / "nope") is False


def test_is_mlx_model_dir_false_for_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    assert is_mlx_model_dir(f) is False


def test_is_mlx_model_dir_false_without_config(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"\x00")
    assert is_mlx_model_dir(folder) is False


def test_is_mlx_model_dir_false_without_safetensors(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert is_mlx_model_dir(folder) is False


def test_mlx_supported_matches_platform():
    import platform

    expected = sys.platform == "darwin" and platform.machine() == "arm64"
    assert MLX_SUPPORTED == expected


def test_load_make_sampler_never_raises():
    # Returns None when mlx_lm is missing/too old (the common non-Mac case), or
    # the callable factory if present — but must never raise.
    result = _load_make_sampler()
    assert result is None or callable(result)


# --- _stream_with_sampling: version-tolerant mlx-vlm dispatch ---------------


def test_stream_with_sampling_uses_sampler_when_supported():
    captured = {}

    def fake_make_sampler(temp, top_p):
        captured["sampler_args"] = (temp, top_p)
        return ("sampler", temp, top_p)

    def fake_stream_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return iter(["ok"])

    out = list(
        _stream_with_sampling(
            fake_stream_generate,
            ("model", "processor", "prompt"),
            {"image": ["x"], "max_tokens": 10},
            0.6,
            0.9,
            make_sampler=fake_make_sampler,
        )
    )

    assert out == ["ok"]
    assert captured["sampler_args"] == (0.6, 0.9)
    assert "sampler" in captured["kwargs"]
    assert "temperature" not in captured["kwargs"]
    assert "top_p" not in captured["kwargs"]
    # Non-sampling kwargs are still forwarded.
    assert captured["kwargs"]["max_tokens"] == 10


def test_stream_with_sampling_falls_back_to_legacy_kwargs():
    captured = {}

    def fake_make_sampler(temp, top_p):
        return "sampler"

    def fake_stream_generate(*args, **kwargs):
        # Emulate an older build whose signature has no `sampler` parameter.
        if "sampler" in kwargs:
            raise TypeError("unexpected keyword argument 'sampler'")
        captured["kwargs"] = kwargs
        return iter(["legacy"])

    out = list(
        _stream_with_sampling(
            fake_stream_generate,
            ("model", "processor", "prompt"),
            {"image": ["x"], "max_tokens": 10},
            0.6,
            0.9,
            make_sampler=fake_make_sampler,
        )
    )

    assert out == ["legacy"]
    assert captured["kwargs"]["temperature"] == 0.6
    assert captured["kwargs"]["top_p"] == 0.9
    assert "sampler" not in captured["kwargs"]


def test_stream_with_sampling_legacy_when_no_make_sampler():
    captured = {}

    def fake_stream_generate(*args, **kwargs):
        captured["kwargs"] = kwargs
        return iter(["x"])

    list(
        _stream_with_sampling(
            fake_stream_generate,
            (),
            {},
            0.0,
            1.0,
            make_sampler=None,
        )
    )

    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["top_p"] == 1.0
    assert "sampler" not in captured["kwargs"]


def test_stream_with_sampling_propagates_unrelated_typeerror():
    # A TypeError from stream_generate that is NOT about the `sampler` kwarg is
    # a genuine error and must propagate, not silently retry legacy kwargs.
    legacy_called = {"hit": False}

    def fake_make_sampler(temp, top_p):
        return "sampler"

    def fake_stream_generate(*args, **kwargs):
        if "sampler" in kwargs:
            raise TypeError("internal explosion unrelated to sampling")
        legacy_called["hit"] = True
        return iter(["legacy"])

    with pytest.raises(TypeError, match="internal explosion"):
        _stream_with_sampling(
            fake_stream_generate,
            ("m", "p", "prompt"),
            {"max_tokens": 10},
            0.6,
            0.9,
            make_sampler=fake_make_sampler,
        )
    assert legacy_called["hit"] is False


def test_stream_with_sampling_propagates_make_sampler_error():
    # If make_sampler() itself fails, that must surface — falling back to legacy
    # kwargs on a newer mlx-vlm would reintroduce the crash this guards against.
    legacy_called = {"hit": False}

    def fake_make_sampler(temp, top_p):
        raise TypeError("make_sampler signature mismatch")

    def fake_stream_generate(*args, **kwargs):
        legacy_called["hit"] = True
        return iter(["legacy"])

    with pytest.raises(TypeError, match="make_sampler"):
        _stream_with_sampling(
            fake_stream_generate,
            ("m", "p", "prompt"),
            {"max_tokens": 10},
            0.6,
            0.9,
            make_sampler=fake_make_sampler,
        )
    assert legacy_called["hit"] is False


# --- caption_video: frame staging, num_images invariant, cleanup -------------


def _solid_frames(n, size=(64, 48)):
    return [Image.new("RGB", size, (i * 30 % 256, 0, 0)) for i in range(n)]


def _loaded_engine():
    """An MlxVlmEngine faked into the loaded state (no real mlx-vlm)."""
    eng = MlxVlmEngine()
    eng.model = object()
    eng.processor = object()
    eng.config = {}
    eng._is_loaded = True
    return eng


def _install_fake_mlx_vlm(monkeypatch, stream_generate, apply_chat_template):
    """Register fake mlx_vlm modules so caption_video's lazy imports resolve."""
    root = types.ModuleType("mlx_vlm")
    root.stream_generate = stream_generate
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = apply_chat_template
    root.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)


def _make_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    return clip


def test_caption_video_stages_frames_and_matches_num_images(tmp_path, monkeypatch):
    captured = {}

    # 3 frames despite the default request of 8 — as if decoding dropped some.
    monkeypatch.setattr(
        engine.video, "sample_frames", lambda path, n: _solid_frames(3)
    )

    def fake_apply_chat_template(processor, config, messages, num_images):
        captured["messages"] = messages
        captured["num_images"] = num_images
        return "formatted"

    def fake_stream_generate(model, processor, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        # The staged PNGs must exist while generation runs.
        captured["existed_during_call"] = [
            Path(p).is_file() for p in kwargs["image"]
        ]
        return iter([types.SimpleNamespace(text="A cat runs.")])

    _install_fake_mlx_vlm(monkeypatch, fake_stream_generate, fake_apply_chat_template)

    caption = _loaded_engine().caption_video(
        _make_clip(tmp_path), "Describe the video."
    )

    assert caption == "A cat runs."
    assert captured["prompt"] == "formatted"
    assert captured["messages"] == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Describe the video."},
    ]

    paths = captured["kwargs"]["image"]
    assert isinstance(paths, list)
    assert all(isinstance(p, str) for p in paths)
    # Staged in temporal order under zero-padded names.
    assert [Path(p).name for p in paths] == [
        "frame_000.png", "frame_001.png", "frame_002.png",
    ]
    # num_images must equal the number of images actually passed.
    assert captured["num_images"] == len(paths) == 3
    assert captured["existed_during_call"] == [True, True, True]
    # The temp dir (and every staged frame) is gone after the call.
    assert all(not Path(p).exists() for p in paths)


def test_caption_video_downscales_staged_frames(tmp_path, monkeypatch):
    sizes = []

    monkeypatch.setattr(
        engine.video,
        "sample_frames",
        lambda path, n: [Image.new("RGB", (1280, 720), "red") for _ in range(2)],
    )

    def fake_stream_generate(model, processor, prompt, **kwargs):
        for p in kwargs["image"]:
            with Image.open(p) as im:
                sizes.append(im.size)
        return iter([types.SimpleNamespace(text="ok")])

    _install_fake_mlx_vlm(
        monkeypatch,
        fake_stream_generate,
        lambda processor, config, messages, num_images: "formatted",
    )

    _loaded_engine().caption_video(_make_clip(tmp_path), "Describe.")

    # 1280x720 thumbnails to 640x360 (max dim 640, aspect preserved).
    assert sizes == [(640, 360), (640, 360)]


def test_caption_video_cancel_mid_stream_returns_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        engine.video, "sample_frames", lambda path, n: _solid_frames(2)
    )

    staged = {}

    def fake_stream_generate(model, processor, prompt, **kwargs):
        staged["paths"] = list(kwargs["image"])
        return iter([
            types.SimpleNamespace(text="A dog"),
            types.SimpleNamespace(text=" jumps"),
            types.SimpleNamespace(text=" high."),
        ])

    _install_fake_mlx_vlm(
        monkeypatch,
        fake_stream_generate,
        lambda processor, config, messages, num_images: "formatted",
    )

    seen = []
    caption = _loaded_engine().caption_video(
        _make_clip(tmp_path),
        "Describe.",
        stream_callback=seen.append,
        cancel_check=lambda: len(seen) >= 1,
    )

    # Cancelled after the first token: partial caption, no crash.
    assert caption == "A dog"
    assert seen == ["A dog"]
    assert all(not Path(p).exists() for p in staged["paths"])


def test_caption_video_cleans_up_temp_frames_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        engine.video, "sample_frames", lambda path, n: _solid_frames(2)
    )

    staged = {}

    def fake_stream_generate(model, processor, prompt, **kwargs):
        staged["paths"] = list(kwargs["image"])
        raise RuntimeError("Metal ran out of memory")

    _install_fake_mlx_vlm(
        monkeypatch,
        fake_stream_generate,
        lambda processor, config, messages, num_images: "formatted",
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        _loaded_engine().caption_video(_make_clip(tmp_path), "Describe.")

    assert staged["paths"]  # frames were staged before the failure
    assert all(not Path(p).exists() for p in staged["paths"])


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 2), (1, 2), (8, 8), (16, 16), (99, 16)],
)
def test_caption_video_clamps_num_frames(tmp_path, monkeypatch, requested, expected):
    asked = {}

    def fake_sample_frames(path, num_frames):
        asked["num_frames"] = num_frames
        return _solid_frames(2)

    monkeypatch.setattr(engine.video, "sample_frames", fake_sample_frames)
    _install_fake_mlx_vlm(
        monkeypatch,
        lambda *args, **kwargs: iter([types.SimpleNamespace(text="ok")]),
        lambda processor, config, messages, num_images: "formatted",
    )

    _loaded_engine().caption_video(
        _make_clip(tmp_path), "Describe.", num_frames=requested
    )

    assert asked["num_frames"] == expected


def test_caption_video_applies_prefix_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        engine.video, "sample_frames", lambda path, n: _solid_frames(2)
    )
    _install_fake_mlx_vlm(
        monkeypatch,
        lambda *args, **kwargs: iter([types.SimpleNamespace(text="A cat runs.")]),
        lambda processor, config, messages, num_images: "formatted",
    )

    caption = _loaded_engine().caption_video(
        _make_clip(tmp_path), "Describe.", prefix="Video:", suffix="[end]"
    )

    assert caption == "Video: A cat runs. [end]"


def test_caption_video_requires_loaded_model(tmp_path):
    with pytest.raises(RuntimeError, match="not loaded"):
        MlxVlmEngine().caption_video(_make_clip(tmp_path), "Describe.")


def test_caption_video_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loaded_engine().caption_video(tmp_path / "missing.mp4", "Describe.")
