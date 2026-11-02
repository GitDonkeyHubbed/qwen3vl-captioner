"""Tests for MLX model-folder detection (engine.mlx_engine).

An MLX model is a directory holding ``config.json`` plus at least one
``*.safetensors`` shard (the vision tower is embedded — no mmproj file).
"""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

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


def _mlx_supported_for(platform_name: str, machine: str) -> bool:
    """Re-evaluate MLX_SUPPORTED under a pretend platform.

    A private module copy is loaded so the cached engine.mlx_engine other
    tests import from is left untouched.
    """
    import importlib.util
    import platform as platform_mod

    spec = importlib.util.spec_from_file_location(
        "_mlx_engine_probe",
        Path(__file__).resolve().parent.parent / "engine" / "mlx_engine.py",
    )
    module = importlib.util.module_from_spec(spec)

    real_platform, real_machine = sys.platform, platform_mod.machine
    try:
        sys.platform = platform_name
        platform_mod.machine = lambda: machine
        spec.loader.exec_module(module)
        return module.MLX_SUPPORTED
    finally:
        sys.platform = real_platform
        platform_mod.machine = real_machine


@pytest.mark.parametrize(
    "platform_name, machine, expected",
    [
        ("darwin", "arm64", True),    # Apple Silicon — the only supported combo
        ("darwin", "x86_64", False),  # Intel Mac
        ("win32", "AMD64", False),
        ("linux", "x86_64", False),
        ("linux", "aarch64", False),  # ARM, but not Apple
    ],
)
def test_mlx_supported_truth_table(platform_name, machine, expected):
    """Assert the actual truth table, not the production expression.

    The old test re-derived `sys.platform == "darwin" and machine == "arm64"`
    as its own oracle, so it passed on x86-64 Linux CI for any mutation that
    is also False there — including dropping the arm64 check entirely.
    """
    assert _mlx_supported_for(platform_name, machine) is expected


def test_mlx_supported_matches_this_machine():
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


class _Chunk:
    def __init__(self, text):
        self.text = text


class _BareChunk:
    """No ``text`` attribute: the engine must fall back to ``str(chunk)``."""

    def __str__(self):
        return " tail"


def _jpeg(path):
    Image.new("RGB", (8, 4), "red").save(path, format="JPEG")


def _loaded_engine():
    engine = MlxVlmEngine()
    engine.model = object()
    engine.processor = object()
    engine.config = {"id": "test"}
    engine._is_loaded = True
    return engine


def _install_mlx(monkeypatch, stream_generate, apply_chat_template):
    mlx_vlm = types.ModuleType("mlx_vlm")
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = apply_chat_template
    mlx_vlm.stream_generate = stream_generate
    mlx_vlm.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    def make_sampler(temp, top_p):
        stream_generate.sampler_args = (temp, top_p)
        return ("sampler", temp, top_p)

    monkeypatch.setattr(
        "engine.mlx_engine._load_make_sampler", lambda: make_sampler,
    )


def test_mlx_caption_gates_sampling_and_omits_an_empty_system_turn(tmp_path, monkeypatch):
    """The MLX path must match the GGUF sampling gate and system-turn rule.

    temperature 0 forces top_p to 1.0, a blank system prompt is not sent, and
    the model sees an EXIF-corrected temp JPEG rather than the source file.
    """
    source = tmp_path / "photo.jpg"
    _jpeg(source)
    seen = {}
    tokens = []

    def apply_chat_template(processor, config, messages, num_images):
        seen["messages"] = messages
        seen["num_images"] = num_images
        return "PROMPT"

    def stream_generate(*args, **kwargs):
        seen["kwargs"] = kwargs
        seen["image"] = kwargs["image"][0]
        seen["existed"] = Path(seen["image"]).is_file()
        yield _Chunk("Caption: a cat")
        yield _Chunk("")
        yield _BareChunk()

    _install_mlx(monkeypatch, stream_generate, apply_chat_template)
    engine = _loaded_engine()

    result = engine.caption_image(
        source, "describe", system_prompt="", temperature=0, top_p=0.2,
        max_tokens=17, prefix=" pre ", suffix=" suf ",
        stream_callback=tokens.append,
    )

    assert seen["messages"] == [{"role": "user", "content": "describe"}]
    assert seen["num_images"] == 1
    assert stream_generate.sampler_args == (0.0, 1.0)
    assert seen["kwargs"]["max_tokens"] == 17
    assert "temperature" not in seen["kwargs"]
    assert seen["image"] != str(source)
    assert seen["existed"] is True
    assert not Path(seen["image"]).exists()
    assert tokens == ["Caption: a cat", " tail"]
    assert result == "pre a cat tail suf"

    result = engine.caption_image(
        source, "describe", temperature=0.5, top_p=0.4, max_tokens=17,
    )
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][0]["content"]
    assert seen["messages"][1] == {"role": "user", "content": "describe"}
    assert stream_generate.sampler_args == (0.5, 0.4)
    assert result == "a cat tail"


def test_mlx_caption_stops_before_later_tokens(tmp_path, monkeypatch):
    source = tmp_path / "photo.jpg"
    _jpeg(source)
    tokens = []
    state = {"n": 0}
    prepared = {}

    def apply_chat_template(processor, config, messages, num_images):
        return "PROMPT"

    def stream_generate(*args, **kwargs):
        prepared["image"] = kwargs["image"][0]
        yield _Chunk("one")
        yield _Chunk("TWO-SHOULD-STOP")

    def cancel_check():
        state["n"] += 1
        return state["n"] > 1

    _install_mlx(monkeypatch, stream_generate, apply_chat_template)
    result = _loaded_engine().caption_image(
        source, "describe", cancel_check=cancel_check,
        stream_callback=tokens.append,
    )

    assert result == "one"
    assert tokens == ["one"]
    assert prepared["image"] != str(source)
    assert not Path(prepared["image"]).exists()


def test_mlx_caption_refuses_a_missing_model_or_image(tmp_path):
    missing = tmp_path / "nope.jpg"
    engine = MlxVlmEngine()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.caption_image(missing, "describe")

    engine._is_loaded = True
    engine.model = object()
    with pytest.raises(FileNotFoundError, match="Image not found"):
        engine.caption_image(missing, "describe")
