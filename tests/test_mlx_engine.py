"""Tests for MLX model-folder detection (engine.mlx_engine).

An MLX model is a directory holding ``config.json`` plus at least one
``*.safetensors`` shard (the vision tower is embedded — no mmproj file).
"""

import sys

import pytest

from engine.mlx_engine import (
    MLX_SUPPORTED,
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
