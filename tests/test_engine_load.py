"""Load-time refusals for both engines.

A missing weight file must fail before llama.cpp or mlx-vlm is constructed,
and a second load must unload the model that is already resident.
"""

import sys
import types

import pytest

from engine.inference import Qwen3VLEngine
from engine.mlx_engine import MlxVlmEngine


def _mlx_folder(path):
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\x00")
    return path


def _install_mlx(monkeypatch):
    import engine.mlx_engine as mlx

    calls = []
    mlx_mod = types.ModuleType("mlx_vlm")
    utils = types.ModuleType("mlx_vlm.utils")

    def load(path):
        calls.append(("load", path))
        return object(), object()

    def load_config(path):
        calls.append(("config", path))
        return {}

    mlx_mod.load = load
    utils.load_config = load_config
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return mlx, calls


def test_mlx_load_refuses_when_the_backend_is_unavailable(tmp_path, monkeypatch):
    mlx, calls = _install_mlx(monkeypatch)
    monkeypatch.setattr(mlx, "MLX_VLM_AVAILABLE", False)
    folder = _mlx_folder(tmp_path / "model")
    engine = MlxVlmEngine()

    with pytest.raises(RuntimeError, match="setup.sh"):
        engine.load_model(folder)

    assert calls == []
    assert engine.is_loaded is False


def test_mlx_load_rejects_a_non_model_then_unloads_on_reload(tmp_path, monkeypatch):
    mlx, calls = _install_mlx(monkeypatch)
    monkeypatch.setattr(mlx, "MLX_VLM_AVAILABLE", True)
    engine = MlxVlmEngine()

    with pytest.raises(FileNotFoundError, match="config.json"):
        engine.load_model(tmp_path / "missing")
    config_only = tmp_path / "config-only"
    config_only.mkdir()
    (config_only / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="config.json"):
        engine.load_model(config_only)
    assert calls == []
    assert engine.is_loaded is False

    first = _mlx_folder(tmp_path / "first")
    messages = []
    engine.load_model(first, progress_callback=messages.append)
    assert [path for _kind, path in calls] == [str(first), str(first)]
    assert messages[0].startswith("Loading MLX model")
    assert messages[-1].endswith("loaded successfully.")
    info = engine.get_model_info()
    assert info["status"] == "Loaded"
    assert info["model_file"] == "first"
    assert info["backend"] == "MLX (Apple Silicon)"
    assert "mmproj_file" not in info

    unloaded = []
    original = engine.unload

    def spy():
        unloaded.append(engine.model)
        original()

    engine.unload = spy
    second = _mlx_folder(tmp_path / "second")
    engine.load_model(second)
    assert unloaded and unloaded[0] is not None
    assert engine.model is not unloaded[0]
    assert engine.is_loaded is True
    assert engine.get_model_info()["model_file"] == "second"


class _Factory:
    """Stand-in constructor. A new object per call, so reload is visible."""

    def __init__(self, log, kind):
        self.log = log
        self.kind = kind

    def __call__(self, **kwargs):
        obj = types.SimpleNamespace(**kwargs)
        self.log.append((self.kind, kwargs))
        return obj


def _install_llama(monkeypatch):
    import engine.inference as inference

    log = []
    monkeypatch.setattr(inference, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(
        inference, "Qwen25VLChatHandler", _Factory(log, "handler"), raising=False,
    )
    monkeypatch.setattr(inference, "Llama", _Factory(log, "llama"), raising=False)
    return inference, log


def test_gguf_load_stops_before_llama_when_a_file_is_missing(tmp_path, monkeypatch):
    inference, log = _install_llama(monkeypatch)
    engine = Qwen3VLEngine()
    assert engine.get_model_info() == {"status": "Not loaded"}

    model = tmp_path / "model.gguf"
    encoder = tmp_path / "encoder.gguf"
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        engine.load_model(model, encoder)
    model.write_bytes(b"gguf")
    with pytest.raises(FileNotFoundError, match="Vision encoder"):
        engine.load_model(model, encoder)
    assert log == []
    assert engine.is_loaded is False

    monkeypatch.setattr(inference, "LLAMA_CPP_AVAILABLE", False)
    encoder.write_bytes(b"mmproj")
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        engine.load_model(model, encoder)
    assert log == []


def test_gguf_load_passes_the_encoder_and_unloads_on_reload(tmp_path, monkeypatch):
    _inference, log = _install_llama(monkeypatch)
    model = tmp_path / "model.gguf"
    encoder = tmp_path / "encoder.gguf"
    model.write_bytes(b"gguf")
    encoder.write_bytes(b"mmproj")
    engine = Qwen3VLEngine()
    messages = []

    engine.load_model(model, encoder, progress_callback=messages.append)

    assert [kind for kind, _kwargs in log] == ["handler", "llama"]
    assert log[0][1]["clip_model_path"] == str(encoder)
    assert log[1][1]["model_path"] == str(model)
    assert log[1][1]["n_ctx"] == 8192
    assert log[1][1]["n_gpu_layers"] == -1
    assert log[1][1]["chat_handler"] is engine.chat_handler
    assert messages[0].startswith("Loading vision encoder")
    assert messages[-1].endswith("loaded successfully.")
    info = engine.get_model_info()
    assert info["model_file"] == "model.gguf"
    assert info["mmproj_file"] == "encoder.gguf"
    assert info["status"] == "Loaded"

    with pytest.raises(FileNotFoundError, match="Image not found"):
        engine.caption_image(tmp_path / "gone.png", "describe")
    assert engine.is_loaded is True

    unloaded = []
    original = engine.unload

    def spy():
        unloaded.append(engine.model)
        original()

    engine.unload = spy
    engine.load_model(model, encoder)
    assert unloaded and unloaded[0] is not None
    assert engine.model is not unloaded[0]
    assert engine.is_loaded is True


def test_gguf_caption_refuses_an_unloaded_model(tmp_path):
    engine = Qwen3VLEngine()
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.caption_image(tmp_path / "gone.png", "describe")
