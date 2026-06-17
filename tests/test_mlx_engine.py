"""Tests for MLX model-folder detection (engine.mlx_engine).

An MLX model is a directory holding ``config.json`` plus at least one
``*.safetensors`` shard (the vision tower is embedded — no mmproj file).
"""

import sys

from engine.mlx_engine import MLX_SUPPORTED, is_mlx_model_dir


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
