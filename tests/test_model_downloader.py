"""Tests for the mmproj auto-downloader (engine.model_downloader).

These avoid the network by injecting a stub ``huggingface_hub`` module, so they
also act as a regression guard: ``hf_hub_download`` must be called WITHOUT the
``local_dir_use_symlinks`` argument, which is removed in huggingface_hub 1.0
(the version this project targets) and raises TypeError there.
"""

import sys
import types
from pathlib import Path

import pytest

from engine.model_downloader import (
    download_mmproj,
    download_named_mmproj,
    find_mmproj_file,
)


@pytest.fixture
def stub_hf(monkeypatch):
    """Install a fake huggingface_hub whose hf_hub_download records its kwargs."""
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        dest = Path(kwargs["local_dir"]) / kwargs["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00")
        return str(dest)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)
    return calls


def test_download_named_mmproj_omits_deprecated_symlink_kwarg(stub_hf, tmp_path):
    result = download_named_mmproj("some/repo", "vision.mmproj.gguf", tmp_path)
    assert len(stub_hf) == 1
    kwargs = stub_hf[0]
    assert "local_dir_use_symlinks" not in kwargs
    assert kwargs["repo_id"] == "some/repo"
    assert kwargs["filename"] == "vision.mmproj.gguf"
    assert kwargs["local_dir"] == str(tmp_path)
    assert result.name == "vision.mmproj.gguf"


def test_download_mmproj_omits_deprecated_symlink_kwarg(stub_hf, tmp_path):
    download_mmproj(tmp_path)
    assert stub_hf, "expected hf_hub_download to be called"
    for kwargs in stub_hf:
        assert "local_dir_use_symlinks" not in kwargs


def test_find_mmproj_file_locates_mmproj(tmp_path):
    (tmp_path / "model.Q4_K_M.gguf").write_bytes(b"\x00")
    (tmp_path / "model.mmproj-f16.gguf").write_bytes(b"\x00")
    found = find_mmproj_file(tmp_path)
    assert found is not None
    assert "mmproj" in found.name.lower()


def test_find_mmproj_file_prefers_f16(tmp_path):
    """With several encoders present, the f16 mmproj is chosen deterministically
    (not whatever iterdir() happens to yield first)."""
    (tmp_path / "model.mmproj-Q8_0.gguf").write_bytes(b"\x00")
    f16 = tmp_path / "model.mmproj-f16.gguf"
    f16.write_bytes(b"\x00")
    assert find_mmproj_file(tmp_path) == f16


def test_find_mmproj_file_none_when_absent(tmp_path):
    (tmp_path / "model.Q4_K_M.gguf").write_bytes(b"\x00")
    assert find_mmproj_file(tmp_path) is None


def test_find_mmproj_file_none_for_missing_dir(tmp_path):
    assert find_mmproj_file(tmp_path / "does-not-exist") is None
