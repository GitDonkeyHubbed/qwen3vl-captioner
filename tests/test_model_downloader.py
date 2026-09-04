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
    MmprojMismatchError,
    default_mmproj_fits,
    download_mmproj,
    download_named_mmproj,
    ensure_mmproj,
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


# ── Model-aware mmproj pairing ──────────────────────────────────────────
#
# Pairing a model with another model's vision encoder does not fail cleanly:
# llama.cpp crashes natively on the first caption. Taking any *mmproj*.gguf in
# the folder made that the default outcome for a browsed model.

def _touch(path: Path):
    path.write_bytes(b"")
    return path


def test_pairs_encoder_named_after_the_model(tmp_path):
    model = _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf")
    _touch(tmp_path / "Gliese-Qwen3.5-4B-Abliterated-Caption.mmproj-f16.gguf")
    match = _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) == match


def test_refuses_a_foreign_encoder(tmp_path):
    # The only encoder present belongs to a different model.
    model = _touch(tmp_path / "Gliese-Qwen3.5-4B-Abliterated-Caption.Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) is None


def test_bare_mmproj_pairs_when_the_folder_holds_one_model(tmp_path):
    # noctrex-style layout: the encoder is just "mmproj-F16.gguf".
    model = _touch(tmp_path / "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "mmproj-F16.gguf")

    assert find_mmproj_file(tmp_path, model) == mmproj


def test_bare_mmproj_is_ambiguous_with_several_models(tmp_path):
    model = _touch(tmp_path / "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf")
    _touch(tmp_path / "Gliese-Qwen3.5-4B-Abliterated-Caption.Q4_K_M.gguf")
    _touch(tmp_path / "mmproj-F16.gguf")

    assert find_mmproj_file(tmp_path, model) is None


def test_size_mismatch_is_refused(tmp_path):
    model = _touch(tmp_path / "Qwen3-VL-4B-Instruct.Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct.mmproj-f16.gguf")
    _touch(tmp_path / "other-model.gguf")  # keeps the folder ambiguous

    assert find_mmproj_file(tmp_path, model) is None


def test_qwen35_size_token_is_not_misread(tmp_path):
    # "Qwen3.5-2B" must parse as a 2B model, not a 352B one.
    from engine.model_downloader import _size_tokens
    assert _size_tokens("Gliese-Qwen3.5-2B-Abliterated-Caption.Q4_K_M.gguf") == {"2"}
    assert _size_tokens("Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf") == {"8"}


def test_no_model_given_keeps_legacy_behaviour(tmp_path):
    # Callers that only ask "is there an encoder here at all?" still get one.
    _touch(tmp_path / "Qwen3-VL-8B-Instruct.mmproj-f16.gguf")
    assert find_mmproj_file(tmp_path) is not None


def test_ensure_mmproj_refuses_the_default_for_a_non_8b_model(tmp_path, stub_hf):
    model = _touch(tmp_path / "Gliese-Qwen3.5-4B-Abliterated-Caption.Q4_K_M.gguf")
    with pytest.raises(MmprojMismatchError) as excinfo:
        ensure_mmproj(tmp_path, model_path=model)
    assert "Qwen3-VL 8B encoder" in str(excinfo.value)


def test_ensure_mmproj_still_downloads_for_a_matching_model(tmp_path, stub_hf):
    model = _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf")
    result = ensure_mmproj(tmp_path, model_path=model)
    assert result.name.endswith(".gguf")


def test_default_mmproj_fits(tmp_path):
    assert default_mmproj_fits(None) is True
    assert default_mmproj_fits(Path("Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf")) is True
    assert default_mmproj_fits(Path("Qwen3-VL-4B-Instruct.Q4_K_M.gguf")) is False
    assert default_mmproj_fits(Path("Gliese-Qwen3.5-8B-Caption.Q4_K_M.gguf")) is False
