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


def test_same_size_encoder_from_foreign_family_is_refused(tmp_path):
    model = _touch(tmp_path / "Qwen3-VL-8B-Instruct.Q4_K_M.gguf")
    _touch(tmp_path / "Gliese-Qwen3.5-8B-Caption.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) is None


def test_same_family_and_size_fallback_is_preserved(tmp_path):
    model = _touch(tmp_path / "PublisherA-Qwen3-VL-8B-Instruct.Q4_K_M.gguf")
    encoder = _touch(tmp_path / "PublisherB-Qwen3-VL-8B.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) == encoder


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


# ── Family identity (stage 3) and K-quant names (stage 1) ───────────────
#
# Stage 3 used to pair on parameter size alone, so any same-size encoder in
# the folder was taken — a Qwen3.5-4B or Gemma-3-4B model got the Qwen3-VL-4B
# encoder. And because "Q4_K_M" left "k"/"m" in the model key, stage 1 never
# recognised a K-quant model's own encoder, leaving that choice to stage 3.

# The owner's real model folder, filenames only.
_REAL_FOLDER_MODELS = (
    "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf",
    "Qwen3-VL-8B-Instruct-abliterated-v2.Q2_K.gguf",
    "Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf",
)
_REAL_FOLDER_ENCODER = "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf"


@pytest.mark.parametrize("model_name", _REAL_FOLDER_MODELS)
def test_real_folder_models_all_pair_with_the_v2_encoder(tmp_path, model_name):
    for name in _REAL_FOLDER_MODELS:
        _touch(tmp_path / name)
    encoder = _touch(tmp_path / _REAL_FOLDER_ENCODER)

    assert find_mmproj_file(tmp_path, tmp_path / model_name) == encoder


@pytest.mark.parametrize("model_name, encoder_name", [
    # Official Qwen naming: "mmproj-" in front and "Qwen3VL" joined.
    ("Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
    # The publisher prefix on the encoder side instead of the model side.
    ("Qwen3-VL-8B-Instruct-abliterated-v2.Q8_0.gguf",
     "Huihui-Qwen3-VL-8B-Instruct-abliterated.mmproj-f16.gguf"),
    # A different publisher prefix on each side.
    ("Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf",
     "mmproj-Qwen_Qwen3-VL-8B-Instruct-f16.gguf"),
    ("Qwen_Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
     "Huihui-Qwen3-VL-8B-Instruct-abliterated.mmproj-f16.gguf"),
    ("huihui-ai_Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf",
     "mmproj-Qwen_Qwen3VL-4B-Instruct-f16.gguf"),
])
def test_same_family_encoder_pairs_across_publishers(
    tmp_path, model_name, encoder_name
):
    model = _touch(tmp_path / model_name)
    _touch(tmp_path / "unrelated-model.gguf")  # rules out the bare-mmproj stage
    encoder = _touch(tmp_path / encoder_name)

    assert find_mmproj_file(tmp_path, model) == encoder


@pytest.mark.parametrize("model_name", [
    "Qwen3.5-4B-Q4_K_M.gguf",
    "Gliese-Qwen3.5-4B-Abliterated-Caption.Q4_K_M.gguf",
    "gemma-3-4b-it-Q4_K_M.gguf",
])
def test_same_size_encoder_of_another_family_is_refused(tmp_path, model_name):
    model = _touch(tmp_path / model_name)
    _touch(tmp_path / "Qwen3-VL-4B-Instruct.Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-4B-Instruct.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) is None


def test_llama_text_model_is_refused_the_qwen3vl_8b_encoder(tmp_path, stub_hf):
    model = _touch(tmp_path / "Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) is None
    # ensure_mmproj agrees: no pairing, and no download of the 8B Qwen3-VL
    # encoder either — it raises and names the encoder it would not use.
    with pytest.raises(MmprojMismatchError, match="belongs to a different model"):
        ensure_mmproj(tmp_path, model_path=model)
    assert stub_hf == []


def test_a_size_without_a_family_is_refused(tmp_path):
    # Its name is also contained in the encoder's, so neither the size nor the
    # name containment of stage 1 may pair it without a family.
    model = _touch(tmp_path / "8B-Instruct-Q4_K_M.gguf")
    _touch(tmp_path / "other-model.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) is None


def test_same_family_distinguishes_qwen_generations():
    from engine.model_downloader import _same_family
    assert _same_family("Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                        "Huihui-Qwen3-VL-8B-Instruct-abliterated.mmproj-f16.gguf")
    assert not _same_family("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                            "Qwen3-VL-7B.mmproj-f16.gguf")
    assert not _same_family("Qwen3.5-4B-Q4_K_M.gguf", "Qwen3-VL-4B.mmproj-f16.gguf")
    # Prefixes on both sides may not reduce the match to a bare shared tail.
    assert not _same_family("Foo-Qwen2.5-VL-7B-Q4_K_M.gguf",
                            "mmproj-Bar-Qwen3.5-VL-7B-f16.gguf")
    assert not _same_family("Keye-VL-8B-Preview-Q4_K_M.gguf",
                            "mmproj-Qwen_Qwen3-VL-8B-Instruct-f16.gguf")
    assert not _same_family("Foo-InternVL3-8B-Q4_K_M.gguf",
                            "mmproj-Bar-InternVL3_5-8B-f16.gguf")
    # No size token: the family is unknown, so it never matches.
    assert not _same_family("Qwen3-VL-Instruct.Q4_K_M.gguf",
                            "Qwen3-VL-Instruct.mmproj-f16.gguf")


@pytest.mark.parametrize("quant", ["Q4_K_M", "Q2_K", "Q6_K", "Q8_0", "IQ4_NL", "IQ2_XXS"])
def test_k_quant_model_takes_its_own_encoder_at_stage_one(tmp_path, quant):
    # The Huihui encoder sorts first and is the same family and size, so it
    # would win at stage 3; only a stage-1 match picks the model's own one.
    model = _touch(tmp_path / f"Qwen3-VL-8B-Instruct-abliterated-v2.{quant}.gguf")
    _touch(tmp_path / "Huihui-Qwen3-VL-8B-Instruct-abliterated.mmproj-f16.gguf")
    own = _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) == own


def test_exact_encoder_name_beats_a_longer_one_that_contains_it(tmp_path):
    model = _touch(tmp_path / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
    _touch(tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf")
    own = _touch(tmp_path / "Qwen3-VL-8B-Instruct.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) == own


def test_own_encoder_wins_over_an_alphabetically_earlier_foreign_one(tmp_path):
    model = _touch(tmp_path / "Qwen3.5-4B-Instruct.Q4_K_M.gguf")
    _touch(tmp_path / "gemma-3-4b-it.Q4_K_M.gguf")
    _touch(tmp_path / "gemma-3-4b-it.mmproj-f16.gguf")
    own = _touch(tmp_path / "Qwen3.5-4B-Instruct.mmproj-f16.gguf")

    assert find_mmproj_file(tmp_path, model) == own


def test_bare_q8_0_mmproj_pairs_when_the_folder_holds_one_model(tmp_path):
    # "mmproj-Q8_0" left a stray "0", so it was not seen as a bare encoder.
    model = _touch(tmp_path / "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf")
    mmproj = _touch(tmp_path / "mmproj-Q8_0.gguf")

    assert find_mmproj_file(tmp_path, model) == mmproj


def test_ensure_mmproj_uses_a_same_family_encoder_and_refuses_a_foreign_one(
    tmp_path, stub_hf
):
    same = tmp_path / "same"
    same.mkdir()
    model = _touch(same / "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf")
    _touch(same / "Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf")
    encoder = _touch(same / _REAL_FOLDER_ENCODER)
    assert ensure_mmproj(same, model_path=model) == encoder

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    model = _touch(foreign / "Qwen3.5-4B-Q4_K_M.gguf")
    _touch(foreign / "Qwen3-VL-4B-Instruct.Q4_K_M.gguf")
    _touch(foreign / "Qwen3-VL-4B-Instruct.mmproj-f16.gguf")
    with pytest.raises(MmprojMismatchError, match="belongs to a different model"):
        ensure_mmproj(foreign, model_path=model)

    assert stub_hf == []  # neither case downloads anything


# ── A publisher prefix may be dropped, a model name may not ─────────────
#
# The trailing-token family match took any shorter family found at the end of
# the longer one as "the same model under a publisher prefix". A version-less
# family such as the "vl" of "VL-8B" is the tail of Qwen3-VL and Keye-VL alike,
# so it paired with the Qwen3-VL-8B encoder — at stage 1 through name
# containment ("vl8b" in "qwen3vl8b") and at stage 3 through size and family.

@pytest.mark.parametrize("model_name, encoder_name", [
    ("Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf",
     "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf"),
    ("Qwen3-VL-8B-Instruct-abliterated-v2.Q6_K.gguf",
     "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf"),
    ("Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf",
     "mmproj-Qwen_Qwen3-VL-8B-Instruct-f16.gguf"),
    ("Qwen3-VL-4B-Instruct-Q8_0.gguf", "mmproj-Qwen3-VL-4B-Instruct-F16.gguf"),
    ("Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf",
     "mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"),
    ("PublisherA-Qwen3-VL-8B-Instruct.Q4_K_M.gguf",
     "PublisherB-Qwen3-VL-8B.mmproj-f16.gguf"),
    # The version may be its own token ("gemma 3"), not only "qwen3".
    ("google_gemma-3-4b-it-Q4_K_M.gguf", "mmproj-gemma-3-4b-it-f16.gguf"),
    # Nothing dropped: a version-less family still pairs with itself.
    ("Keye-VL-8B-Preview-Q4_K_M.gguf", "mmproj-Keye-VL-8B-f16.gguf"),
    # A version-less family under an org prefix (bartowski's "org_model"
    # against ggml-org's bare name) is a publisher, not another model.
    ("mistral-community_pixtral-12b-Q4_K_M.gguf", "mmproj-pixtral-12b-f16.gguf"),
    ("XiaomiMiMo_MiMo-VL-7B-RL-Q4_K_M.gguf", "mmproj-MiMo-VL-7B-RL-f16.gguf"),
    ("CohereForAI_aya-vision-8b-Q4_K_M.gguf", "mmproj-aya-vision-8b-f16.gguf"),
])
def test_publisher_prefixed_encoder_still_pairs(tmp_path, model_name, encoder_name):
    model = _touch(tmp_path / model_name)
    _touch(tmp_path / "unrelated-model.gguf")  # rules out the bare-mmproj stage
    encoder = _touch(tmp_path / encoder_name)

    assert find_mmproj_file(tmp_path, model) == encoder


@pytest.mark.parametrize("model_name, encoder_name", [
    ("Qwen3.5-4B-Q4_K_M.gguf", "Qwen3-VL-4B-Instruct.mmproj-f16.gguf"),
    ("Gliese-Qwen3.5-4B-Abliterated-Caption.Q4_K_M.gguf",
     "Qwen3-VL-4B-Instruct.mmproj-f16.gguf"),
    ("gemma-3-4b-it-Q4_K_M.gguf", "Qwen3-VL-4B-Instruct.mmproj-f16.gguf"),
    ("Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf"),
    ("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3-VL-7B-F16.gguf"),
    ("Qwen3-VL-2B-Instruct-Q4_K_M.gguf", "Qwen3-VL-8B-Instruct.mmproj-f16.gguf"),
    # A version-less family on the model side (stage 1 containment) ...
    ("VL-8B-Q4_K_M.gguf", "Qwen3-VL-8B.mmproj-f16.gguf"),
    ("VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf"),
    # ... and where only stage 3 (same size and family) could pair it.
    ("VL-8B-Chat-Q4_K_M.gguf", "Qwen3-VL-8B-Instruct.mmproj-f16.gguf"),
    ("VL-8B-Q4_K_M.gguf", "Keye-VL-8B.mmproj-f16.gguf"),
    # A dropped token carrying a version ("v1") is part of the model's name.
    ("llava-v1.6-mistral-7b-Q4_K_M.gguf", "mmproj-Mistral-7B-Instruct-f16.gguf"),
])
def test_encoder_of_another_model_is_refused(tmp_path, model_name, encoder_name):
    model = _touch(tmp_path / model_name)
    _touch(tmp_path / "unrelated-model.gguf")  # rules out the bare-mmproj stage
    _touch(tmp_path / encoder_name)

    assert find_mmproj_file(tmp_path, model) is None


def test_same_family_does_not_drop_a_model_name_as_a_prefix():
    from engine.model_downloader import _same_family
    assert not _same_family("VL-8B-Q4_K_M.gguf", "Qwen3-VL-8B.mmproj-f16.gguf")
    assert not _same_family("Qwen3-VL-8B.mmproj-f16.gguf", "VL-8B-Q4_K_M.gguf")
    assert _same_family("Qwen3-VL-8B-Q4_K_M.gguf",
                        "Huihui-Qwen3-VL-8B.mmproj-f16.gguf")
