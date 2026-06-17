"""Tests for CUDA toolkit detection and wheel-tag mapping (engine.cuda_setup).

Regression guard for the ggml.dll / wheel-mismatch crashes in issues #8/#10:
an installed CUDA Toolkit version must map to the matching llama-cpp-python
wheel tag, and toolkit version comparison must be numeric (so 12.10 ranks
above 12.8, not below it as a string compare would).
"""

import pytest

from engine import cuda_setup
from engine.cuda_setup import (
    DEFAULT_WHEEL_TAG,
    cuda_toolkit_missing_message,
    detect_cuda_toolkit,
    installed_wheel_cuda_tag,
    parse_cuda_version,
    recommended_wheel_tag,
    wheel_mismatch_message,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("v12.4", (12, 4)),
        ("12.4", (12, 4)),
        ("v13.0", (13, 0)),
        ("CUDA Version 12.10", (12, 10)),
        ("v13.1", (13, 1)),
    ],
)
def test_parse_cuda_version_ok(text, expected):
    assert parse_cuda_version(text) == expected


@pytest.mark.parametrize("text", ["", "v13", "nonsense", "cuda"])
def test_parse_cuda_version_rejects_garbage(text):
    assert parse_cuda_version(text) is None


@pytest.mark.parametrize(
    "version, tag",
    [
        ((13, 1), "cu131"),
        ((13, 5), "cu131"),
        ((14, 0), "cu131"),
        ((13, 0), "cu130"),
        ((12, 8), "cu128"),
        ((12, 9), "cu128"),
        ((12, 6), "cu126"),
        ((12, 7), "cu126"),
        ((12, 4), "cu124"),
        ((12, 5), "cu124"),
    ],
)
def test_recommended_wheel_tag(version, tag):
    assert recommended_wheel_tag(version) == tag


def test_recommended_wheel_tag_none_is_default():
    assert recommended_wheel_tag(None) == DEFAULT_WHEEL_TAG
    assert DEFAULT_WHEEL_TAG == "cu124"


def test_recommended_wheel_tag_below_floor_is_default():
    # Toolkits older than any shipped wheel fall back to the default tag.
    assert recommended_wheel_tag((12, 3)) == DEFAULT_WHEEL_TAG
    assert recommended_wheel_tag((11, 8)) == DEFAULT_WHEEL_TAG


def test_wheel_tag_ordering_is_numeric_not_lexical():
    # 12.10 must rank above 12.8 (a string compare would pick the wrong tag).
    assert recommended_wheel_tag((12, 10)) == "cu128"
    assert recommended_wheel_tag((12, 10)) != "cu131"


def test_cu131_tag_is_reachable():
    tags = [tag for _, tag in cuda_setup._WHEEL_TAGS]
    assert "cu131" in tags
    assert recommended_wheel_tag((13, 1)) == "cu131"


def test_installed_wheel_cuda_tag_shape():
    # None when no CUDA llama-cpp-python wheel is installed; a "cuNNN" string
    # when one is present. Either way it must never raise.
    tag = installed_wheel_cuda_tag()
    assert tag is None or tag.startswith("cu")


def test_toolkit_missing_message_is_actionable():
    msg = cuda_toolkit_missing_message()
    assert "CUDA Toolkit" in msg
    assert "setup.bat" in msg


def test_wheel_mismatch_message_names_both_tags():
    msg = wheel_mismatch_message("cu130", "cu124")
    assert "cu130" in msg
    assert "cu124" in msg
    assert "setup.bat" in msg


def test_detect_cuda_toolkit_picks_newest_numerically(tmp_path, monkeypatch):
    # Three installed toolkits; detection picks v13.0 and ranks v12.10 above
    # v12.4 (numeric, not lexical).
    monkeypatch.delenv("CUDA_PATH", raising=False)
    for name in ("v12.4", "v12.10", "v13.0"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(cuda_setup, "_TOOLKIT_ROOT", tmp_path)

    detected = detect_cuda_toolkit()
    assert detected is not None
    version, root = detected
    assert version == (13, 0)
    assert root.name == "v13.0"


def test_detect_cuda_toolkit_none_when_no_install(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(cuda_setup, "_TOOLKIT_ROOT", tmp_path / "does-not-exist")
    assert detect_cuda_toolkit() is None
