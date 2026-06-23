"""Tests for CUDA toolkit detection and wheel-tag mapping (engine.cuda_setup).

Regression guard for the ggml.dll / wheel-mismatch crashes in issues #8/#10:
an installed CUDA Toolkit version must map to the matching llama-cpp-python
wheel tag, and toolkit version comparison must be numeric (so 12.10 ranks
above 12.8, not below it as a string compare would).
"""

import json
import sys

import pytest

from engine import cuda_setup
from engine.cuda_setup import (
    DEFAULT_WHEEL_TAG,
    _version_from_install,
    cuda_toolkit_missing_message,
    detect_cuda_toolkit,
    installed_wheel_cuda_tag,
    parse_cuda_version,
    recommended_wheel_tag,
    startup_failure_advice,
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
        ("v13.3", (13, 3)),  # regression: exact version from issue #10 report
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
        ((13, 3), "cu131"),  # regression: CUDA 13.3 from issue #10 must use cu131
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


def test_detect_cuda_toolkit_with_version_json(tmp_path, monkeypatch):
    # CUDA_PATH points at a non-versioned directory whose version is read
    # from version.json — the fallback path used when the installer creates
    # a directory without a version number in its name.
    monkeypatch.delenv("CUDA_PATH", raising=False)
    install_dir = tmp_path / "cuda"
    install_dir.mkdir()
    (install_dir / "version.json").write_text(
        json.dumps({"cuda": {"version": "13.3.0"}}), encoding="utf-8"
    )
    monkeypatch.setattr(cuda_setup, "_TOOLKIT_ROOT", install_dir.parent)
    monkeypatch.setenv("CUDA_PATH", str(install_dir))

    detected = detect_cuda_toolkit()
    assert detected is not None
    version, root = detected
    assert version == (13, 3)
    # Confirm the detected version maps to the expected wheel tag (issue #10 regression).
    assert recommended_wheel_tag(version) == "cu131"


def test_version_from_install_reads_version_json(tmp_path):
    (tmp_path / "version.json").write_text(
        json.dumps({"cuda": {"version": "12.6.0"}}), encoding="utf-8"
    )
    assert _version_from_install(tmp_path) == (12, 6)


def test_version_from_install_returns_none_when_missing(tmp_path):
    assert _version_from_install(tmp_path) is None


def test_version_from_install_returns_none_for_corrupt_json(tmp_path):
    (tmp_path / "version.json").write_text("not valid json", encoding="utf-8")
    assert _version_from_install(tmp_path) is None


def test_startup_failure_advice_non_windows(monkeypatch):
    # On non-Windows platforms the function returns a generic message.
    monkeypatch.setattr(sys, "platform", "linux")
    msg = startup_failure_advice("some error")
    assert "llama-cpp-python failed" in msg
    assert "some error" in msg


def test_startup_failure_advice_cuda_13_3_maps_to_cu131(tmp_path, monkeypatch):
    # Simulate a Windows machine with CUDA 13.3 installed but the wrong wheel.
    # The advice must name the correct recommended tag (cu131).
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    (tmp_path / "v13.3").mkdir()
    monkeypatch.setattr(cuda_setup, "_TOOLKIT_ROOT", tmp_path)

    # Patch installed_wheel_cuda_tag to simulate the mismatched cu124 wheel.
    monkeypatch.setattr(cuda_setup, "installed_wheel_cuda_tag", lambda: "cu124")

    # Patch the rest of diagnose() so we don't need an actual llama_cpp install.
    def _fake_diagnose():
        r = {
            "platform": "win32",
            "python": "3.12.0",
            "toolkit_version": "13.3",
            "toolkit_path": str(tmp_path / "v13.3"),
            "wheel_cuda_tag": "cu124",
            "recommended_tag": "cu131",
            "tags_match": False,
            "llama_cpp_installed": True,
            "llama_cpp_version": "0.3.40+cu124",
            "llama_cpp_importable": False,
            "import_error": None,
            "gpu_name": None,
            "driver_version": None,
        }
        return r

    monkeypatch.setattr(cuda_setup, "diagnose", _fake_diagnose)
    msg = startup_failure_advice("ggml.dll not found")
    assert "cu131" in msg
    assert "setup.bat" in msg
