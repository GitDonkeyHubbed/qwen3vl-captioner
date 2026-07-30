"""Tests for CUDA toolkit detection and wheel-tag mapping (engine.cuda_setup).

Regression guard for the ggml.dll / wheel-mismatch crashes in issues #8/#10:
an installed CUDA Toolkit version must map to the matching llama-cpp-python
wheel tag, and toolkit version comparison must be numeric (so 12.10 ranks
above 12.8, not below it as a string compare would).
"""

import os

import pytest

from engine import cuda_setup
from engine.cuda_setup import (
    DEFAULT_WHEEL_TAG,
    cuda_toolkit_missing_message,
    detect_cuda_toolkit,
    dll_shadowing_message,
    find_shadowing_dlls,
    installed_wheel_cuda_tag,
    installed_wheel_is_cuda_build,
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


# --- CUDA-build detection (issue #22) -------------------------------------
# The v0.3.40 wheels keep +cuNNN only in the wheel FILENAME; the installed
# dist metadata reports plain '0.3.40'. Detection must fall back to the
# ggml-cuda.dll the CUDA builds ship, or every healthy CUDA install gets a
# false "CPU build detected" warning.


def test_is_cuda_build_true_when_version_has_tag(monkeypatch):
    monkeypatch.setattr(cuda_setup, "installed_wheel_cuda_tag", lambda: "cu131")
    assert installed_wheel_is_cuda_build() is True


def test_is_cuda_build_true_via_ggml_cuda_dll(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "ggml-cuda.dll").touch()
    monkeypatch.setattr(cuda_setup, "installed_wheel_cuda_tag", lambda: None)
    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [lib])
    assert installed_wheel_is_cuda_build() is True


def test_is_cuda_build_false_without_ggml_cuda_dll(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "ggml.dll").touch()
    monkeypatch.setattr(cuda_setup, "installed_wheel_cuda_tag", lambda: None)
    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [lib])
    assert installed_wheel_is_cuda_build() is False


def test_is_cuda_build_none_when_not_installed(monkeypatch):
    import importlib.metadata

    def raise_missing(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(cuda_setup, "installed_wheel_cuda_tag", lambda: None)
    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [])
    monkeypatch.setattr(importlib.metadata, "version", raise_missing)
    assert installed_wheel_is_cuda_build() is None


# --- DLL shadowing scan (WinError 127 diagnostics) ------------------------


def test_find_shadowing_dlls_flags_rogue_copy(tmp_path, monkeypatch):
    wheel = tmp_path / "wheel-lib"
    wheel.mkdir()
    # unrelated.dll ships in the wheel too, so only the family-prefix
    # filter (not the wheel-name intersection) can exclude the rogue copy.
    for name in ("ggml.dll", "ggml-base.dll", "ggml-cuda.dll", "unrelated.dll"):
        (wheel / name).touch()

    rogue = tmp_path / "other-app"
    rogue.mkdir()
    (rogue / "ggml-base.dll").touch()
    (rogue / "unrelated.dll").touch()  # not llama.cpp-family: never flagged

    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [wheel])
    monkeypatch.setenv("PATH", str(rogue))

    found = find_shadowing_dlls()
    assert ("ggml-base.dll", str(rogue.resolve()), "path") in found
    assert all(name != "unrelated.dll" for name, _, _ in found)


def test_find_shadowing_dlls_excludes_wheel_own_dirs(tmp_path, monkeypatch):
    wheel = tmp_path / "wheel-lib"
    wheel.mkdir()
    (wheel / "ggml.dll").touch()

    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [wheel])
    # The wheel's own dir on PATH (setup_cuda_dll_path puts it there) must
    # not be reported as a conflict with itself.
    monkeypatch.setenv("PATH", str(wheel))

    assert all(
        directory != str(wheel.resolve())
        for _, directory, _ in find_shadowing_dlls()
    )


def test_find_shadowing_dlls_flags_cwd_copy(tmp_path, monkeypatch):
    # CWD outranks PATH (and the wheel dirs) in the legacy dependent-DLL
    # search order the engine load uses — a rogue copy there must be found
    # and ranked as causal, not as an ordinary PATH hit.
    wheel = tmp_path / "wheel-lib"
    wheel.mkdir()
    (wheel / "ggml.dll").touch()

    cwd = tmp_path / "launch-dir"
    cwd.mkdir()
    (cwd / "ggml.dll").touch()

    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [wheel])
    monkeypatch.setenv("PATH", "")
    monkeypatch.chdir(cwd)

    assert ("ggml.dll", str(cwd.resolve()), "cwd") in find_shadowing_dlls()


def test_find_shadowing_dlls_survives_bad_path_entries(tmp_path, monkeypatch):
    # Nonexistent dirs and file-as-dir PATH entries must be skipped without
    # aborting the scan or discarding hits from other entries.
    wheel = tmp_path / "wheel-lib"
    wheel.mkdir()
    (wheel / "ggml.dll").touch()

    rogue = tmp_path / "other-app"
    rogue.mkdir()
    (rogue / "ggml.dll").touch()

    not_a_dir = tmp_path / "actually-a-file"
    not_a_dir.touch()

    monkeypatch.setattr(cuda_setup, "_llama_cpp_dll_dirs", lambda: [wheel])
    path = os.pathsep.join(
        [str(tmp_path / "does-not-exist"), str(not_a_dir), str(rogue)]
    )
    monkeypatch.setenv("PATH", path)

    assert ("ggml.dll", str(rogue.resolve()), "path") in find_shadowing_dlls()


def test_dll_shadowing_message_lists_conflicts_and_always_suggests_vcredist():
    # The MSVC-runtime advice must not be suppressed by conflict hits: a
    # PATH hit is often innocent while the stale runtime is the real cause.
    msg = dll_shadowing_message([("ggml-base.dll", r"C:\SomeApp", "path")])
    assert "WinError 127" in msg
    assert "ggml-base.dll" in msg
    assert r"C:\SomeApp" in msg
    assert "VCRedist" in msg


def test_dll_shadowing_message_ranks_causal_hits_with_system32_caution():
    msg = dll_shadowing_message(
        [("libomp140.x86_64.dll", r"C:\Windows\System32", "system")]
    )
    assert "libomp140.x86_64.dll" in msg
    assert "BEFORE" in msg
    assert "NOT delete" in msg
    assert "VCRedist" in msg


def test_dll_shadowing_message_without_conflicts_suggests_vcredist():
    msg = dll_shadowing_message([])
    assert "Visual C++" in msg
    assert "VCRedist" in msg


# --- CUDA tag recovery from PEP 610 direct_url.json -----------------------


def test_installed_wheel_cuda_tag_recovered_from_direct_url(monkeypatch):
    # v0.3.40 metadata is tag-less, but setup.bat installs from a URL whose
    # wheel filename carries '%2BcuNNN' — recorded in direct_url.json.
    import importlib.metadata

    class FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return (
                '{"url": "https://github.com/JamePeng/llama-cpp-python/'
                "releases/download/v0.3.40-cu131-win-20260608/"
                'llama_cpp_python-0.3.40%2Bcu131-cp312-cp312-win_amd64.whl"}'
            )

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.3.40")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: FakeDist())
    assert installed_wheel_cuda_tag() == "cu131"


def test_installed_wheel_cuda_tag_none_without_direct_url(monkeypatch):
    import importlib.metadata

    class FakeDist:
        def read_text(self, name):
            return None  # importlib returns None when the file is absent

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.3.40")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: FakeDist())
    assert installed_wheel_cuda_tag() is None
