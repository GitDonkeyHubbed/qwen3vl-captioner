"""Tests for doctor.py — the install-diagnostics report users paste into issues.

The Windows smoke workflow treats exit 1 as "problems found" (expected on a
GPU-less runner) and exit 0 as healthy. A regression that prints a false
"[OK] Wheel/CUDA match" for a toolkit older than 12.4, or that treats the
optional MLX extra as a hard failure on Mac, is exactly how those exit codes
went wrong before.
"""

import platform

import doctor


def _report(**overrides):
    report = {
        "platform": "win32",
        "python": "3.12.0",
        "toolkit_version": "12.8",
        "toolkit_path": r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        "wheel_cuda_tag": "cu128",
        "wheel_is_cuda_build": True,
        "recommended_tag": "cu128",
        "tags_match": True,
        "shadowing_dlls": [],
        "llama_cpp_installed": True,
        "llama_cpp_version": "0.3.40",
        "llama_cpp_importable": True,
        "import_error": None,
        "gpu_name": "NVIDIA GeForce RTX 4080",
        "driver_version": "560.94",
        "nvml_available": True,
        "gpu_query_error": None,
        "toolkit_too_old": False,
    }
    report.update(overrides)
    return report


def test_old_toolkit_is_a_failure_even_when_tags_happen_to_match(capsys):
    """diagnose() used to compare a too-old toolkit against the cu124 fallback
    and print a green match that the toolkit cannot actually run."""
    problems = []
    doctor._windows_checks(
        _report(
            toolkit_version="12.3",
            toolkit_too_old=True,
            wheel_cuda_tag="cu124",
            recommended_tag="cu124",
            tags_match=True,
        ),
        problems,
    )
    out = capsys.readouterr().out
    assert "[FAIL] CUDA Toolkit:" in out
    assert "12.3" in out
    assert "[FAIL] Wheel/CUDA match:" in out
    assert "[OK] Wheel/CUDA match:" not in out
    assert any("12.3" in p and "too old" in p for p in problems)


def test_missing_nvml_package_is_not_a_driver_problem(capsys):
    problems = []
    doctor._windows_checks(
        _report(gpu_name=None, nvml_available=False, gpu_query_error="pynvml import failed"),
        problems,
    )
    out = capsys.readouterr().out
    assert "nvidia-ml-py" in out
    assert "nvidia.com/drivers" not in out
    assert any("nvidia-ml-py" in p for p in problems)
    assert not any("nvidia.com/drivers" in p for p in problems)


def test_winerror_127_warns_against_deleting_system32(capsys):
    problems = []
    doctor._windows_checks(
        _report(
            llama_cpp_importable=False,
            import_error="[WinError 127] The specified procedure could not be found",
            shadowing_dlls=[
                ("libomp140.x86_64.dll", r"C:\Windows\System32", "system"),
            ],
        ),
        problems,
    )
    out = capsys.readouterr().out
    assert "libomp140.x86_64.dll" in out
    assert "System32" in out
    assert any("VCRedist" in p for p in problems)
    assert any("do NOT delete" in p and "System32" in p for p in problems)


def test_macos_missing_mlx_is_not_a_problem(monkeypatch, capsys):
    """A healthy Metal-only install must not exit 1 over the optional extra."""
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    problems = []
    doctor._macos_checks(_report(), problems)
    out = capsys.readouterr().out
    assert problems == []
    assert "MLX backend" in out


def test_main_returns_zero_when_linux_engine_imports(monkeypatch, capsys):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    monkeypatch.setattr(doctor, "diagnose", lambda: _report())
    assert doctor.main() == 0
    assert "All checks passed" in capsys.readouterr().out


def test_main_returns_one_when_windows_toolkit_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(
        doctor,
        "diagnose",
        lambda: _report(toolkit_version=None, toolkit_path=None, tags_match=None),
    )
    assert doctor.main() == 1
    out = capsys.readouterr().out
    assert "PROBLEMS FOUND" in out
    assert "CUDA Toolkit" in out
