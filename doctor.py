"""
Installation diagnostics for Qwen3-VL Captioner.

Run via diagnose.bat (Windows), or:  .venv/bin/python doctor.py (macOS/Linux)

Prints a report of the install state with specific remediation steps.
If you open a GitHub issue about installation problems, please paste
this report into it.
"""

import platform
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.cuda_setup import diagnose  # noqa: E402


OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[ -- ]"


def _check_llama(report: dict, problems: list, setup_cmd: str):
    """Shared llama-cpp-python install/import checks."""
    if report["llama_cpp_installed"]:
        print(f"{OK} llama-cpp-python: {report['llama_cpp_version']}")
        if report["llama_cpp_importable"]:
            print(f"{OK} Engine import:   llama_cpp loads successfully")
        else:
            print(f"{FAIL} Engine import:   {report['import_error']}")
            problems.append(
                f"llama_cpp failed to load. Re-run {setup_cmd}; if it persists, "
                "open a GitHub issue with this report."
            )
    else:
        print(f"{FAIL} llama-cpp-python: NOT INSTALLED")
        problems.append(f"Run {setup_cmd} to install all dependencies")


def _windows_checks(report: dict, problems: list):
    """CUDA toolkit / wheel matching — the common Windows failure modes."""
    if report["gpu_name"]:
        print(f"{OK} GPU:             {report['gpu_name']} (driver {report['driver_version']})")
    else:
        print(f"{WARN} GPU:             could not query NVML — NVIDIA driver missing or no NVIDIA GPU")
        problems.append(
            "Install/update the NVIDIA GPU driver: https://www.nvidia.com/drivers"
        )

    if report["toolkit_version"] and report.get("toolkit_too_old"):
        print(f"{FAIL} CUDA Toolkit:    v{report['toolkit_version']} is older than the minimum supported 12.4")
        problems.append(
            f"CUDA Toolkit v{report['toolkit_version']} is too old for the published "
            "llama-cpp-python wheels (oldest build is cu124).\n"
            "         Upgrade the toolkit:  winget install Nvidia.CUDA  — then re-run setup.bat"
        )
    elif report["toolkit_version"]:
        print(f"{OK} CUDA Toolkit:    v{report['toolkit_version']}  ({report['toolkit_path']})")
    else:
        print(f"{FAIL} CUDA Toolkit:    NOT FOUND (the GPU driver alone is not enough)")
        problems.append(
            "Install the CUDA Toolkit:  winget install Nvidia.CUDA\n"
            "         (or https://developer.nvidia.com/cuda-downloads), then re-run setup.bat"
        )

    _check_llama(report, problems, "setup.bat")

    wheel_tag = report["wheel_cuda_tag"]
    rec_tag = report["recommended_tag"]
    if wheel_tag:
        if report.get("toolkit_too_old"):
            # tags_match compares against the cu124 fallback, which a too-old
            # toolkit cannot run — an "OK match" here would be a lie.
            print(f"{FAIL} Wheel/CUDA match: wheel '{wheel_tag}' cannot run on toolkit v{report['toolkit_version']} (needs 12.4+)")
        elif report["tags_match"] is False:
            print(f"{FAIL} Wheel/CUDA match: wheel is '{wheel_tag}' but your toolkit needs '{rec_tag}'")
            problems.append(
                f"Re-run setup.bat — it will replace the {wheel_tag} wheel with the {rec_tag} build"
            )
        elif report["tags_match"]:
            print(f"{OK} Wheel/CUDA match: wheel '{wheel_tag}' matches toolkit (needs '{rec_tag}')")
        else:
            print(f"{WARN} Wheel/CUDA match: wheel is '{wheel_tag}' but no toolkit found to compare")
    elif report["llama_cpp_installed"]:
        print(f"{WARN} Wheel build:      CPU build detected (no CUDA tag) — GPU acceleration disabled")


def _macos_checks(report: dict, problems: list):
    """Metal / MLX checks — no CUDA on Macs."""
    arch = platform.machine()
    if arch == "arm64":
        print(f"{OK} Architecture:    Apple Silicon ({arch}) — Metal GPU acceleration available")
    else:
        print(f"{WARN} Architecture:    Intel ({arch}) — CPU only; MLX unavailable")

    _check_llama(report, problems, "./setup.sh")

    if arch == "arm64":
        try:
            from importlib.metadata import version
            mlx_ver = version("mlx-vlm")
            print(f"{OK} MLX backend:     mlx-vlm {mlx_ver} installed")
        except Exception:
            # Optional backend — print the suggestion inline rather than
            # appending to `problems`, so a healthy Metal-only setup doesn't
            # exit 1 / print "PROBLEMS FOUND" over a missing optional extra.
            print(f"{WARN} MLX backend:     mlx-vlm not installed (optional — MLX models hidden)")
            print(f"{INFO}                  to enable: ./setup.sh  (or: .venv/bin/pip install mlx-vlm)")


def main() -> int:
    print("=" * 64)
    print("  Qwen3-VL Captioner — Install Diagnostics")
    print("=" * 64)

    report = diagnose()
    problems: list = []

    print(f"{INFO} Platform:        {report['platform']} ({platform.machine()})")
    print(f"{INFO} Python:          {report['python']}")

    if sys.platform == "win32":
        _windows_checks(report, problems)
    elif sys.platform == "darwin":
        _macos_checks(report, problems)
    else:
        _check_llama(report, problems, "pip install (see README — Linux)")

    print("-" * 64)
    if problems:
        print("  PROBLEMS FOUND — suggested fixes (in order):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        print()
        print("  After fixing, run the diagnostics again to verify.")
        result = 1
    else:
        print("  All checks passed. If the app still fails, open a GitHub")
        print("  issue and include this report.")
        result = 0
    print("=" * 64)
    return result


if __name__ == "__main__":
    # Exit codes: 0 = healthy, 1 = problems found (expected on e.g. GPU-less
    # CI runners), 2 = doctor itself crashed. CI normalizes only 1, so a
    # genuine crash in the diagnostics still fails the workflow.
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
