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
            if "WinError 127" in (report["import_error"] or ""):
                _report_winerror_127(report, problems)
            else:
                problems.append(
                    f"llama_cpp failed to load. Re-run {setup_cmd}; if it persists, "
                    "open a GitHub issue with this report."
                )
    else:
        print(f"{FAIL} llama-cpp-python: NOT INSTALLED")
        problems.append(f"Run {setup_cmd} to install all dependencies")


def _report_winerror_127(report: dict, problems: list):
    """WinError 127 = a dependency DLL was found but the loaded copy was
    missing a required function: an outdated MSVC/OpenMP runtime, or a
    same-named DLL loaded instead of the wheel's own copy."""
    shadowing = report.get("shadowing_dlls") or []
    causal = [s for s in shadowing if s[2] != "path"]
    on_path = [s for s in shadowing if s[2] == "path"]

    if causal:
        print(f"{WARN} DLL conflicts:   copies that OVERRIDE the app's DLL folders:")
        for name, directory, _ in causal:
            print(f"{INFO}                    {name}  in  {directory}")
    if on_path:
        print(f"{WARN} DLL conflicts:   same-named DLLs on PATH (less likely the cause):")
        for name, directory, _ in on_path:
            print(f"{INFO}                    {name}  in  {directory}")
    if not shadowing:
        print(f"{INFO} DLL conflicts:   none found in the scanned DLL search locations")

    problems.append(
        "WinError 127 means Windows loaded an incompatible version of a DLL\n"
        "         the engine needs. Most common fix: update the MSVC runtime with\n"
        "         winget install Microsoft.VCRedist.2015+.x64  — then reboot and\n"
        "         re-run diagnose.bat"
    )
    if causal:
        problems.append(
            "Conflicting DLL copies override the app's own (see the list above).\n"
            "         If they are leftovers from another app, rename or remove them —\n"
            "         but do NOT delete files inside the Windows directory (System32);\n"
            "         rename them (add .bak) from an administrator prompt instead."
        )
    elif on_path:
        problems.append(
            "The same-named DLLs on PATH (list above) usually belong to another\n"
            "         AI app (Ollama, LM Studio, ComfyUI, ...). If the error persists\n"
            "         after the runtime update, try removing their directory from PATH."
        )


def _windows_checks(report: dict, problems: list):
    """CUDA toolkit / wheel matching — the common Windows failure modes."""
    if report["gpu_name"]:
        print(f"{OK} GPU:             {report['gpu_name']} (driver {report['driver_version']})")
    elif not report.get("nvml_available"):
        # The NVML bindings are missing from the app environment — that says
        # nothing about the user's driver, so don't send them to nvidia.com.
        print(f"{WARN} GPU:             cannot query — the 'nvidia-ml-py' package is not installed")
        problems.append(
            "GPU status could not be checked: the 'nvidia-ml-py' package is missing "
            "from the app environment (this is not a driver problem).\n"
            "         Re-run setup.bat to reinstall dependencies."
        )
    else:
        print(f"{WARN} GPU:             NVML query failed — NVIDIA driver missing or no NVIDIA GPU")
        if report.get("gpu_query_error"):
            print(f"{INFO}                  {report['gpu_query_error']}")
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
    elif report["llama_cpp_installed"] and report.get("wheel_is_cuda_build"):
        # v0.3.40 wheels omit the +cuNNN tag from their dist metadata even
        # for CUDA builds (issue #22) — ggml-cuda.dll is the real signal.
        # Neutral, not OK: without a tag the toolkit match can't be verified.
        print(f"{INFO} Wheel build:      CUDA build (ggml-cuda.dll present; variant unknown — cannot verify toolkit match)")
    elif report["llama_cpp_installed"]:
        print(f"{WARN} Wheel build:      CPU build detected (no ggml-cuda.dll) — GPU acceleration disabled")


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
            print(f"{INFO}                  to enable: ./setup.sh")
            print(f"{INFO}                  (or: uv pip install --python .venv/bin/python mlx-vlm)")


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
