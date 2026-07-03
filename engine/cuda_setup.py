"""
Windows CUDA toolkit detection, DLL path setup, and install diagnostics.

This module must be importable (and its setup function callable) BEFORE any
llama_cpp import. GPU drivers alone do not ship the CUDA runtime DLLs
(cudart64_*, cublas64_*, etc.) that ggml-cuda.dll depends on — those come
from the CUDA Toolkit. A missing toolkit, or a llama-cpp-python wheel built
for a different CUDA major version than the installed toolkit, are the two
most common causes of "Failed to load shared library ... ggml.dll" and
"access violation" errors on Windows (issues #8 and #10).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Default toolkit install root used by every NVIDIA installer
_TOOLKIT_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

# Map of (major, minor) toolkit floor -> JamePeng wheel tag.
# Checked in order; the first floor that is <= the installed version wins.
# All five tags exist for the pinned v0.3.40 release (cp312, win_amd64).
_WHEEL_TAGS = [
    ((13, 1), "cu131"),
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
]

DEFAULT_WHEEL_TAG = "cu124"

# Oldest toolkit any published wheel supports. A detected toolkit below this
# floor still maps to DEFAULT_WHEEL_TAG (setup scripts parse that contract),
# but diagnose()/doctor flag it as too old instead of reporting a false match.
MIN_SUPPORTED_TOOLKIT = (12, 4)


def parse_cuda_version(text: str) -> Optional[tuple[int, int]]:
    """Parse 'v12.4', '12.4', or 'v13.0' into a (major, minor) tuple."""
    m = re.search(r"v?(\d+)\.(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _cuda_install_roots() -> list[Path]:
    """Return CUDA toolkit install roots, newest version first.

    CUDA_PATH (set by the toolkit installer) is checked first, then every
    versioned directory under the standard install root, sorted numerically
    so v12.10 ranks above v12.4 and v13.x above both.
    """
    roots: list[Path] = []

    cuda_env = os.environ.get("CUDA_PATH")
    if cuda_env and Path(cuda_env).is_dir():
        roots.append(Path(cuda_env))

    if _TOOLKIT_ROOT.is_dir():
        versioned = []
        for p in _TOOLKIT_ROOT.iterdir():
            if p.is_dir():
                ver = parse_cuda_version(p.name)
                if ver:
                    versioned.append((ver, p))
        versioned.sort(key=lambda item: item[0], reverse=True)
        roots.extend(p for _, p in versioned)

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def detect_cuda_toolkit() -> Optional[tuple[tuple[int, int], Path]]:
    """Return ((major, minor), install_root) of the newest CUDA Toolkit, or None."""
    best: Optional[tuple[tuple[int, int], Path]] = None
    for root in _cuda_install_roots():
        ver = parse_cuda_version(root.name)
        if ver is None:
            # CUDA_PATH may point at a non-versioned dir; try version.json
            ver = _version_from_install(root)
        if ver and (best is None or ver > best[0]):
            best = (ver, root)
    return best


def _version_from_install(root: Path) -> Optional[tuple[int, int]]:
    """Try to read the toolkit version from version.json inside an install."""
    vfile = root / "version.json"
    if vfile.is_file():
        try:
            import json
            data = json.loads(vfile.read_text(encoding="utf-8"))
            ver_str = data.get("cuda", {}).get("version", "")
            return parse_cuda_version(ver_str)
        except Exception:
            pass
    return None


def recommended_wheel_tag(toolkit_version: Optional[tuple[int, int]]) -> str:
    """Map an installed toolkit version to the best matching wheel tag."""
    if toolkit_version is None:
        return DEFAULT_WHEEL_TAG
    for floor, tag in _WHEEL_TAGS:
        if toolkit_version >= floor:
            return tag
    return DEFAULT_WHEEL_TAG


def installed_wheel_cuda_tag() -> Optional[str]:
    """Return the CUDA tag ('cu124', 'cu130', ...) of the installed
    llama-cpp-python wheel, parsed from its version string, or None if
    llama-cpp-python is missing or is a CPU build."""
    try:
        from importlib.metadata import version
        ver = version("llama_cpp_python")
    except Exception:
        return None
    m = re.search(r"\+(cu\d+)", ver)
    return m.group(1) if m else None


def setup_cuda_dll_path() -> Optional[Path]:
    """Register CUDA and llama_cpp library directories on the DLL search path
    and preload the core CUDA runtime DLLs.

    Safe to call on any platform (no-op outside Windows) and safe to call
    multiple times. Returns the primary CUDA bin directory, or None if no
    toolkit was found.
    """
    if sys.platform != "win32":
        return None

    import ctypes

    bin_dirs: list[Path] = []
    root_version: dict[Path, tuple[int, int]] = {}
    for root in _cuda_install_roots():
        ver = parse_cuda_version(root.name) or _version_from_install(root) or (0, 0)
        for sub in ("bin", os.path.join("bin", "x64")):
            path = root / sub
            if path.is_dir():
                resolved = path.resolve()
                bin_dirs.append(resolved)
                root_version[resolved] = ver

    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            pass
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    # llama_cpp ships its own lib dir (ggml.dll etc.) — register it too so
    # dependent DLLs resolve even when the venv isn't on PATH.
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lib_dir = Path(spec.origin).resolve().parent / "lib"
            if lib_dir.is_dir():
                try:
                    os.add_dll_directory(str(lib_dir))
                except OSError:
                    pass
                os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    if not bin_dirs:
        return None

    # Preload runtime DLLs for whatever CUDA major version is present. Scan
    # ALL collected bin dirs, not just the first: CUDA 13.x on Windows moved
    # cudart64/cublas64 into bin\x64, so globbing only the plain bin dir made
    # this preload a silent no-op for exactly the cu130+ installs it targets.
    # Scan newest toolkit first (matching how the wheel is chosen) — the raw
    # bin_dirs order puts CUDA_PATH first, which may be an OLDER install, and
    # the per-name dedup must not let its DLLs win over a newer toolkit's.
    preload_order = sorted(
        bin_dirs, key=lambda d: root_version.get(d, (0, 0)), reverse=True
    )
    preload_patterns = ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
    preloaded: set[str] = set()
    for bin_dir in preload_order:
        for pattern in preload_patterns:
            for dll_path in sorted(bin_dir.glob(pattern), reverse=True):
                if dll_path.name.lower() in preloaded:
                    continue  # same DLL name already loaded from a newer root
                try:
                    ctypes.CDLL(str(dll_path), winmode=ctypes.RTLD_GLOBAL)
                    preloaded.add(dll_path.name.lower())
                except OSError:
                    pass

    return bin_dirs[0]


def cuda_toolkit_missing_message() -> str:
    """User-facing remediation text for a missing CUDA Toolkit."""
    return (
        "CUDA runtime DLLs were not found.\n\n"
        "Your NVIDIA GPU driver may be installed, but llama-cpp-python also "
        "needs the CUDA Toolkit (the driver alone is not enough).\n\n"
        "Fix:\n"
        "  1. Install the CUDA Toolkit:  winget install Nvidia.CUDA\n"
        "     (or download from https://developer.nvidia.com/cuda-downloads)\n"
        "  2. Re-run setup.bat so the matching llama-cpp-python wheel is installed.\n\n"
        "To see a full report of what's wrong, run diagnose.bat."
    )


def wheel_mismatch_message(toolkit_tag: str, wheel_tag: str) -> str:
    """User-facing remediation text for a toolkit/wheel version mismatch."""
    return (
        f"Your CUDA Toolkit needs the '{toolkit_tag}' build of llama-cpp-python, "
        f"but the '{wheel_tag}' build is installed.\n\n"
        "Fix: re-run setup.bat — it detects your CUDA version and installs the "
        "matching wheel.\n\n"
        "To see a full report, run diagnose.bat."
    )


def diagnose() -> dict:
    """Collect a structured report of the CUDA / llama-cpp install state.

    Used by doctor.py (diagnose.bat) and by the GUI to explain startup
    failures with specific, actionable advice.
    """
    report: dict = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "toolkit_version": None,
        "toolkit_path": None,
        "wheel_cuda_tag": None,
        "recommended_tag": None,
        "tags_match": None,
        "llama_cpp_installed": False,
        "llama_cpp_version": None,
        "llama_cpp_importable": False,
        "import_error": None,
        "gpu_name": None,
        "driver_version": None,
    }

    toolkit = detect_cuda_toolkit()
    if toolkit:
        ver, root = toolkit
        report["toolkit_version"] = f"{ver[0]}.{ver[1]}"
        report["toolkit_path"] = str(root)
        report["recommended_tag"] = recommended_wheel_tag(ver)
        # A toolkit older than every published wheel floor falls through to
        # the cu124 default, which such a toolkit cannot actually run — flag
        # it so doctor.py reports a real failure instead of a false OK match.
        report["toolkit_too_old"] = ver < MIN_SUPPORTED_TOOLKIT
    else:
        report["recommended_tag"] = DEFAULT_WHEEL_TAG
        report["toolkit_too_old"] = False

    try:
        from importlib.metadata import version
        report["llama_cpp_version"] = version("llama_cpp_python")
        report["llama_cpp_installed"] = True
    except Exception:
        pass

    report["wheel_cuda_tag"] = installed_wheel_cuda_tag()
    if report["wheel_cuda_tag"] and toolkit:
        report["tags_match"] = report["wheel_cuda_tag"] == report["recommended_tag"]

    if report["llama_cpp_installed"]:
        setup_cuda_dll_path()
        try:
            import importlib
            importlib.import_module("llama_cpp")
            report["llama_cpp_importable"] = True
        except Exception as e:
            report["import_error"] = str(e)

    # GPU info via NVML if available (best effort)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        report["gpu_name"] = name.decode() if isinstance(name, bytes) else name
        drv = pynvml.nvmlSystemGetDriverVersion()
        report["driver_version"] = drv.decode() if isinstance(drv, bytes) else drv
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return report


def startup_failure_advice(error_text: str) -> str:
    """Translate an early llama_cpp init failure into specific advice using
    the diagnostic report. Returns a user-facing message."""
    if sys.platform != "win32":
        return (
            "llama-cpp-python failed to initialize.\n\n"
            f"Error: {error_text}\n\n"
            "On Linux/macOS, install it with the appropriate backend flags "
            "(see README — Manual Installation)."
        )

    report = diagnose()

    if not report["llama_cpp_installed"]:
        return (
            "llama-cpp-python is not installed in the app's environment.\n\n"
            "Fix: run setup.bat."
        )

    if report["toolkit_version"] is None:
        return cuda_toolkit_missing_message()

    if report["tags_match"] is False:
        return wheel_mismatch_message(
            report["recommended_tag"], report["wheel_cuda_tag"]
        )

    return (
        "llama-cpp-python failed to initialize.\n\n"
        f"Error: {error_text}\n\n"
        f"Detected CUDA Toolkit: {report['toolkit_version']}\n"
        f"Installed wheel: {report['llama_cpp_version'] or 'unknown'}\n\n"
        "Run diagnose.bat for a full report, and include its output if you "
        "open a GitHub issue."
    )
