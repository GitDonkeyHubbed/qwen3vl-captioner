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
    llama-cpp-python is missing or its version carries no CUDA tag.

    None does NOT mean a CPU build: the v0.3.40 JamePeng wheels keep the
    +cuNNN tag in the wheel FILENAME but ship plain '0.3.40' in their
    dist metadata (issue #22), so use installed_wheel_is_cuda_build() to
    decide CPU vs CUDA and treat this tag as a bonus when present."""
    try:
        from importlib.metadata import version
        ver = version("llama_cpp_python")
    except Exception:
        return None
    m = re.search(r"\+(cu\d+)", ver)
    if m:
        return m.group(1)
    # Tag-less metadata (v0.3.40): recover the tag from the install-source
    # URL that pip/uv record per PEP 610 — setup.bat installs from a URL
    # whose wheel filename still carries '+cuNNN' ('%2BcuNNN' when encoded).
    try:
        from importlib.metadata import distribution
        direct_url = distribution("llama_cpp_python").read_text("direct_url.json")
        if direct_url:
            m = re.search(r"(?:%2B|\+)(cu\d+)", direct_url, re.IGNORECASE)
            if m:
                return m.group(1).lower()
    except Exception:
        pass
    return None


def _llama_cpp_dll_dirs() -> list[Path]:
    """Return the installed llama_cpp package's DLL directories.

    Older wheels ship everything in llama_cpp/lib; the v0.3.40 wheels ship
    both llama_cpp/lib and llama_cpp/bin. Returns only directories that
    exist, or [] when llama-cpp-python is not installed.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if not (spec and spec.origin):
            return []
        pkg_root = Path(spec.origin).resolve().parent
        return [pkg_root / sub for sub in ("lib", "bin") if (pkg_root / sub).is_dir()]
    except Exception:
        # A stat failure (e.g. WinError 5 from AV/EDR interference) must
        # degrade gracefully — this runs at app import time.
        return []


def installed_wheel_is_cuda_build() -> Optional[bool]:
    """Return True if the installed llama-cpp-python is a CUDA build,
    False if it is a CPU-only build, None if it is not installed.

    The version tag alone is not reliable (see installed_wheel_cuda_tag),
    so also check for the ggml-cuda.dll backend the CUDA wheels ship."""
    if installed_wheel_cuda_tag() is not None:
        return True
    dll_dirs = _llama_cpp_dll_dirs()
    if not dll_dirs:
        # Distinguish "not installed" from "installed, no DLL dirs found"
        try:
            from importlib.metadata import version
            version("llama_cpp_python")
        except Exception:
            return None
        return False
    try:
        return any((d / "ggml-cuda.dll").is_file() for d in dll_dirs)
    except OSError:
        return False


# DLL basenames the llama-cpp-python wheels ship. A same-named DLL that
# Windows resolves from ANOTHER directory (System32, the Python dir, or a
# PATH entry added by other AI apps such as Ollama / LM Studio / ComfyUI)
# shadows the wheel's copy, and a version mismatch then fails the engine
# load with WinError 127 "The specified procedure could not be found".
_SHADOWABLE_DLL_PREFIXES = ("ggml", "llama", "mtmd", "libomp")


def find_shadowing_dlls() -> list[tuple[str, str, str]]:
    """Scan DLL-resolution directories for llama.cpp-family DLLs that can
    shadow the ones shipped inside the llama-cpp-python wheel.

    Returns (dll_name, directory, location_kind) triples, deduplicated,
    wheel dirs excluded. location_kind ranks how dangerous a hit is:

      'app'    — the Python executable's directory (and the base
                 interpreter's dir when running through a venv launcher)
      'system' — System32 / the Windows directory
      'cwd'    — the process working directory
      'path'   — an ordinary PATH entry

    'app', 'system', and 'cwd' outrank the wheel's own lib/bin dirs in the
    legacy dependent-DLL search order the engine load uses, so a hit there
    genuinely shadows the wheel's copy. Plain 'path' hits normally CANNOT
    win (setup_cuda_dll_path prepends the wheel dirs to PATH) and are
    reported for completeness only. Already-loaded same-named modules are
    inherently invisible to a filesystem scan.
    """
    wheel_dirs = set()
    for d in _llama_cpp_dll_dirs():
        try:
            wheel_dirs.add(d.resolve())
        except OSError:
            continue
    wheel_dlls: set[str] = set()
    for d in wheel_dirs:
        try:
            wheel_dlls.update(p.name.lower() for p in d.glob("*.dll"))
        except OSError:
            continue
    if not wheel_dlls:
        wheel_dlls = {f"{prefix}.dll" for prefix in ("ggml", "ggml-base", "llama")}

    search_dirs: list[tuple[Path, str]] = [(Path(sys.executable).parent, "app")]
    base_exe = getattr(sys, "_base_executable", None)
    if base_exe:
        # Under a venv launcher (uv), the loader's application directory is
        # the base interpreter's dir, not the venv's Scripts dir.
        search_dirs.append((Path(base_exe).parent, "app"))
    if sys.platform == "win32":
        windir = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        search_dirs.append((windir / "System32", "system"))
        search_dirs.append((windir / "System", "system"))
        search_dirs.append((windir, "system"))
    try:
        search_dirs.append((Path.cwd(), "cwd"))
    except OSError:
        pass
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            search_dirs.append((Path(entry.strip()), "path"))

    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    checked: set[Path] = set()
    for directory, kind in search_dirs:
        # One unreadable entry (deny-ACL'd dir, dead share) must not
        # abort the scan and discard hits already collected.
        try:
            resolved = directory.resolve()
            if resolved in checked or resolved in wheel_dirs or not resolved.is_dir():
                continue
            checked.add(resolved)
            names = {p.name.lower() for p in resolved.glob("*.dll")}
        except OSError:
            continue
        for name in sorted(names & wheel_dlls):
            if not name.startswith(_SHADOWABLE_DLL_PREFIXES):
                continue
            key = (name, str(resolved))
            if key not in seen:
                seen.add(key)
                found.append((name, str(resolved), kind))
    return found


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

    # llama_cpp ships its own DLL dirs (lib/ on older wheels, lib/ AND bin/
    # on v0.3.40) — register them too so dependent DLLs resolve even when
    # the venv isn't on PATH.
    for dll_dir in _llama_cpp_dll_dirs():
        try:
            os.add_dll_directory(str(dll_dir))
        except OSError:
            pass
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

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


def dll_shadowing_message(shadowing: list[tuple[str, str, str]]) -> str:
    """User-facing remediation text for a WinError 127 engine-load failure.

    WinError 127 ('The specified procedure could not be found') means a DLL
    the engine depends on WAS found, but the loaded copy was missing a
    required function — an outdated Microsoft C/OpenMP runtime, or an
    incompatible same-named DLL loaded instead of the wheel's own copy.
    Takes the (name, directory, kind) triples from find_shadowing_dlls().
    """
    lines = [
        "The engine failed to load with WinError 127: Windows loaded an "
        "incompatible version of a DLL the engine needs — usually an "
        "outdated Microsoft Visual C++ runtime, or a same-named DLL from "
        "another install shadowing the engine's own copy.",
        "",
        "First, update the Microsoft Visual C++ runtime (most common fix):",
        "    winget install Microsoft.VCRedist.2015+.x64",
        "then reboot and run diagnose.bat again.",
    ]
    causal = [s for s in shadowing if s[2] != "path"]
    on_path = [s for s in shadowing if s[2] == "path"]
    if causal:
        lines.append("")
        lines.append(
            "These copies are searched BEFORE the app's own DLL folders and "
            "can shadow them:"
        )
        lines.extend(f"  - {name}  in  {directory}" for name, directory, _ in causal)
        lines.append("")
        lines.append(
            "If they are leftovers from another app, rename or remove them. "
            "For files inside the Windows directory (System32), do NOT "
            "delete — rename them (add .bak) from an administrator prompt "
            "so the change is reversible."
        )
    if on_path:
        lines.append("")
        lines.append(
            "Also found on PATH (less likely the cause — the app searches "
            "its own DLL folders first):"
        )
        lines.extend(f"  - {name}  in  {directory}" for name, directory, _ in on_path)
        lines.append("")
        lines.append(
            "These usually belong to another AI app (Ollama, LM Studio, "
            "ComfyUI, ...). If the error persists after the runtime update, "
            "try removing those directories from PATH."
        )
    return "\n".join(lines)


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
        "wheel_is_cuda_build": None,
        "recommended_tag": None,
        "tags_match": None,
        "shadowing_dlls": [],
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
    report["wheel_is_cuda_build"] = installed_wheel_is_cuda_build()
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
        if sys.platform == "win32" and not report["llama_cpp_importable"]:
            try:
                report["shadowing_dlls"] = find_shadowing_dlls()
            except Exception:
                pass

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

    combined_error = f"{error_text} {report.get('import_error') or ''}"
    if "WinError 127" in combined_error:
        shadowing = report.get("shadowing_dlls") or []
        if not shadowing:
            # diagnose() only scans when its own import fails; the engine
            # can still hit WinError 127 later (e.g. at model load).
            try:
                shadowing = find_shadowing_dlls()
            except Exception:
                shadowing = []
        return dll_shadowing_message(shadowing)

    return (
        "llama-cpp-python failed to initialize.\n\n"
        f"Error: {error_text}\n\n"
        f"Detected CUDA Toolkit: {report['toolkit_version']}\n"
        f"Installed wheel: {report['llama_cpp_version'] or 'unknown'}\n\n"
        "Run diagnose.bat for a full report, and include its output if you "
        "open a GitHub issue."
    )
