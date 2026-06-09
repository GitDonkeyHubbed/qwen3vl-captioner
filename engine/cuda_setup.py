"""
Windows CUDA DLL path setup for llama-cpp-python.

Must run before any llama_cpp import. GPU drivers alone do not ship the
CUDA runtime DLLs (cudart, cublas, etc.) that ggml-cuda.dll needs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _cuda_install_roots() -> list[Path]:
    """Return CUDA toolkit install roots, newest version first."""
    roots: list[Path] = []

    cuda_env = os.environ.get("CUDA_PATH")
    if cuda_env:
        roots.append(Path(cuda_env))

    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.is_dir():
        version_dirs = sorted(
            (p for p in toolkit_root.iterdir() if p.is_dir() and p.name.startswith("v")),
            key=lambda p: p.name,
            reverse=True,
        )
        roots.extend(version_dirs)

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _cuda_bin_dirs() -> list[Path]:
    """Return existing CUDA bin directories to add to the DLL search path."""
    dirs: list[Path] = []
    for root in _cuda_install_roots():
        for sub in ("bin", os.path.join("bin", "x64")):
            path = root / sub
            if path.is_dir():
                dirs.append(path.resolve())
    return dirs


def setup_cuda_dll_path() -> Path | None:
    """
    Add CUDA and llama_cpp library directories to the DLL search path.

    Returns the first CUDA bin directory found, or None if no toolkit is installed.
    """
    if sys.platform != "win32":
        return None

    import ctypes

    for bin_dir in _cuda_bin_dirs():
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    try:
        import importlib.util

        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lib_dir = Path(spec.origin).resolve().parent / "lib"
            if lib_dir.is_dir():
                os.add_dll_directory(str(lib_dir))
                os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

    cuda_bins = _cuda_bin_dirs()
    if not cuda_bins:
        return None

    cuda_bin = cuda_bins[0]
    for dll_path in sorted(cuda_bin.glob("cudart64_*.dll")):
        try:
            ctypes.CDLL(str(dll_path), winmode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    for dll_name in (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
    ):
        dll_path = cuda_bin / dll_name
        if dll_path.exists():
            try:
                ctypes.CDLL(str(dll_path), winmode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    return cuda_bin


def cuda_toolkit_missing_message() -> str:
    """Return a user-facing message when CUDA runtime DLLs are unavailable."""
    return (
        "CUDA runtime DLLs were not found. The NVIDIA GPU driver is installed, but "
        "llama-cpp-python also needs the CUDA Toolkit (not just the driver).\n\n"
        "Install CUDA Toolkit, then re-run setup.bat:\n"
        "  winget install Nvidia.CUDA\n"
        "  or https://developer.nvidia.com/cuda-downloads\n\n"
        "Also ensure the llama-cpp-python wheel matches your CUDA version "
        "(re-run setup.bat after installing CUDA)."
    )
