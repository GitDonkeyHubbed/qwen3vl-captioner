"""
Auto-download utility for the Qwen3-VL mmproj (vision encoder) GGUF file.

On first run, if the mmproj file is not found next to the main model,
this module downloads it from HuggingFace Hub.
"""

import importlib.util
import os
from pathlib import Path
from typing import Callable, Optional

# Use HuggingFace's high-performance Xet transfer (Rust-based hf_xet client,
# shipped with huggingface_hub >= 1.0) when available — it speeds up
# hf_hub_download for Xet-backed repos. (The legacy HF_HUB_ENABLE_HF_TRANSFER
# flag is deprecated and ignored by current huggingface_hub.)
try:
    if importlib.util.find_spec("hf_xet") is not None:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
except Exception:
    pass


# Primary repo: the user's abliterated model (has matching mmproj files)
MMPROJ_REPO_ID = "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF"
MMPROJ_FILENAME = "Qwen3-VL-8B-Instruct-abliterated-v1.mmproj-f16.gguf"

# Fallback repos to try if the primary one fails
FALLBACK_REPOS = [
    (
        "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "Qwen3-VL-8B-Instruct-abliterated-v1.mmproj-Q8_0.gguf",
    ),
    (
        "bartowski/Qwen3-VL-8B-Instruct-GGUF",
        "Qwen3-VL-8B-Instruct-mmproj-f16.gguf",
    ),
    (
        "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    ),
]


def find_mmproj_file(model_dir: Path) -> Optional[Path]:
    """
    Search for an existing mmproj file in the given directory.
    Looks for files matching common mmproj naming patterns.
    
    Args:
        model_dir: Directory to search in.
        
    Returns:
        Path to the mmproj file if found, None otherwise.
    """
    if not model_dir.is_dir():
        return None

    # A model folder can legitimately hold more than one encoder (e.g. an f16
    # and a Q8_0 mmproj). iterdir() order is filesystem-dependent, so sort for
    # a deterministic pick and prefer the higher-quality f16 when present.
    matches = sorted(
        (
            f for f in model_dir.iterdir()
            if f.is_file() and f.suffix == ".gguf" and "mmproj" in f.name.lower()
        ),
        key=lambda f: (0 if "f16" in f.name.lower() else 1, f.name.lower()),
    )
    return matches[0] if matches else None


def download_mmproj(
    target_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """
    Download the mmproj vision encoder GGUF from HuggingFace Hub.
    
    Tries the primary abliterated model repo first, then fallbacks.
    
    Args:
        target_dir: Directory to save the downloaded file.
        progress_callback: Called with (message, progress_fraction) during download.
        
    Returns:
        Path to the downloaded mmproj file.
        
    Raises:
        RuntimeError: If download fails from all sources.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is not installed. Run:\n"
            "  pip install huggingface-hub"
        )
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Try primary repo first, then fallbacks
    attempts = [(MMPROJ_REPO_ID, MMPROJ_FILENAME)] + FALLBACK_REPOS
    
    for repo_id, filename in attempts:
        try:
            if progress_callback:
                progress_callback(f"Downloading {filename} from {repo_id}...", 0.1)
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
            )
            
            result_path = Path(downloaded_path)
            
            if progress_callback:
                progress_callback(f"Downloaded: {result_path.name}", 1.0)
            
            return result_path
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Failed from {repo_id}: {e}. Trying next...", 0.0)
            continue
    
    raise RuntimeError(
        "Could not download mmproj file from any known source.\n"
        "Please download it manually from HuggingFace and place it in:\n"
        f"  {target_dir}\n\n"
        "Recommended file:\n"
        "  https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF\n"
        "  -> Qwen3-VL-8B-Instruct-abliterated-v1.mmproj-f16.gguf"
    )


def download_named_mmproj(
    repo_id: str,
    filename: str,
    target_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Download a SPECIFIC mmproj (repo_id + filename).

    Unlike download_mmproj (which tries a default set of repos), this fetches
    exactly the encoder the caller names — i.e. the one that matches a given
    model — so the vision encoder can never be mispaired with the wrong model.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is not installed. Run: pip install huggingface-hub"
        )

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Downloading {filename} from {repo_id}...", 0.1)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    result = Path(downloaded_path)

    if progress_callback:
        progress_callback(f"Downloaded: {result.name}", 1.0)
    return result


def ensure_mmproj(
    model_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """
    Ensure the mmproj file exists. If not found, download it.
    
    This is the main entry point — call this before loading the model.
    
    Args:
        model_dir: Directory containing the main GGUF model.
        progress_callback: Optional progress callback.
        
    Returns:
        Path to the mmproj file (existing or newly downloaded).
    """
    existing = find_mmproj_file(model_dir)
    if existing:
        if progress_callback:
            progress_callback(f"Found existing mmproj: {existing.name}", 1.0)
        return existing
    
    if progress_callback:
        progress_callback("mmproj file not found. Downloading...", 0.0)
    
    return download_mmproj(model_dir, progress_callback)
