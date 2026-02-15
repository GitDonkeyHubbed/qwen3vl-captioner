"""
Auto-download utility for the Qwen3-VL mmproj (vision encoder) GGUF file.

On first run, if the mmproj file is not found next to the main model,
this module downloads it from HuggingFace Hub.
"""

import os
from pathlib import Path
from typing import Callable, Optional


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
    
    for f in model_dir.iterdir():
        name_lower = f.name.lower()
        if f.is_file() and f.suffix == ".gguf" and "mmproj" in name_lower:
            return f
    
    return None


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
                local_dir_use_symlinks=False,
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
