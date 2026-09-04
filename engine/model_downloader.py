"""
Auto-download utility for the Qwen3-VL mmproj (vision encoder) GGUF file.

On first run, if the mmproj file is not found next to the main model,
this module downloads it from HuggingFace Hub.
"""

import importlib.util
import os
import re
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


class MmprojMismatchError(RuntimeError):
    """Raised when no vision encoder can be paired with a model with confidence.

    Loading a model with another model's mmproj does not fail cleanly — it
    crashes llama.cpp natively on the first caption — so refusing is always
    better than guessing.
    """


# Quantisation / precision tokens that appear in GGUF filenames. They identify
# a *build* of a model, not the model, so they are stripped before comparing a
# model to a candidate encoder.
_QUANT_RE = re.compile(
    r"^(?:iq\d+(?:_[a-z0-9]+)*|q\d+(?:_[a-z0-9]+)*|f\d+|bf\d+|fp\d+)$", re.I
)
_SIZE_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)b$", re.I)


def _tokens(name: str) -> list[str]:
    """Split a filename stem into comparable lowercase tokens."""
    return [t for t in re.split(r"[-_. ]+", name.lower()) if t]


def _model_key(stem: str) -> str:
    """Normalised identity of a model, with quant and mmproj tokens removed."""
    keep = [t for t in _tokens(stem) if t != "mmproj" and not _QUANT_RE.match(t)]
    return "".join(keep)


def _size_tokens(name: str) -> set[str]:
    """Parameter-count tokens in a filename, e.g. {"8"} for ...-8B-...

    Matched per token rather than by scanning the whole string, so the "3.5"
    in "Qwen3.5-2B" cannot be misread as a 352B size.
    """
    return {
        m.group(1) for t in _tokens(name) if (m := _SIZE_TOKEN_RE.match(t))
    }


def _is_generic_mmproj(path: Path) -> bool:
    """True for an encoder named only "mmproj" plus a precision tag.

    Several publishers ship the encoder as a bare `mmproj-F16.gguf` next to
    the model. Such a file carries no model identity of its own, so it can be
    paired only when the folder leaves no doubt which model it belongs to.
    """
    rest = [
        t for t in _tokens(path.stem)
        if t != "mmproj" and not _QUANT_RE.match(t)
    ]
    return not rest


def _mmproj_candidates(model_dir: Path) -> list[Path]:
    """All mmproj GGUFs in a directory, best quality first.

    A model folder can legitimately hold more than one encoder (e.g. an f16
    and a Q8_0 mmproj). iterdir() order is filesystem-dependent, so sort for a
    deterministic pick and prefer the higher-quality f16 when present.
    """
    if not model_dir.is_dir():
        return []
    return sorted(
        (
            f for f in model_dir.iterdir()
            if f.is_file() and f.suffix == ".gguf" and "mmproj" in f.name.lower()
        ),
        key=lambda f: (0 if "f16" in f.name.lower() else 1, f.name.lower()),
    )


def _model_files(model_dir: Path) -> list[Path]:
    """Non-mmproj GGUF model files in a directory."""
    if not model_dir.is_dir():
        return []
    return [
        f for f in model_dir.iterdir()
        if f.is_file() and f.suffix == ".gguf" and "mmproj" not in f.name.lower()
    ]


def find_mmproj_file(
    model_dir: Path, model_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Search for the vision encoder belonging to a model.

    When *model_path* is given the pairing is model-aware: an encoder is
    accepted only when it names the same model, or when it is a bare
    `mmproj-*.gguf` in a folder holding exactly one model. Taking any
    `*mmproj*.gguf` in the folder — the previous behaviour — silently paired a
    browsed model with a foreign encoder, which crashes llama.cpp natively on
    the first caption.

    Args:
        model_dir: Directory to search in.
        model_path: The model the encoder must pair with. Omitting it keeps
            the old "any encoder here" behaviour, for callers that only ask
            whether the folder holds an encoder at all.

    Returns:
        Path to the mmproj file if one can be paired with confidence, else None.
    """
    candidates = _mmproj_candidates(model_dir)
    if not candidates or model_path is None:
        return candidates[0] if candidates else None

    key = _model_key(Path(model_path).stem)
    model_sizes = _size_tokens(Path(model_path).name)

    # 1. An encoder that names this model (the usual publisher layout,
    #    "<model stem>.mmproj-f16.gguf").
    for cand in candidates:
        if key and key in _model_key(cand.stem):
            return cand

    # 2. A bare "mmproj-F16.gguf" — only trustworthy when the folder holds a
    #    single model, otherwise it is anyone's encoder.
    if len(_model_files(model_dir)) == 1:
        for cand in candidates:
            if _is_generic_mmproj(cand):
                return cand

    # 3. Same family and size, differing only in build tokens.
    for cand in candidates:
        cand_sizes = _size_tokens(cand.name)
        if cand_sizes and model_sizes and cand_sizes == model_sizes:
            return cand

    # Everything left names a different model — refuse rather than guess.
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
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is not installed. Run:\n"
            "  pip install huggingface-hub"
        ) from exc
    
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
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is not installed. Run: pip install huggingface-hub"
        ) from exc

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


# What the built-in fallback download actually provides. Fetching it for any
# other model produces a mismatched pairing, which crashes llama.cpp natively.
DEFAULT_MMPROJ_FAMILY = "qwen3vl"
DEFAULT_MMPROJ_SIZE = "8"


def default_mmproj_fits(model_path: Optional[Path]) -> bool:
    """True when the built-in Qwen3-VL 8B encoder can serve *model_path*."""
    if model_path is None:
        return True  # nothing known about the model — caller decides
    name = Path(model_path).name
    sizes = _size_tokens(name)
    if sizes and DEFAULT_MMPROJ_SIZE not in sizes:
        return False
    return DEFAULT_MMPROJ_FAMILY in _model_key(Path(name).stem)


def ensure_mmproj(
    model_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    model_path: Optional[Path] = None,
) -> Path:
    """
    Ensure the mmproj file exists. If not found, download it.

    This is the main entry point — call this before loading the model.

    Args:
        model_dir: Directory containing the main GGUF model.
        progress_callback: Optional progress callback.
        model_path: The model the encoder must pair with. When given, an
            encoder in the folder is accepted only if it belongs to this
            model, and the built-in Qwen3-VL 8B download is used only if it
            actually fits — a non-8B or non-Qwen3-VL model raises instead of
            being handed an encoder that crashes on the first caption.

    Returns:
        Path to the mmproj file (existing or newly downloaded).

    Raises:
        MmprojMismatchError: no encoder can be paired with confidence.
    """
    existing = find_mmproj_file(model_dir, model_path)
    if existing:
        if progress_callback:
            progress_callback(f"Found existing mmproj: {existing.name}", 1.0)
        return existing

    if not default_mmproj_fits(model_path):
        present = _mmproj_candidates(model_dir)
        note = (
            f"\n\nThe folder does contain {', '.join(f.name for f in present)}, "
            "but that encoder belongs to a different model."
            if present else ""
        )
        raise MmprojMismatchError(
            f"No vision encoder (mmproj) matching {Path(model_path).name} was "
            f"found in:\n  {model_dir}\n\n"
            "The built-in download only provides the Qwen3-VL 8B encoder, "
            "which this model cannot use.\n\n"
            "Download the mmproj published alongside your model and put it in "
            f"that folder.{note}"
        )

    if progress_callback:
        progress_callback("mmproj file not found. Downloading...", 0.0)

    return download_mmproj(model_dir, progress_callback)
