"""
Model Download Manager for VL-CAPTIONER Studio Pro.

Provides:
  - MODEL_REGISTRY: maps model combo display names to HF repo info
  - ModelDownloadWorker: QObject that downloads a GGUF in a background QThread
  - get_all_model_display_names(): returns ordered list of display names for combo
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtCore import QObject, pyqtSignal


# ---------------------------------------------------------------------------
# Registry: combo text -> download info
#
# All models from: prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF
# Each key is the human-readable dropdown label:
#   "Qwen3-VL 8B Abliterated — <QUANT> (<SIZE>)"
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Main model quants (smallest → largest) ──────────────────
    "Qwen3-VL 8B ABL — Q2_K (3.28 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q2_K.gguf",
        "size_gb": 3.28,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q3_K_S (3.77 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q3_K_S.gguf",
        "size_gb": 3.77,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q3_K_M (4.12 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q3_K_M.gguf",
        "size_gb": 4.12,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q3_K_L (4.43 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q3_K_L.gguf",
        "size_gb": 4.43,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — IQ4_XS (4.59 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.IQ4_XS.gguf",
        "size_gb": 4.59,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q4_K_S (4.80 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q4_K_S.gguf",
        "size_gb": 4.80,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q4_K_M (5.03 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q4_K_M.gguf",
        "size_gb": 5.03,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q5_K_S (5.72 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q5_K_S.gguf",
        "size_gb": 5.72,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q5_K_M (5.85 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q5_K_M.gguf",
        "size_gb": 5.85,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q6_K (6.73 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q6_K.gguf",
        "size_gb": 6.73,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — Q8_0 (8.71 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q8_0.gguf",
        "size_gb": 8.71,
        "gated": False,
    },
    "Qwen3-VL 8B ABL — F16 (16.4 GB)": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.f16.gguf",
        "size_gb": 16.4,
        "gated": False,
    },
}

# Ordered list for the dropdown — sorted by quant size ascending
_MODEL_ORDER = [
    "Qwen3-VL 8B ABL — Q2_K (3.28 GB)",
    "Qwen3-VL 8B ABL — Q3_K_S (3.77 GB)",
    "Qwen3-VL 8B ABL — Q3_K_M (4.12 GB)",
    "Qwen3-VL 8B ABL — Q3_K_L (4.43 GB)",
    "Qwen3-VL 8B ABL — IQ4_XS (4.59 GB)",
    "Qwen3-VL 8B ABL — Q4_K_S (4.80 GB)",
    "Qwen3-VL 8B ABL — Q4_K_M (5.03 GB)",
    "Qwen3-VL 8B ABL — Q5_K_S (5.72 GB)",
    "Qwen3-VL 8B ABL — Q5_K_M (5.85 GB)",
    "Qwen3-VL 8B ABL — Q6_K (6.73 GB)",
    "Qwen3-VL 8B ABL — Q8_0 (8.71 GB)",
    "Qwen3-VL 8B ABL — F16 (16.4 GB)",
]


def get_all_model_display_names() -> List[str]:
    """Return the ordered list of model display names for the dropdown combo."""
    return list(_MODEL_ORDER)


def get_model_info(combo_text: str) -> Optional[Dict[str, Any]]:
    """Return registry entry for *combo_text*, or None if not downloadable."""
    return MODEL_REGISTRY.get(combo_text)


def model_file_exists(model_dir: Path, filename: str) -> bool:
    """Check whether *filename* already exists in *model_dir*."""
    return (model_dir / filename).is_file()


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

class ModelDownloadWorker(QObject):
    """Downloads a single GGUF file from HuggingFace Hub.

    Signals
    -------
    progress(message, fraction)   0.0-1.0 progress updates
    finished(local_path)          download complete — passes the local file path
    error(message)                something went wrong
    """

    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)      # str(local_path)
    error = pyqtSignal(str)

    def __init__(
        self,
        repo_id: str,
        filename: str,
        target_dir: Path,
        hf_token: str = "",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.target_dir = target_dir
        self.hf_token = hf_token or None
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the download."""
        self._cancelled = True

    def run(self):
        """Execute the download (call from a QThread)."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self.error.emit(
                "huggingface-hub is not installed.\n"
                "Run: pip install huggingface-hub"
            )
            return

        if self._cancelled:
            self.error.emit("Download cancelled before starting.")
            return

        try:
            self.progress.emit(
                f"Downloading {self.filename} ...",
                0.05,
            )

            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=str(self.target_dir),
                local_dir_use_symlinks=False,
                token=self.hf_token,
            )

            if self._cancelled:
                self.error.emit("Download cancelled.")
                return

            self.progress.emit("Download complete", 1.0)
            self.finished.emit(str(local_path))

        except Exception as exc:
            if self._cancelled:
                self.error.emit("Download cancelled.")
                return
            msg = str(exc)
            # Surface auth errors clearly
            if "401" in msg or "403" in msg:
                msg = (
                    f"Authentication error ({msg[:120]}).\n\n"
                    "This model may require a HuggingFace token.\n"
                    "Add your token in Settings (gear icon) and try again."
                )
            self.error.emit(msg)
