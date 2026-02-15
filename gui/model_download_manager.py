"""
Model Download Manager for VL-CAPTIONER Studio Pro.

Provides:
  - MODEL_REGISTRY: maps model combo display names to HF repo info
  - ModelDownloadWorker: QObject that downloads a GGUF in a background QThread
"""

from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtCore import QObject, pyqtSignal


# ---------------------------------------------------------------------------
# Registry: combo text -> download info
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "QWEN 3 VL 8B_Q8": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q8_0.gguf",
        "size_gb": 8.5,
        "gated": False,
    },
    "QWEN 3 VL 8B_Q6": {
        "repo_id": "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
        "filename": "Qwen3-VL-8B-Instruct-abliterated-v1.Q6_K.gguf",
        "size_gb": 6.6,
        "gated": False,
    },
}


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

            self.progress.emit("Download complete", 1.0)
            self.finished.emit(str(local_path))

        except Exception as exc:
            msg = str(exc)
            # Surface auth errors clearly
            if "401" in msg or "403" in msg:
                msg = (
                    f"Authentication error ({msg[:120]}).\n\n"
                    "This model may require a HuggingFace token.\n"
                    "Add your token in Settings (gear icon) and try again."
                )
            self.error.emit(msg)
