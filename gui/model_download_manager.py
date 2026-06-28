"""
Model Download Manager for VL-CAPTIONER Studio Pro.

Provides:
  - MODEL_REGISTRY: maps model combo display names to HF repo info
  - ModelDownloadWorker: QObject that downloads a GGUF in a background QThread
  - get_all_model_display_names(): returns ordered list of display names for combo
"""

import concurrent.futures
import importlib.util
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtCore import QObject, pyqtSignal

# Opt into HuggingFace's high-performance Xet transfer (Rust-based, via the
# hf_xet client that ships with huggingface_hub >= 1.0). This accelerates the
# library-driven downloads — mmproj via hf_hub_download and MLX folders via
# snapshot_download — for Xet-backed repos. The main GGUF model uses our own
# parallel range downloader below, which is also cancellable.
# (Note: the older HF_HUB_ENABLE_HF_TRANSFER / hf_transfer flag is deprecated
# and ignored by current huggingface_hub, so we do not set it.)
try:
    XET_HIGH_PERF = importlib.util.find_spec("hf_xet") is not None
except Exception:
    XET_HIGH_PERF = False
if XET_HIGH_PERF:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


# ---------------------------------------------------------------------------
# Registry: combo text -> download info
#
# All models from: prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF
# Each key is the human-readable dropdown label:
#   "Qwen3-VL 8B Abliterated — <QUANT> (<SIZE>)"
# ---------------------------------------------------------------------------

def _gguf_family(
    label: str, repo_id: str, stem: str, mmproj: str,
    quants: Dict[str, float], recommended: bool = False,
    quant_template: str = "{stem}.{quant}.gguf",
) -> Dict[str, Dict[str, Any]]:
    """Build registry entries for one GGUF model family."""
    entries: Dict[str, Dict[str, Any]] = {}
    for quant, size_gb in quants.items():
        entries[f"{label} — {quant} ({size_gb:.2f} GB)"] = {
            "repo_id": repo_id,
            "filename": quant_template.format(stem=stem, quant=quant),
            "mmproj_filename": mmproj,
            "size_gb": size_gb,
            "gated": False,
            "recommended": recommended,
        }
    return entries


# ── Recommended default: prithivMLmods abliterated v2 (Dec 2025) ─────
_V2 = _gguf_family(
    "Qwen3-VL 8B ABL v2",
    "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v2-GGUF",
    "Qwen3-VL-8B-Instruct-abliterated-v2",
    "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf",
    {
        "Q2_K": 3.06, "Q3_K_S": 3.51, "Q3_K_M": 3.84, "Q3_K_L": 4.13,
        "IQ4_XS": 4.28, "Q4_K_S": 4.47, "Q4_K_M": 4.68, "Q5_K_S": 5.33,
        "Q5_K_M": 5.45, "Q6_K": 6.26, "Q8_0": 8.11, "f16": 15.26,
    },
    recommended=True,
)

# ── Captioning-specialized abliterated fine-tune ──────────────────────
_CAPTION_IT = _gguf_family(
    "Qwen3-VL 8B Caption-it",
    "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it-GGUF",
    "Qwen3-VL-8B-Abliterated-Caption-it",
    "Qwen3-VL-8B-Abliterated-Caption-it.mmproj-f16.gguf",
    {"Q4_K_M": 4.68, "Q6_K": 6.26, "Q8_0": 8.11},
)

# ── huihui-ai abliteration (quantized by noctrex) ─────────────────────
_HUIHUI = _gguf_family(
    "Huihui Qwen3-VL 8B ABL",
    "noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
    "Huihui-Qwen3-VL-8B-Instruct-abliterated",
    "mmproj-F16.gguf",
    {"Q4_K_M": 4.68, "Q6_K": 6.26, "Q8_0": 8.11},
    quant_template="{stem}-{quant}.gguf",
)

# ── Legacy v1 (kept for existing installs; other quants still load
#    via the local-file scan if already on disk) ──────────────────────
_V1 = _gguf_family(
    "Qwen3-VL 8B ABL",
    "prithivMLmods/Qwen3-VL-8B-Instruct-abliterated-v1-GGUF",
    "Qwen3-VL-8B-Instruct-abliterated-v1",
    "Qwen3-VL-8B-Instruct-abliterated-v1.mmproj-f16.gguf",
    {"Q4_K_M": 5.03, "Q6_K": 6.73, "Q8_0": 8.71},
)
# Preserve the original v1 display names (with their original size text)
_V1 = {
    "Qwen3-VL 8B ABL — Q4_K_M (5.03 GB)": _V1["Qwen3-VL 8B ABL — Q4_K_M (5.03 GB)"],
    "Qwen3-VL 8B ABL — Q6_K (6.73 GB)": _V1["Qwen3-VL 8B ABL — Q6_K (6.73 GB)"],
    "Qwen3-VL 8B ABL — Q8_0 (8.71 GB)": _V1["Qwen3-VL 8B ABL — Q8_0 (8.71 GB)"],
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {**_V2, **_CAPTION_IT, **_HUIHUI, **_V1}

# Dropdown groups (separator drawn between groups)
_GGUF_GROUPS: List[List[str]] = [
    list(_V2.keys()),
    list(_CAPTION_IT.keys()),
    list(_HUIHUI.keys()),
    list(_V1.keys()),
]

# Flat ordered list (kept for compatibility)
_MODEL_ORDER = [name for group in _GGUF_GROUPS for name in group]

# ---------------------------------------------------------------------------
# MLX models (Apple Silicon only) — folders of safetensors, no mmproj needed.
# Note: these are the standard (non-abliterated) Qwen3-VL Instruct weights;
# no MLX conversion of the abliterated variant has been published yet.
# ---------------------------------------------------------------------------

# Abliterated v2 — the app's default GGUF model, converted to MLX and published
# by the project (the only MLX build of this model). Matches the recommended
# GGUF default so Mac and Windows users run the same model.
_MLX_V2 = {
    "Qwen3-VL 8B ABL v2 MLX — 4bit (5.4 GB)": {
        "repo_id": "LethalDonkey/Qwen3-VL-8B-Instruct-abliterated-v2-MLX-4bit",
        "folder": "Qwen3-VL-8B-Instruct-abliterated-v2-MLX-4bit",
        "size_gb": 5.40,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
    "Qwen3-VL 8B ABL v2 MLX — 6bit (7.4 GB)": {
        "repo_id": "LethalDonkey/Qwen3-VL-8B-Instruct-abliterated-v2-MLX-6bit",
        "folder": "Qwen3-VL-8B-Instruct-abliterated-v2-MLX-6bit",
        "size_gb": 7.40,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
    "Qwen3-VL 8B ABL v2 MLX — 8bit (9.2 GB)": {
        "repo_id": "LethalDonkey/Qwen3-VL-8B-Instruct-abliterated-v2-MLX-8bit",
        "folder": "Qwen3-VL-8B-Instruct-abliterated-v2-MLX-8bit",
        "size_gb": 9.20,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
}

def _mlx_quants(label, prefix, sizes, recommended=False):
    """Build 4/6/8-bit MLX registry entries for one LethalDonkey-published model."""
    out = {}
    for q in (4, 6, 8):
        sz = sizes[q]
        out[f"{label} MLX — {q}bit ({sz:.1f} GB)"] = {
            "repo_id": f"LethalDonkey/{prefix}-{q}bit",
            "folder": f"{prefix}-{q}bit",
            "size_gb": sz,
            "gated": False,
            "backend": "mlx",
            "recommended": recommended,
        }
    return out


# Gliese caption family (Qwen3.5, abliterated, captioning-specialized) —
# converted to MLX and published by the project across sizes (the only MLX
# builds of these). The 4B is marked recommended (the captioning sweet spot).
_MLX_GLIESE = {
    **_mlx_quants("Gliese Caption 0.8B", "Gliese-Qwen3.5-0.8B-Abliterated-Caption-MLX", {4: 0.6, 6: 0.8, 8: 1.0}),
    **_mlx_quants("Gliese Caption 2B", "Gliese-Qwen3.5-2B-Abliterated-Caption-MLX", {4: 1.7, 6: 2.1, 8: 2.6}),
    **_mlx_quants("Gliese Caption 4B", "Gliese-Qwen3.5-4B-Abliterated-Caption-MLX", {4: 2.9, 6: 3.9, 8: 4.9}, recommended=True),
    **_mlx_quants("Gliese Caption 9B", "Gliese-Qwen3.5-9B-Abliterated-Caption-MLX", {4: 5.6, 6: 7.7, 8: 9.8}),
}

# Newest Qwen3-VL c_abliterated builds, converted to MLX and published.
_MLX_CABL = {
    **_mlx_quants("Qwen3-VL 8B c_abl v3", "Qwen3-VL-8B-Instruct-c_abliterated-v3-MLX", {4: 5.4, 6: 7.4, 8: 9.2}),
    **_mlx_quants("Qwen3-VL 4B c_abl v2", "Qwen3-VL-4B-Instruct-c_abliterated-v2-MLX", {4: 2.9, 6: 3.9, 8: 4.8}),
}


# Abliterated (huihui-ai weights, converted by alexgusevski) + standard
_MLX_ABL = {
    "Qwen3-VL 8B MLX ABL — 4bit (5.4 GB)": {
        "repo_id": "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx",
        "folder": "Huihui-Qwen3-VL-8B-Instruct-abliterated-q4-mlx",
        "size_gb": 5.38,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
    "Qwen3-VL 8B MLX ABL — 6bit (7.3 GB)": {
        "repo_id": "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q6-mlx",
        "folder": "Huihui-Qwen3-VL-8B-Instruct-abliterated-q6-mlx",
        "size_gb": 7.29,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
    "Qwen3-VL 8B MLX ABL — 8bit (9.2 GB)": {
        "repo_id": "alexgusevski/Huihui-Qwen3-VL-8B-Instruct-abliterated-q8-mlx",
        "folder": "Huihui-Qwen3-VL-8B-Instruct-abliterated-q8-mlx",
        "size_gb": 9.19,
        "gated": False,
        "backend": "mlx",
        "recommended": True,
    },
}

_MLX_STD = {
    "Qwen3-VL 8B MLX — 4bit (5.4 GB)": {
        "repo_id": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
        "folder": "Qwen3-VL-8B-Instruct-MLX-4bit",
        "size_gb": 5.38,
        "gated": False,
        "backend": "mlx",
    },
    "Qwen3-VL 8B MLX — 6bit (7.3 GB)": {
        "repo_id": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-6bit",
        "folder": "Qwen3-VL-8B-Instruct-MLX-6bit",
        "size_gb": 7.29,
        "gated": False,
        "backend": "mlx",
    },
    "Qwen3-VL 8B MLX — 8bit (9.2 GB)": {
        "repo_id": "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit",
        "folder": "Qwen3-VL-8B-Instruct-MLX-8bit",
        "size_gb": 9.19,
        "gated": False,
        "backend": "mlx",
    },
}

# Next-gen Qwen3.5 abliterated VLM (experimental; needs mlx-vlm >= 0.6)
_MLX_NEXTGEN = {
    "Qwen3.5 4B ABL VLM MLX — 6bit (3.8 GB)": {
        "repo_id": "monyschuk/Qwen3.5-4B-Claude-Opus-abliterated-VLM-MLX",
        "folder": "Qwen3.5-4B-Claude-Opus-abliterated-VLM-MLX",
        "size_gb": 3.82,
        "gated": False,
        "backend": "mlx",
    },
}

MLX_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    **_MLX_V2, **_MLX_GLIESE, **_MLX_CABL, **_MLX_ABL, **_MLX_STD, **_MLX_NEXTGEN,
}

_MLX_GROUPS: List[List[str]] = [
    list(_MLX_V2.keys()),
    list(_MLX_GLIESE.keys()),
    list(_MLX_CABL.keys()),
    list(_MLX_ABL.keys()),
    list(_MLX_STD.keys()),
    list(_MLX_NEXTGEN.keys()),
]
_MLX_MODEL_ORDER = list(MLX_MODEL_REGISTRY.keys())


def mlx_backend_supported() -> bool:
    """MLX models only make sense on Apple Silicon Macs."""
    import platform
    import sys
    return sys.platform == "darwin" and platform.machine() == "arm64"


def get_all_model_display_names() -> List[str]:
    """Return the ordered list of model display names for the dropdown combo.

    MLX entries are only offered on Apple Silicon, where they can run."""
    names = list(_MODEL_ORDER)
    if mlx_backend_supported():
        names.extend(_MLX_MODEL_ORDER)
    return names


# Human-readable headers shown above each dropdown group (parallel to the
# group order returned by get_model_groups()).
_GGUF_GROUP_LABELS = [
    "★ Recommended — Abliterated v2",
    "Captioning-tuned",
    "Huihui abliterated",
    "Legacy v1",
]
_MLX_GROUP_LABELS = [
    "★ MLX · Abliterated v2 (Apple Silicon)",
    "★ MLX · Gliese Caption (Qwen3.5)",
    "MLX · Qwen3-VL c_abliterated (newest)",
    "MLX · Abliterated (huihui)",
    "MLX · Standard",
    "MLX · Qwen3.5 (experimental)",
]


def get_model_groups() -> List[List[str]]:
    """Return display names grouped for the dropdown (separator between
    groups): v2 (recommended), Caption-it, Huihui, legacy v1, then MLX
    groups on Apple Silicon."""
    groups = [list(g) for g in _GGUF_GROUPS]
    if mlx_backend_supported():
        groups.extend(list(g) for g in _MLX_GROUPS)
    return groups


def get_model_group_labels() -> List[str]:
    """Return a header label for each group from get_model_groups(), in the
    same order."""
    labels = list(_GGUF_GROUP_LABELS)
    if mlx_backend_supported():
        labels.extend(_MLX_GROUP_LABELS)
    return labels


def get_model_info(combo_text: str) -> Optional[Dict[str, Any]]:
    """Return registry entry for *combo_text*, or None if not downloadable.

    Entries carry a "backend" key: "gguf" (default) or "mlx"."""
    info = MODEL_REGISTRY.get(combo_text)
    if info is not None:
        return {**info, "backend": "gguf"}
    return MLX_MODEL_REGISTRY.get(combo_text)


def model_file_exists(model_dir: Path, filename: str) -> bool:
    """Check whether *filename* already exists in *model_dir*."""
    return (model_dir / filename).is_file()


def mlx_model_exists(model_dir: Path, folder: str) -> bool:
    """Check whether an MLX model folder is present and complete-looking."""
    path = model_dir / folder
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when HF redirects to its CDN.

    Storage backends (S3 etc.) reject requests that carry both their signed
    URL parameters and a Bearer token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc
            new_host = urllib.parse.urlparse(newurl).netloc
            if old_host != new_host:
                new.headers.pop("Authorization", None)
        return new


class ModelDownloadWorker(QObject):
    """Downloads a single GGUF file from HuggingFace with streaming progress.

    Downloads to a .part file (resumable via HTTP Range) and renames it on
    completion, so a cancelled or interrupted download picks up where it
    left off the next time.

    Signals
    -------
    progress(message, fraction)   0.0-1.0 progress updates
    finished(local_path)          download complete — passes the local file path
    error(message)                something went wrong (or user cancelled)
    """

    progress = pyqtSignal(str, float)
    finished = pyqtSignal(str)      # str(local_path)
    error = pyqtSignal(str)

    _CHUNK_SIZE = 1024 * 1024  # 1 MiB reads — keeps cancel latency low
    # Files at or above this size use the parallel multi-connection downloader;
    # smaller ones use a single stream (parallel overhead isn't worth it).
    _PARALLEL_THRESHOLD = 32 * 1024 * 1024  # 32 MiB
    _DEFAULT_CONNECTIONS = 8
    _SOCKET_TIMEOUT = 60   # seconds; per-connection read timeout
    _MAX_RETRIES = 4       # per-segment retries with no progress before failing

    def __init__(
        self,
        repo_id: str,
        filename: str,
        target_dir: Path,
        hf_token: str = "",
        snapshot_folder: Optional[str] = None,
        max_connections: int = _DEFAULT_CONNECTIONS,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.target_dir = target_dir
        self.hf_token = hf_token or None
        # When set, download the whole repo into target_dir/snapshot_folder
        # (used for MLX models, which are folders rather than single files)
        self.snapshot_folder = snapshot_folder
        # Number of parallel HTTP connections for large single-file downloads.
        self.max_connections = max(1, int(max_connections))
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the download.

        For the GGUF range download this takes effect within ~1 MiB per
        connection; for an MLX folder (snapshot) download it takes effect
        between files (a single large shard cannot be interrupted mid-file).
        A user-cancelled download discards its partial file/folder so a
        different model can be chosen; an interrupted single-stream GGUF
        download keeps its .part for resume.
        """
        self._cancelled = True

    def _run_snapshot(self):
        """Download a whole model repo (MLX folder) file-by-file, so the Stop
        button takes effect between files and real progress is reported (the
        old single blocking snapshot_download ignored cancel and showed none)."""
        from huggingface_hub import HfApi, hf_hub_download

        dest = Path(self.target_dir) / self.snapshot_folder
        # Don't delete an already-complete folder if the user cancels a re-verify
        already_complete = (
            dest.is_dir()
            and (dest / "config.json").is_file()
            and any(dest.glob("*.safetensors"))
        )

        def _discard_partial():
            if not already_complete:
                shutil.rmtree(dest, ignore_errors=True)

        try:
            if self._cancelled:
                self.error.emit("Download cancelled.")
                return
            self.progress.emit(f"Listing {self.repo_id} …", 0.0)
            files = HfApi().list_repo_files(self.repo_id, token=self.hf_token)
            n = len(files) or 1
            for i, fname in enumerate(files):
                if self._cancelled:
                    _discard_partial()
                    self.error.emit("Download cancelled.")
                    return
                self.progress.emit(
                    f"{self.snapshot_folder}/ — {fname} ({i + 1}/{n})",
                    i / n,
                )
                hf_hub_download(
                    self.repo_id, fname,
                    local_dir=str(dest), token=self.hf_token,
                )
            if self._cancelled:
                _discard_partial()
                self.error.emit("Download cancelled.")
                return
            self.progress.emit("Download complete", 1.0)
            self.finished.emit(str(dest))
        except Exception as exc:
            self.error.emit(self._format_error(exc))

    def run(self):
        """Execute the download (call from a QThread)."""
        try:
            from huggingface_hub import hf_hub_url
        except ImportError:
            self.error.emit(
                "huggingface-hub is not installed.\n"
                "Run: pip install huggingface-hub"
            )
            return

        if self._cancelled:
            self.error.emit("Download cancelled before starting.")
            return

        if self.snapshot_folder:
            self._run_snapshot()
            return

        target = Path(self.target_dir) / self.filename
        part = target.with_name(target.name + ".part")

        # A parallel pre-allocation killed mid-flight leaves a full-size but
        # hole-filled .part flagged by a .parallel marker. Size alone can't tell
        # it apart from a complete single-stream .part, so discard both and
        # start fresh rather than resume/finalize a corrupt file.
        parallel_marker = part.with_name(part.name + ".parallel")
        if parallel_marker.exists():
            self._safe_unlink(part)
            self._safe_unlink(parallel_marker)

        try:
            url = hf_hub_url(repo_id=self.repo_id, filename=self.filename)
        except Exception as exc:
            self.error.emit(self._format_error(exc))
            return

        # Pick the fastest safe strategy. HuggingFace's CDN throttles each
        # connection (~25-30 MB/s), so large files download many times faster
        # over several parallel range requests. Fall back to a single resumable
        # stream for small files, servers without Range support, or when a
        # resumable .part already exists.
        total, supports_range = self._probe(url)

        # Pre-flight disk-space check so a multi-GB download fails fast with a
        # clear message instead of dying mid-write — the parallel path even
        # pre-allocates the full size, which can sparse-succeed and only fail
        # later as opaque write errors.
        if total:
            probe_dir = self.target_dir
            while not probe_dir.exists() and probe_dir != probe_dir.parent:
                probe_dir = probe_dir.parent
            try:
                free = shutil.disk_usage(probe_dir).free
            except Exception:
                free = None
            # Only the not-yet-downloaded bytes need to fit: a resumable .part
            # already on disk counts toward the file, so don't demand room for a
            # second full copy (which would block a nearly-complete resume).
            existing_part = part.stat().st_size if part.exists() else 0
            needed = max(total - existing_part, 0) + 256 * 1024 * 1024  # +256 MiB headroom
            if free is not None and free < needed:
                self.error.emit(
                    f"Not enough free disk space for {self.filename}: need "
                    f"~{needed / 1024**3:.1f} GB, but only "
                    f"{free / 1024**3:.1f} GB free at {probe_dir}."
                )
                return

        use_parallel = (
            supports_range
            and total >= self._PARALLEL_THRESHOLD
            and self.max_connections > 1
            and not part.exists()
        )
        if use_parallel:
            self._run_parallel(url, target, part, total)
        else:
            self._run_single(url, target, part)

    # -- helpers ---------------------------------------------------------

    def _base_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "qwen3vl-captioner"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    @staticmethod
    def _format_error(exc: Exception) -> str:
        msg = str(exc)
        if "401" in msg or "403" in msg:
            msg = (
                f"Authentication error ({msg[:120]}).\n\n"
                "This model may require a HuggingFace token.\n"
                "Add your token in Settings (gear icon) and try again."
            )
        return msg

    @staticmethod
    def _safe_unlink(path: Path):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _probe(self, url: str):
        """Return (total_bytes, supports_range) for *url* (0/False on failure)."""
        headers = self._base_headers()
        headers["Range"] = "bytes=0-0"
        opener = urllib.request.build_opener(_StripAuthOnRedirect)
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=30) as resp:
                if resp.status == 206:
                    content_range = resp.headers.get("Content-Range", "")
                    total = (
                        int(content_range.rsplit("/", 1)[-1])
                        if "/" in content_range else 0
                    )
                    return total, True
                total = int(resp.headers.get("Content-Length") or 0)
                return total, False
        except Exception:
            return 0, False

    def _run_parallel(self, url, target: Path, part: Path, total: int):
        """Download *url* into *part* over several parallel range connections."""
        conns = max(2, min(self.max_connections, 16))
        # Marker: flags that this .part is a full-size but hole-filled parallel
        # pre-allocation. It is removed only on clean finalize. If the process
        # is killed mid-download, the marker survives and run() discards the
        # corrupt .part instead of resuming/finalizing it.
        marker = part.with_name(part.name + ".parallel")
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text("1")
            with open(part, "wb") as f:
                f.truncate(total)  # pre-allocate so each thread can seek+write
        except Exception as exc:
            self._safe_unlink(marker)
            self.error.emit(f"Could not create download file: {exc}")
            return

        seg = total // conns
        ranges = []
        for i in range(conns):
            start = i * seg
            end = (total - 1) if i == conns - 1 else (start + seg - 1)
            ranges.append((start, end))

        lock = threading.Lock()
        state = {"done": 0, "err": None}

        def fetch(start: int, end: int):
            """Download one byte range, retrying transient stalls/timeouts.

            Resumes mid-segment after a failure — one flaky connection out of
            several shouldn't doom a multi-GB download — and only gives up after
            _MAX_RETRIES consecutive attempts that make no forward progress.
            """
            seg_total = end - start + 1
            seg_done = 0
            attempts = 0
            while seg_done < seg_total and not self._cancelled:
                resume_at = start + seg_done
                headers = self._base_headers()
                headers["Range"] = f"bytes={resume_at}-{end}"
                opener = urllib.request.build_opener(_StripAuthOnRedirect)
                request = urllib.request.Request(url, headers=headers)
                before = seg_done
                last_exc = None
                try:
                    with opener.open(request, timeout=self._SOCKET_TIMEOUT) as resp, \
                            open(part, "r+b") as f:
                        # Every segment request carries a Range header, so a
                        # compliant server MUST answer 206. A 200 means Range
                        # was ignored and the body is the WHOLE file — writing
                        # it at this segment's offset would overrun into other
                        # segments. Treat as a transient failure so the retry
                        # loop re-requests (and ultimately fails loudly).
                        if resp.status != 206:
                            raise IOError(
                                f"server ignored Range for segment {start}-{end} "
                                f"(status {resp.status})"
                            )
                        f.seek(resume_at)
                        remaining = seg_total - seg_done  # never write past this segment
                        while remaining > 0:
                            if self._cancelled:
                                return
                            chunk = resp.read(min(self._CHUNK_SIZE, remaining))
                            if not chunk:
                                break
                            f.write(chunk)
                            n = len(chunk)
                            seg_done += n
                            remaining -= n
                            with lock:
                                state["done"] += n
                except Exception as exc:
                    if self._cancelled:
                        return
                    last_exc = exc

                if seg_done >= seg_total:
                    return  # segment complete
                if seg_done > before:
                    attempts = 0  # made progress this round — refresh retry budget
                else:
                    attempts += 1
                    if attempts > self._MAX_RETRIES:
                        with lock:
                            if state["err"] is None:
                                state["err"] = last_exc or TimeoutError(
                                    "Download stalled (no progress after retries)."
                                )
                        self._cancelled = True  # stop the other connections
                        return
                if not self._cancelled:
                    time.sleep(min(2 ** attempts, 8))  # backoff before retrying

        self.progress.emit(
            f"{self.filename} — {total / 1024**3:.2f} GB via {conns} connections...",
            0.0,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=conns) as pool:
            futures = [pool.submit(fetch, s, e) for s, e in ranges]
            last_emit = 0.0
            while any(not fut.done() for fut in futures):
                now = time.monotonic()
                if now - last_emit >= 0.5:
                    last_emit = now
                    with lock:
                        done = state["done"]
                    fraction = done / total if total else 0.0
                    self.progress.emit(
                        f"{self.filename} — {done / 1024**3:.2f} / "
                        f"{total / 1024**3:.2f} GB ({fraction * 100:.0f}%, "
                        f"{conns} connections)",
                        fraction,
                    )
                time.sleep(0.1)

        # A parallel .part is written at offsets, so any interruption leaves it
        # unusable for resume — always discard it on cancel or error.
        if self._cancelled:
            self._safe_unlink(part)
            self._safe_unlink(marker)
            if state["err"] is not None:
                self.error.emit(self._format_error(state["err"]))
            else:
                self.error.emit("Download cancelled.")
            return

        with lock:
            done = state["done"]
        if done < total:
            self._safe_unlink(part)
            self._safe_unlink(marker)
            self.error.emit(
                f"Download incomplete ({done / 1024**3:.2f} of "
                f"{total / 1024**3:.2f} GB) — connection dropped. Try again."
            )
            return

        try:
            part.replace(target)
        except Exception as exc:
            self.error.emit(f"Could not finalize download: {exc}")
            return
        self._safe_unlink(marker)  # finalized cleanly — drop the hole-filled flag
        self.progress.emit("Download complete", 1.0)
        self.finished.emit(str(target))

    @staticmethod
    def _part_meta_path(part: Path) -> Path:
        return part.with_name(part.name + ".meta")

    def _write_part_identity(self, part: Path, total: int):
        """Record which file this .part belongs to, so a later resume can tell
        it apart from a stale .part left under the same name by a different
        download (e.g. a different quant, or a renamed registry stem)."""
        try:
            self._part_meta_path(part).write_text(
                json.dumps({
                    "repo_id": self.repo_id,
                    "filename": self.filename,
                    "total": int(total or 0),
                }),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _part_identity_matches(self, part: Path, probed_total: int) -> bool:
        """True only if the .part's identity sidecar matches this exact
        repo/file (and the recorded total size, when known)."""
        try:
            data = json.loads(
                self._part_meta_path(part).read_text(encoding="utf-8")
            )
        except Exception:
            return False
        if (data.get("repo_id") != self.repo_id
                or data.get("filename") != self.filename):
            return False
        recorded = int(data.get("total") or 0)
        if recorded and probed_total and recorded != probed_total:
            return False
        return True

    def _run_single(self, url, target: Path, part: Path):
        """Download *url* into *part* over a single resumable connection."""
        try:
            headers = self._base_headers()

            resume_pos = part.stat().st_size if part.exists() else 0
            if resume_pos:
                # Guard against resuming onto a stale/mismatched .part. Size
                # alone can't prove a partial belongs to this file — a leftover
                # .part from a *different* file under the same name that is
                # SMALLER than the new remote file would otherwise be appended
                # onto, producing a full-size but corrupt result. So discard the
                # .part unless its identity sidecar matches this exact repo/file
                # (and recorded size), or if it already exceeds the remote size.
                # (A matching .part exactly == total is kept and the 416 branch
                # below finalizes it.)
                probed_total, _ = self._probe(url)
                if (
                    not self._part_identity_matches(part, probed_total)
                    or (probed_total and resume_pos > probed_total)
                ):
                    self._safe_unlink(part)
                    self._safe_unlink(self._part_meta_path(part))
                    resume_pos = 0
            if resume_pos:
                headers["Range"] = f"bytes={resume_pos}-"
                self.progress.emit(
                    f"Resuming {self.filename} at {resume_pos / 1024**3:.2f} GB...", 0.0
                )

            opener = urllib.request.build_opener(_StripAuthOnRedirect)
            request = urllib.request.Request(url, headers=headers)

            try:
                response = opener.open(request, timeout=self._SOCKET_TIMEOUT)
            except urllib.error.HTTPError as http_err:
                if http_err.code == 416 and resume_pos:
                    # 416 = requested range unsatisfiable. Only treat the .part
                    # as complete when its size matches the known remote total —
                    # a 416 alone doesn't prove completeness (the probe may have
                    # failed, or the .part may be oversized/corrupt). When we
                    # can't verify, keep the .part for a future resume rather
                    # than promoting possibly-corrupt data to the final file.
                    if probed_total and resume_pos == probed_total:
                        part.replace(target)
                        self._safe_unlink(self._part_meta_path(part))
                        self.progress.emit("Download complete", 1.0)
                        self.finished.emit(str(target))
                        return
                    self.error.emit(
                        "Could not verify the existing partial download; "
                        "download again to retry."
                    )
                    return
                raise

            with response:
                if resume_pos and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = (
                        int(content_range.rsplit("/", 1)[-1])
                        if "/" in content_range else 0
                    )
                    mode = "ab"
                else:
                    # Server ignored the Range request — start over
                    total = int(response.headers.get("Content-Length") or 0)
                    resume_pos = 0
                    mode = "wb"

                downloaded = resume_pos
                last_emit = 0.0
                self.target_dir.mkdir(parents=True, exist_ok=True)

                if resume_pos == 0:
                    # Fresh .part — stamp its identity so a future resume can
                    # confirm the partial belongs to this exact file.
                    self._write_part_identity(part, total)

                with open(part, mode) as f:
                    while True:
                        if self._cancelled:
                            # User cancelled — discard so a new model can be chosen
                            self._safe_unlink(part)
                            self._safe_unlink(self._part_meta_path(part))
                            self.error.emit("Download cancelled.")
                            return
                        chunk = response.read(self._CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.monotonic()
                        if now - last_emit >= 0.5:
                            last_emit = now
                            if total:
                                fraction = downloaded / total
                                self.progress.emit(
                                    f"{self.filename} — "
                                    f"{downloaded / 1024**3:.2f} / {total / 1024**3:.2f} GB "
                                    f"({fraction * 100:.0f}%)",
                                    fraction,
                                )
                            else:
                                self.progress.emit(
                                    f"{self.filename} — {downloaded / 1024**3:.2f} GB...",
                                    0.0,
                                )

            if total and downloaded < total:
                # Connection dropped (not a user cancel) — keep .part for resume
                self.error.emit(
                    f"Download incomplete ({downloaded / 1024**3:.2f} of "
                    f"{total / 1024**3:.2f} GB) — connection dropped. "
                    "Download again to resume."
                )
                return

            part.replace(target)
            self._safe_unlink(self._part_meta_path(part))
            self.progress.emit("Download complete", 1.0)
            self.finished.emit(str(target))

        except Exception as exc:
            if self._cancelled:
                self._safe_unlink(part)
                self._safe_unlink(self._part_meta_path(part))
                self.error.emit("Download cancelled.")
                return
            self.error.emit(self._format_error(exc))
