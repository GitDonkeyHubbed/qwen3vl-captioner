"""
Persistent application configuration for VL-CAPTIONER Studio Pro.

Stores settings as JSON in ~/.vlcaptioner/config.json.
"""

import json
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path.home() / ".vlcaptioner"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: Dict[str, Any] = {
    "theme": "dark",
    "hf_token": "",
    "model_search_paths": [],
}


def get_model_search_paths() -> list:
    """Return the list of extra directories to scan for GGUF models."""
    return load_config().get("model_search_paths", [])


def set_model_search_paths(paths: list):
    """Persist the list of extra model search directories."""
    cfg = load_config()
    cfg["model_search_paths"] = [str(p) for p in paths]
    save_config(cfg)


def _ensure_dir():
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load config from disk, falling back to defaults for missing keys."""
    cfg = dict(_DEFAULTS)
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                cfg.update(stored)
        except Exception:
            pass  # corrupt file — use defaults
    return cfg


def save_config(cfg: Dict[str, Any]):
    """Persist the full config dict to disk."""
    _ensure_dir()
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass  # best-effort


def get_hf_token() -> str:
    """Convenience: return the stored HuggingFace token (may be empty)."""
    return load_config().get("hf_token", "")


def get_theme() -> str:
    """Convenience: return the stored theme mode ('dark' or 'light')."""
    return load_config().get("theme", "dark")
