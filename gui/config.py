"""
Persistent application configuration for VL-CAPTIONER Studio Pro.

Stores settings as JSON in ~/.vlcaptioner/config.json.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path.home() / ".vlcaptioner"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: Dict[str, Any] = {
    "theme": "dark",
    "hf_token": "",
    # Absolute paths of user-added GGUF models (issue #7)
    "custom_models": [],
    # When True, captions are written to .txt sidecars without the
    # per-image confirmation popup
    "auto_save_captions": False,
}


def _ensure_dir():
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load config from disk, falling back to defaults for missing keys."""
    # Deep copy so mutable defaults (e.g. the custom_models list) are never
    # shared with _DEFAULTS — otherwise callers like add_custom_model() would
    # mutate the module-level defaults in place when no config file exists yet.
    cfg = copy.deepcopy(_DEFAULTS)
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


def get_custom_models() -> list:
    """Return the list of user-added GGUF model paths (strings)."""
    models = load_config().get("custom_models", [])
    return models if isinstance(models, list) else []


def add_custom_model(path: str):
    """Remember a user-added GGUF model path (deduplicated, most recent last)."""
    cfg = load_config()
    models = cfg.get("custom_models", [])
    if not isinstance(models, list):
        models = []
    if path in models:
        models.remove(path)
    models.append(path)
    cfg["custom_models"] = models
    save_config(cfg)


def get_auto_save_captions() -> bool:
    """Return whether captions should be saved without the confirmation popup."""
    return bool(load_config().get("auto_save_captions", False))


def set_auto_save_captions(enabled: bool):
    cfg = load_config()
    cfg["auto_save_captions"] = bool(enabled)
    save_config(cfg)
