"""Tests for persistent app config (gui.config).

Covers the save/load round-trip, default-fill for missing keys, corrupt-file
recovery, custom-model dedup (most-recent-last), and the auto-save toggle.

Each test is isolated to a temp config file via an autouse fixture, so the
developer's real ~/.vlcaptioner/config.json is never read or written.
"""

import pytest

from gui import config as cfg


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Redirect the module's config dir/file to a per-test temp location."""
    cfg_dir = tmp_path / ".vlcaptioner"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(cfg, "_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(cfg, "_CONFIG_FILE", cfg_file)
    return cfg_file


def test_load_defaults_when_no_file():
    loaded = cfg.load_config()
    assert loaded["theme"] == "dark"
    assert loaded["hf_token"] == ""
    assert loaded["custom_models"] == []
    assert loaded["auto_save_captions"] is False


def test_save_load_round_trip(_isolated_config):
    data = {
        "theme": "light",
        "hf_token": "hf_secret",
        "custom_models": ["/models/a.gguf"],
        "auto_save_captions": True,
    }
    cfg.save_config(data)
    assert _isolated_config.exists()
    loaded = cfg.load_config()
    for key, value in data.items():
        assert loaded[key] == value


def test_partial_config_merges_defaults():
    cfg.save_config({"theme": "light"})
    loaded = cfg.load_config()
    assert loaded["theme"] == "light"
    # Keys absent from disk fall back to defaults.
    assert loaded["hf_token"] == ""
    assert loaded["auto_save_captions"] is False


def test_corrupt_file_falls_back_to_defaults(_isolated_config):
    _isolated_config.parent.mkdir(parents=True, exist_ok=True)
    _isolated_config.write_text("{not valid json", encoding="utf-8")
    loaded = cfg.load_config()
    assert loaded["theme"] == "dark"


def test_add_custom_model_dedups_most_recent_last():
    cfg.add_custom_model("/m/a.gguf")
    cfg.add_custom_model("/m/b.gguf")
    cfg.add_custom_model("/m/a.gguf")  # re-adding moves it to the end
    assert cfg.get_custom_models() == ["/m/b.gguf", "/m/a.gguf"]


def test_get_custom_models_returns_list():
    assert cfg.get_custom_models() == []
    cfg.add_custom_model("/m/a.gguf")
    assert cfg.get_custom_models() == ["/m/a.gguf"]


def test_auto_save_toggle_round_trips():
    assert cfg.get_auto_save_captions() is False
    cfg.set_auto_save_captions(True)
    assert cfg.get_auto_save_captions() is True
    cfg.set_auto_save_captions(False)
    assert cfg.get_auto_save_captions() is False


def test_theme_and_token_convenience():
    cfg.save_config({"theme": "light", "hf_token": "hf_abc"})
    assert cfg.get_theme() == "light"
    assert cfg.get_hf_token() == "hf_abc"


def test_save_config_reports_failure(monkeypatch, _isolated_config):
    """A failed write must return False (the settings dialog surfaces it)
    and must not clobber the previously saved file."""
    assert cfg.save_config({"theme": "light"}) is True

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cfg.os, "replace", boom)
    assert cfg.save_config({"theme": "dark"}) is False
    # The atomic write failed at the swap — the old content must survive.
    assert cfg.load_config()["theme"] == "light"


def test_save_config_returns_true_on_success():
    assert cfg.save_config({"theme": "dark"}) is True


def test_last_import_dir_round_trip():
    assert cfg.get_last_import_dir() == ""
    cfg.set_last_import_dir("/data/images")
    assert cfg.get_last_import_dir() == "/data/images"


def test_last_import_dir_rejects_non_string():
    cfg.save_config({"last_import_dir": ["not", "a", "string"]})
    assert cfg.get_last_import_dir() == ""
