"""Settings dialog: version check, cancel, and save.

The update check renders a remote tag in a rich-text label. A string compare
would call 1.4.10 older than 1.4.4, and an unescaped tag is HTML the dialog
will follow links out of.
"""

import io
import json
import urllib.error
import urllib.request

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from gui import config as cfg  # noqa: E402
from gui.app_settings_dialog import AppSettingsDialog  # noqa: E402
from gui.version import GITHUB_REPO  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".vlcaptioner"
    monkeypatch.setattr(cfg, "_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(cfg, "_CONFIG_FILE", cfg_dir / "config.json")


_RELEASES = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_TAGS = f"https://api.github.com/repos/{GITHUB_REPO}/tags"


class _Body(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _check(monkeypatch, current, routes):
    monkeypatch.setattr("gui.version.APP_VERSION", current)
    seen = []

    def urlopen(req, timeout=None):
        seen.append((req.full_url, timeout))
        action = routes[req.full_url]
        if isinstance(action, BaseException):
            raise action
        return _Body(json.dumps(action).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    dialog = AppSettingsDialog()
    got = []
    dialog._update_check_done.connect(
        lambda message, is_new: got.append((message, is_new))
    )
    dialog._fetch_latest_version()
    assert len(got) == 1
    return got[0], seen


@pytest.mark.parametrize(
    ("current", "latest", "is_new"),
    [
        ("1.4.4", "v1.4.10", True),  # "1.4.10" < "1.4.4" as text
        ("1.9.0", "1.10.0", True),  # "1.10.0" < "1.9.0" as text
        ("1.4.4", "v1.4.4", False),
        ("1.4.4", "1.4.3", False),
    ],
)
def test_update_compare_is_numeric(qapp, monkeypatch, current, latest, is_new):
    (message, got_new), seen = _check(
        monkeypatch, current, {_RELEASES: {"tag_name": latest}}
    )

    assert got_new is is_new
    assert [url for url, _timeout in seen] == [_RELEASES]
    assert seen[0][1] == 6
    assert latest in message
    if is_new:
        assert "New version available" in message
    else:
        assert "up to date" in message


def test_remote_tag_is_escaped_in_the_update_notice(qapp, monkeypatch):
    latest = "v1.5.0<img src=x onerror=alert(1)>"
    (message, is_new), _seen = _check(
        monkeypatch, "1.4.4", {_RELEASES: {"tag_name": latest}}
    )

    assert is_new is True
    assert "<img" not in message
    assert "&lt;img" in message
    # The only raw tags left are the releases link the app itself wrote.
    stripped = message.replace("<a ", "", 1).replace("</a>", "", 1)
    assert "<" not in stripped
    assert f"https://github.com/{GITHUB_REPO}/releases" in message


def test_remote_tag_is_escaped_when_already_up_to_date(qapp, monkeypatch):
    latest = "1.0.0<script>alert(1)</script>"
    (message, is_new), _seen = _check(
        monkeypatch, "1.4.4", {_RELEASES: {"tag_name": latest}}
    )

    assert is_new is False
    assert "up to date" in message
    assert "<script" not in message
    assert "&lt;script" in message


def test_published_release_wins_over_a_newer_tag(qapp, monkeypatch):
    """A release answer must stop the lookup. Falling through to tags would
    advertise v9.9.9 from a stale tag list."""
    (message, is_new), seen = _check(
        monkeypatch,
        "1.4.4",
        {
            _RELEASES: {"tag_name": "v1.4.4"},
            _TAGS: [{"name": "v9.9.9"}],
        },
    )

    assert is_new is False
    assert "9.9.9" not in message
    assert [url for url, _timeout in seen] == [_RELEASES]


def test_update_check_falls_back_to_tags(qapp, monkeypatch):
    (message, is_new), seen = _check(
        monkeypatch,
        "1.4.4",
        {
            _RELEASES: {"draft": True},
            _TAGS: [{"name": "v1.4.10"}],
        },
    )

    assert is_new is True
    assert "v1.4.10" in message
    assert [url for url, _timeout in seen] == [_RELEASES, _TAGS]
    assert all(timeout == 6 for _url, timeout in seen)


def test_update_check_reports_when_nothing_is_published(qapp, monkeypatch):
    (message, is_new), seen = _check(
        monkeypatch,
        "1.4.4",
        {
            _RELEASES: urllib.error.URLError("offline"),
            _TAGS: urllib.error.URLError("offline"),
        },
    )

    assert is_new is False
    assert message.startswith("No releases found")
    assert len(seen) == 2
    assert all(timeout == 6 for _url, timeout in seen)


def test_cancel_restores_the_theme_preview(qapp):
    dialog = AppSettingsDialog()
    seen = []
    dialog.theme_changed.connect(seen.append)

    dialog._dark_mode_cb.setChecked(False)
    dialog.reject()

    assert seen == ["light", "dark"]
    assert cfg.load_config()["theme"] == "dark"


def test_cancel_without_a_theme_change_emits_nothing(qapp):
    dialog = AppSettingsDialog()
    seen = []
    dialog.theme_changed.connect(seen.append)

    dialog.reject()

    assert seen == []


def test_save_strips_the_token_and_stores_the_theme(qapp):
    dialog = AppSettingsDialog()
    dialog._dark_mode_cb.setChecked(False)
    dialog._token_input.setText("  hf_secret  ")

    dialog._save_and_close()

    assert dialog.result() == QDialog.DialogCode.Accepted
    loaded = cfg.load_config()
    assert loaded["hf_token"] == "hf_secret"
    assert loaded["theme"] == "light"


def test_failed_save_keeps_the_dialog_open(qapp, monkeypatch):
    warnings = []

    def warning(*args, **kwargs):
        warnings.append(args)
        return 0

    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr("gui.app_settings_dialog.save_config", lambda data: False)

    dialog = AppSettingsDialog()
    dialog._token_input.setText("hf_should_not_land")
    dialog._save_and_close()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert warnings
    assert cfg.load_config()["hf_token"] == ""
