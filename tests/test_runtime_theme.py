"""Regression tests for palette-dependent inline styles at runtime."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from gui.dataset_panel import DatasetPanel  # noqa: E402
from gui.notification_panel import NotificationPanel, NotificationStore  # noqa: E402
from gui.settings_panel import SettingsPanel  # noqa: E402
from gui.theme import COLORS, set_theme  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_dark_theme():
    set_theme("dark")
    yield
    set_theme("dark")


def test_dataset_refresh_theme_restyles_all_palette_widgets(qapp):
    panel = DatasetPanel()
    set_theme("light")

    panel.refresh_theme()

    assert COLORS["bg_dark"] in panel.styleSheet()
    assert COLORS["bg_card"] in panel._stats_frame.styleSheet()
    assert COLORS["bg_darkest"] in panel._table.styleSheet()
    assert COLORS["bg_surface"] in panel._table.styleSheet()
    assert COLORS["text_dim"] in panel._empty_label.styleSheet()
    assert COLORS["text_primary"] in panel._title.styleSheet()
    for stat in (
        panel._stat_total, panel._stat_captioned,
        panel._stat_uncaptioned, panel._stat_coverage,
    ):
        assert COLORS["text_dim"] in stat.findChild(
            QLabel, "stat_label"
        ).styleSheet()
        assert COLORS["text_primary"] in stat.findChild(
            QLabel, "stat_value"
        ).styleSheet()
    panel.deleteLater()


def test_settings_refresh_theme_targets_scroll_area(qapp):
    panel = SettingsPanel()
    set_theme("light")

    panel.refresh_theme()

    assert COLORS["bg_darkest"] in panel._scroll.styleSheet()
    assert "settingsScroll" not in panel.styleSheet()
    panel.deleteLater()


def test_notification_refresh_theme_restyles_header_controls(qapp):
    panel = NotificationPanel(NotificationStore())
    set_theme("light")

    panel.refresh_theme()

    assert COLORS["bg_darkest"] in panel.styleSheet()
    assert COLORS["border"] in panel._header.styleSheet()
    assert COLORS["text_primary"] in panel._title.styleSheet()
    assert COLORS["text_dim"] in panel._clear_btn.styleSheet()
    assert COLORS["text_secondary"] in panel._clear_btn.styleSheet()
    panel.deleteLater()
