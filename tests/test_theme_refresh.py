"""Runtime theme switches and thumbnail status resets.

A theme switch swaps the COLORS dict in place and re-applies the app
stylesheet, but a widget's own inline stylesheet (and any brush set on a table
cell) keeps the hex strings it was built with. Each panel's refresh_theme() has
to re-resolve those itself. These tests switch dark -> light exactly the way
MainWindow._on_theme_changed does and compare against the new palette.

The last group covers ThumbnailItem.set_status: leaving "queued"/"processing"
must not strand the "Queued"/"Captioning..." placeholder on the row.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QLabel, QPushButton, QScrollArea, QWidget,
)

from gui import dataset_panel, theme  # noqa: E402
from gui.theme import COLORS, get_stylesheet, set_theme  # noqa: E402

# Keep a reference: a garbage-collected QApplication takes every widget with it.
_APP = QApplication.instance() or QApplication([])

DARK, LIGHT = theme._DARK_COLORS, theme._LIGHT_COLORS


def _switch(mode):
    # Mirrors MainWindow._on_theme_changed.
    set_theme(mode)
    _APP.setStyleSheet(get_stylesheet(mode))


@pytest.fixture
def dark_app():
    """Start in dark mode and always put dark back (COLORS is module state)."""
    # setStyleSheet re-polishes every live widget. Windows that earlier test
    # modules deleteLater()'d are never destroyed without an event loop, and
    # thousands of them made each switch take seconds.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    _switch("dark")
    yield _APP
    _switch("dark")


def _pixel(widget, x=2, y=2):
    return widget.grab().toImage().pixelColor(x, y).name()


# ── DatasetPanel ────────────────────────────────────────────────────────

def _images(tmp_path, n=4):
    """n tiny PNGs; the even-numbered ones have a caption sidecar."""
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (32, 24), (200, 10, 10)).save(p)
        if i % 2 == 0:
            p.with_suffix(".txt").write_text(f"caption {i}", encoding="utf-8")
        paths.append(p)
    return paths


def _populated_panel(tmp_path):
    panel = dataset_panel.DatasetPanel()
    panel.resize(800, 500)
    panel.show()
    panel.populate(_images(tmp_path))
    _APP.processEvents()
    return panel


def _status_cells(panel):
    table = panel._table
    return [table.item(r, 3) for r in range(table.rowCount())]


def test_dataset_refresh_theme_does_not_reread_sidecars(dark_app, tmp_path, monkeypatch):
    # refresh_theme used to call populate(), re-reading every sidecar (and
    # every image header) on the UI thread — once per checkbox toggle in the
    # settings dialog's live preview, even with the Dataset tab hidden.
    panel = _populated_panel(tmp_path)
    calls = []
    real_read_caption = dataset_panel.read_caption
    monkeypatch.setattr(
        dataset_panel, "read_caption",
        lambda path: calls.append(path) or real_read_caption(path),
    )

    _switch("light")
    panel.refresh_theme()

    assert calls == []
    assert panel._table.rowCount() == 4
    assert [c.text() for c in _status_cells(panel)] == ["Yes", "No", "Yes", "No"]


def test_dataset_refresh_theme_restyles_every_inline_sheet(dark_app, tmp_path):
    panel = _populated_panel(tmp_path)
    assert _pixel(panel) == DARK["bg_dark"]

    _switch("light")
    panel.refresh_theme()
    _APP.processEvents()

    # Every inline stylesheet matches a panel built fresh in light mode.
    fresh = dataset_panel.DatasetPanel()
    old_widgets = [panel] + panel.findChildren(QWidget)
    new_widgets = [fresh] + fresh.findChildren(QWidget)
    stale = [
        (type(a).__name__, a.objectName(), a.styleSheet().strip()[:60])
        for a, b in zip(old_widgets, new_widgets, strict=True)
        if a.styleSheet() != b.styleSheet()
    ]
    assert not stale, f"stale inline styles after theme switch: {stale}"

    # And it actually paints light.
    assert _pixel(panel) == LIGHT["bg_dark"]
    assert _pixel(panel._stats_frame, 8, 8) == LIGHT["bg_card"]
    assert _pixel(panel._table.horizontalHeader(), 3, 3) == LIGHT["bg_surface"]


def test_dataset_refresh_theme_recolours_existing_status_cells(dark_app, tmp_path, monkeypatch):
    # success/error happen to be identical in both palettes today, so give
    # them distinct values to prove the existing cell brushes are re-resolved.
    panel = _populated_panel(tmp_path)

    _switch("light")
    monkeypatch.setitem(COLORS, "success", "#123456")
    monkeypatch.setitem(COLORS, "error", "#654321")
    panel.refresh_theme()

    colours = {c.text(): c.foreground().color() for c in _status_cells(panel)}
    assert colours == {"Yes": QColor("#123456"), "No": QColor("#654321")}


def test_dataset_refresh_theme_on_empty_panel(dark_app):
    panel = dataset_panel.DatasetPanel()
    _switch("light")
    panel.refresh_theme()
    assert panel._table.rowCount() == 0
    assert LIGHT["bg_dark"] in panel.styleSheet()


# ── SettingsPanel / NotificationPanel ───────────────────────────────────

def test_settings_scroll_follows_runtime_theme_switch(dark_app):
    from gui.settings_panel import SettingsPanel

    panel = SettingsPanel()
    panel.show()
    _APP.processEvents()
    scroll = panel.findChild(QScrollArea, "settingsScroll")
    assert scroll.palette().window().color().name() == DARK["bg_darkest"]

    _switch("light")
    panel.refresh_theme()
    _APP.processEvents()

    # The rule used to be re-applied to the SettingsPanel itself, where the
    # scroll area's own (stale) stylesheet outranked it.
    assert scroll.palette().window().color().name() == LIGHT["bg_darkest"]
    header = panel.layout().itemAt(0).widget()
    assert LIGHT["border"] in header.styleSheet()
    assert DARK["border"] not in header.styleSheet()


def test_notification_header_follows_runtime_theme_switch(dark_app):
    from gui.notification_panel import NotificationPanel, NotificationStore

    store = NotificationStore()
    store.add("caption saved", "success")
    panel = NotificationPanel(store)
    panel.show()
    _APP.processEvents()

    header = panel.layout().itemAt(0).widget()
    title = next(w for w in header.findChildren(QLabel) if w.text() == "Notifications")
    clear = next(w for w in header.findChildren(QPushButton) if w.text() == "Clear All")
    assert title.palette().windowText().color().name() == DARK["text_primary"]

    _switch("light")
    panel.refresh_theme()
    _APP.processEvents()

    # The title was #f4f4f5 on #e4e4e7 (1.15:1) after a switch to light.
    assert title.palette().windowText().color().name() == LIGHT["text_primary"]
    assert LIGHT["border"] in header.styleSheet()
    assert DARK["border"] not in header.styleSheet()
    assert f"color: {LIGHT['text_secondary']}" in clear.styleSheet()  # hover


# ── ThumbnailItem.set_status ────────────────────────────────────────────

@pytest.mark.parametrize("start", ["processing", "queued"])
@pytest.mark.parametrize("end", ["idle", "generated", "done"])
def test_leaving_transient_status_clears_placeholder(tmp_path, start, end):
    # A cancelled or failed batch resets uncaptioned rows to idle; with no
    # cached preview the label kept "Captioning..." / "Queued" indefinitely.
    from gui.file_browser import ThumbnailItem

    item = ThumbnailItem(tmp_path / "x.png")
    item.set_status(start)
    item.set_status(end)
    assert item.preview_label.text() == ""
    assert item.preview_label.property("class") == "thumb-preview"


def test_leaving_transient_status_restores_existing_preview(tmp_path):
    from gui.file_browser import ThumbnailItem

    item = ThumbnailItem(tmp_path / "x.png")
    item.set_caption_preview("a red square on white")
    item.set_status("processing")
    assert item.preview_label.text() == "Captioning..."
    assert item.preview_label.property("class") == "thumb-preview-active"

    item.set_status("idle")
    assert item.preview_label.text() == "a red square on white"
    assert item.preview_label.property("class") == "thumb-preview"

    item.set_status("done")
    assert item.preview_label.text() == "a red square on white"
    assert item._check_overlay.isVisibleTo(item)
