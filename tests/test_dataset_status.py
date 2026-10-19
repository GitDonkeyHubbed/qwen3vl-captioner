"""Dataset coverage counts captions, not sidecar files.

An empty or unreadable ``.txt`` used to show "Yes" and push Coverage to
100%. A legacy encoding must still count, or the tab offers it up to be
overwritten.
"""

import pytest

pytest.importorskip("PyQt6")

from PIL import Image  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

import gui.dataset_panel as dataset_panel  # noqa: E402
from gui.caption_io import CaptionFile  # noqa: E402
from gui.dataset_panel import DatasetPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _png(path):
    Image.new("RGB", (8, 6), (20, 30, 40)).save(path)


def _stats(panel):
    def value(widget):
        return widget.findChild(QLabel, "stat_value").text()

    return {
        "total": value(panel._stat_total),
        "captioned": value(panel._stat_captioned),
        "uncaptioned": value(panel._stat_uncaptioned),
        "coverage": value(panel._stat_coverage),
    }


def _rows(panel):
    table = panel._table
    rows = []
    for row in range(table.rowCount()):
        rows.append((
            table.item(row, 0).text(),
            table.item(row, 3).text(),
            table.item(row, 4).text(),
        ))
    return rows


def test_blank_sidecars_are_not_coverage(qapp, tmp_path):
    """A whitespace sidecar is uncaptioned. 1/3 floors to 33%; 2/3 floors to
    66% (rounding would report 67)."""
    blank = tmp_path / "a.png"
    missing = tmp_path / "b.png"
    kept = tmp_path / "c.png"
    also = tmp_path / "d.png"
    for path in (blank, missing, kept, also):
        _png(path)
    blank.with_suffix(".txt").write_text("  \n\t", encoding="utf-8")
    kept.with_suffix(".txt").write_text("hello", encoding="utf-8")
    also.with_suffix(".txt").write_text("world", encoding="utf-8")
    panel = DatasetPanel()
    panel.populate([blank, missing, kept])

    assert _rows(panel) == [
        ("a.png", "No", "(empty file)"),
        ("b.png", "No", ""),
        ("c.png", "Yes", "hello"),
    ]
    assert _stats(panel) == {
        "total": "3",
        "captioned": "1",
        "uncaptioned": "2",
        "coverage": "33%",
    }

    panel.populate([blank, kept, also])
    assert _stats(panel)["captioned"] == "2"
    assert _stats(panel)["coverage"] == "66%"
    panel.deleteLater()


def test_read_error_is_uncaptioned(qapp, tmp_path, monkeypatch):
    locked = tmp_path / "locked.png"
    _png(locked)
    real = dataset_panel.read_caption

    def fake(path):
        if path == locked:
            return CaptionFile(
                text="", exists=True, mtime=None, decode_error=False,
                read_error="denied",
            )
        return real(path)

    monkeypatch.setattr(dataset_panel, "read_caption", fake)
    panel = DatasetPanel()
    panel.populate([locked])

    assert _rows(panel) == [("locked.png", "No", "(read error)")]
    assert _stats(panel)["coverage"] == "0%"
    panel.deleteLater()


def test_preview_truncates_after_120_and_lossy_text_still_counts(qapp, tmp_path):
    exact = tmp_path / "exact.png"
    over = tmp_path / "over.png"
    legacy = tmp_path / "legacy.png"
    for path in (exact, over, legacy):
        _png(path)
    exact.with_suffix(".txt").write_text("x" * 120, encoding="utf-8")
    over.with_suffix(".txt").write_text("y" * 121, encoding="utf-8")
    legacy.with_suffix(".txt").write_bytes(b"caf\xe9 caption")

    panel = DatasetPanel()
    panel.populate([exact, over, legacy])

    rows = {name: (status, preview) for name, status, preview in _rows(panel)}
    assert rows["exact.png"] == ("Yes", "x" * 120)
    assert rows["over.png"] == ("Yes", "y" * 120 + "...")
    status, preview = rows["legacy.png"]
    assert status == "Yes"
    assert "\ufffd" in preview
    assert preview != "(empty file)"
    assert _stats(panel)["captioned"] == "3"
    panel.deleteLater()
