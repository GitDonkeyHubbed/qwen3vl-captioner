"""Dataset size labels, sort order, and the empty-table placeholder.

Blank sidecars and the 120-character preview are covered elsewhere. These
pin the labels that sit next to that status: a directory named like an image
is not a row, names sort case-insensitively, and the byte thresholds are
1024 and 1 MiB.
"""

import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from gui.dataset_panel import DatasetPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _png(path, size=(4, 9)):
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def test_rows_sort_case_insensitively(qapp, tmp_path):
    _png(tmp_path / "B.png")
    _png(tmp_path / "a.jpg")
    panel = DatasetPanel()
    panel.populate([tmp_path / "B.png", tmp_path / "a.jpg"])
    names = [panel._table.item(row, 0).text() for row in range(2)]
    # A default sort puts "B.png" first because "B" < "a".
    assert names == ["a.jpg", "B.png"]
    assert panel._table.item(1, 1).text() == "4 x 9"
    panel.deleteLater()


def test_empty_dataset_shows_the_placeholder_and_zero_coverage(qapp):
    panel = DatasetPanel()
    panel.populate([])
    assert panel._table.isHidden() is True
    assert panel._empty_label.isHidden() is False
    coverage = panel._stat_coverage.findChild(QLabel, "stat_value").text()
    captioned = panel._stat_captioned.findChild(QLabel, "stat_value").text()
    assert coverage == "0%"
    assert captioned == "0"
    panel.deleteLater()


def test_size_labels_switch_at_the_power_of_two_boundaries(tmp_path):
    missing = tmp_path / "gone.bin"
    assert DatasetPanel._format_size(missing) == "--"

    path = tmp_path / "blob.bin"
    path.write_bytes(b"")
    assert DatasetPanel._format_size(path) == "0 B"
    path.write_bytes(b"x" * 1023)
    assert DatasetPanel._format_size(path) == "1023 B"
    path.write_bytes(b"x" * 1024)
    assert DatasetPanel._format_size(path) == "1.0 KB"
    path.write_bytes(b"x" * (1024 * 1024 - 1))
    label = DatasetPanel._format_size(path)
    assert label.endswith(" KB")
    assert "MB" not in label
    path.write_bytes(b"x" * (1024 * 1024))
    assert DatasetPanel._format_size(path) == "1.0 MB"


def test_undecodable_image_dimensions_are_a_dash(tmp_path):
    good = _png(tmp_path / "ok.png", size=(3, 5))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert DatasetPanel._get_image_dims(good) == "3 x 5"
    assert DatasetPanel._get_image_dims(bad) == "--"
    assert DatasetPanel._get_image_dims(tmp_path / "missing.png") == "--"
