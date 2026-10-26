"""Image-viewer wheel zoom and the settings panel's busy state across a batch."""

import pytest

pytest.importorskip("PyQt6")

from PIL import Image  # noqa: E402
from PyQt6.QtCore import QPoint, QPointF, QSize, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.image_viewer import ImageViewer  # noqa: E402
from gui.settings_panel import SettingsPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _wheel(widget, dy, dx=0, phase=Qt.ScrollPhase.NoScrollPhase):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    return QWheelEvent(
        pos, QPointF(widget.mapToGlobal(pos.toPoint())), QPoint(0, 0),
        QPoint(dx, dy), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        phase, False,
    )


def _zoomed_viewer(qapp, tmp_path):
    path = tmp_path / "big.png"
    image = QImage(2000, 1500, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    image.save(str(path))

    viewer = ImageViewer()
    viewer.resize(800, 600)
    viewer.show()
    qapp.processEvents()
    viewer.set_image(path)
    viewer._set_zoom(3.0)  # large enough that the view can scroll
    qapp.processEvents()
    return viewer


def test_wheel_on_the_viewport_zooms_instead_of_scrolling(qapp, tmp_path):
    # The view's wheelEvent is replaced on the instance; sip dispatches that
    # override, so a real wheel event zooms. Without it the view scrolls.
    viewer = _zoomed_viewer(qapp, tmp_path)
    viewport = viewer._view.viewport()
    scroll = viewer._view.verticalScrollBar()
    assert scroll.maximum() > 0
    start = scroll.value()

    QApplication.sendEvent(viewport, _wheel(viewport, -120))
    assert viewer._zoom == pytest.approx(2.95)
    assert scroll.value() == start

    QApplication.sendEvent(viewport, _wheel(viewport, 120))
    assert viewer._zoom == pytest.approx(3.0)
    viewer.close()
    viewer.deleteLater()


@pytest.mark.parametrize("dx, dy, phase", [
    (120, 0, Qt.ScrollPhase.NoScrollPhase),  # sideways swipe / tilt wheel
    (0, 0, Qt.ScrollPhase.ScrollBegin),  # trackpad gesture start ...
    (0, 0, Qt.ScrollPhase.ScrollEnd),  # ... and end
])
def test_wheel_without_vertical_motion_leaves_the_zoom(qapp, tmp_path, dx, dy, phase):
    # A zero vertical delta took the zoom-out branch, so every trackpad
    # gesture's begin/end events and any sideways scroll shrank the image.
    viewer = _zoomed_viewer(qapp, tmp_path)
    viewport = viewer._view.viewport()

    QApplication.sendEvent(viewport, _wheel(viewport, dy, dx=dx, phase=phase))
    assert viewer._zoom == pytest.approx(3.0)
    viewer.close()
    viewer.deleteLater()


def test_zoom_stops_at_the_floor_and_the_ceiling(qapp, tmp_path):
    """A wheel past either stop used to drive the transform through 0 and blank
    the picture, or keep scaling without a limit."""
    viewer = _zoomed_viewer(qapp, tmp_path)
    viewport = viewer._view.viewport()

    viewer._set_zoom(100)
    assert viewer._zoom == pytest.approx(5.0)
    assert viewer.zoom_label.text() == "500%"
    QApplication.sendEvent(viewport, _wheel(viewport, 120))
    assert viewer._zoom == pytest.approx(5.0)

    viewer._set_zoom(-1)
    assert viewer._zoom == pytest.approx(0.1)
    assert viewer.zoom_label.text() == "10%"
    QApplication.sendEvent(viewport, _wheel(viewport, -120))
    assert viewer._zoom == pytest.approx(0.1)

    viewer.close()
    viewer.deleteLater()


def test_fit_does_not_upscale_and_a_tiny_view_stays_visible(qapp, tmp_path, monkeypatch):
    path = tmp_path / "tiny.png"
    image = QImage(40, 20, QImage.Format.Format_RGB32)
    image.fill(QColor("blue"))
    image.save(str(path))

    viewer = ImageViewer()
    viewer.resize(800, 600)
    viewer.show()
    qapp.processEvents()
    viewer.set_image(path)
    qapp.processEvents()

    view = viewer._view.viewport().size()
    raw = min((view.width() - 20) / 40, (view.height() - 20) / 20)
    assert raw > 1
    assert viewer._zoom == pytest.approx(1.0)

    # (8 - 20) / 40 is negative. Without the 0.1 floor the transform flips
    # and the image disappears.
    monkeypatch.setattr(viewer._view.viewport(), "size", lambda: QSize(8, 8))
    viewer._fit_to_view()
    assert viewer._zoom == pytest.approx(0.1)

    viewer.close()
    viewer.deleteLater()


def test_viewer_honours_exif_orientation(qapp, tmp_path):
    """The preview applies the same Orientation tag the encoder does.

    Dropping setAutoTransform shows a sideways photo while the model is
    handed upright pixels.
    """
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (100, 60), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, "JPEG", exif=exif.tobytes())

    viewer = ImageViewer()
    viewer.show()
    qapp.processEvents()
    viewer.set_image(path)

    assert viewer._pixmap.width() == 60
    assert viewer._pixmap.height() == 100
    assert "60" in viewer.filename_label.text()
    viewer.close()
    viewer.deleteLater()


def test_undecodable_file_is_an_error_not_a_blank_canvas(qapp, tmp_path):
    path = tmp_path / "nope.png"
    path.write_bytes(b"not an image")

    viewer = ImageViewer()
    viewer.show()
    qapp.processEvents()
    viewer.set_image(path)

    assert viewer.filename_label.text() == "Error"
    assert "Failed to load" in viewer._empty_label.text()
    assert viewer._empty_label.isVisible()
    assert viewer._pixmap.isNull()
    assert viewer._pixmap_item is None
    viewer.close()
    viewer.deleteLater()


def test_batch_gap_keeps_cancel_and_locks_model_buttons(qapp):
    # MainWindow's order between items: set_generating(False) on the finished
    # caption, set_batch_progress for the next one, then ~100 ms later
    # set_generating(True). That gap must not look idle.
    panel = SettingsPanel()
    panel.set_batch_progress(1, 3)
    panel.set_generating(True)
    panel.set_generating(False)
    panel.set_batch_progress(2, 3)

    # isHidden(): the panel is never shown, so isVisible() is always False.
    assert panel.cancel_btn.isHidden() is False
    assert panel.load_model_btn.isEnabled() is False
    assert panel._download_btn.isEnabled() is False
    assert panel.batch_btn.isEnabled() is False

    panel.set_batch_progress(0, 0)  # batch complete / cancelled
    assert panel.cancel_btn.isHidden() is True
    assert panel.load_model_btn.isEnabled() is True
    assert panel._download_btn.isEnabled() is True
    assert panel.batch_btn.isEnabled() is True
    panel.deleteLater()
