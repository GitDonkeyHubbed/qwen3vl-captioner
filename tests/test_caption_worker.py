"""CaptionWorker routing, sidecar protection, and the encoder-download dialog.

These are the last gates before a generated string or a locked sidecar is
treated as safe to write. The workflow suite stubs CaptionWorker; these
tests drive the real objects.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QCloseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.caption_io import CaptionFile  # noqa: E402
from gui.main_window import (  # noqa: E402
    CaptionWorker,
    _sidecar_is_protected,
    _UnclosableProgressDialog,
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class _Engine:
    """Returns a caption and honours cancel_check the way the real engines do."""

    def __init__(self, text="a red car"):
        self.text = text
        self.calls = 0

    def caption_image(self, **kwargs):
        self.calls += 1
        callback = kwargs.get("stream_callback")
        if callback:
            callback(self.text)
        cancel = kwargs.get("cancel_check")
        if cancel and cancel():
            return self.text
        return self.text


def _worker(engine, image=Path("pic.jpg")):
    return CaptionWorker(
        engine, image, prompt="describe", temperature=0.6, top_p=0.9,
        max_tokens=64, prefix="", suffix="",
    )


def test_finished_generation_emits_finished_not_error(qapp):
    engine = _Engine()
    worker = _worker(engine)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert finished == ["a red car"]
    assert errors == []


def test_cancelled_generation_is_not_emitted_as_finished(qapp):
    """A cancelled stream returns the partial text via a normal return. If
    that lands on finished(), auto-save writes the fragment as a caption."""
    engine = _Engine("partial")
    worker = _worker(engine)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.cancel()
    worker.run()

    assert errors == ["cancelled"]
    assert finished == []


def test_cancel_during_caption_image_still_routes_to_error(qapp):
    holder = []

    class _MidCancelEngine:
        def caption_image(self, **kwargs):
            holder[0].cancel()
            kwargs["stream_callback"]("hel")
            if kwargs["cancel_check"]():
                return "hel"
            return "hello"

    engine = _MidCancelEngine()
    worker = _worker(engine)
    holder.append(worker)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == ["cancelled"]
    assert finished == []


def test_engine_exception_is_reported_on_error(qapp):
    class _Boom:
        def caption_image(self, **kwargs):
            raise RuntimeError("native crash")

    worker = _worker(_Boom())
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    assert errors and "native crash" in errors[0]


def test_empty_sidecar_is_not_protected():
    missing = CaptionFile(
        text="", exists=False, mtime=None, decode_error=False, read_error=None
    )
    blank = CaptionFile(
        text="", exists=True, mtime=1.0, decode_error=False, read_error=None
    )
    assert _sidecar_is_protected(missing) is False
    assert _sidecar_is_protected(blank) is False


def test_text_or_unreadable_sidecar_is_protected():
    captioned = CaptionFile(
        text="a cat", exists=True, mtime=1.0, decode_error=False, read_error=None
    )
    locked = CaptionFile(
        text="", exists=True, mtime=None, decode_error=False,
        read_error="[Errno 13] Permission denied",
    )
    assert _sidecar_is_protected(captioned) is True
    assert _sidecar_is_protected(locked) is True


def test_encoder_dialog_ignores_close_until_allowed(qapp):
    """Esc / the title-bar close used to hide the dialog while the nested
    download event loop kept running — an invisible, unstoppable transfer."""
    dlg = _UnclosableProgressDialog("Downloading encoder…", None, 0, 0)

    blocked = QCloseEvent()
    dlg.closeEvent(blocked)
    assert blocked.isAccepted() is False

    dlg.reject()  # Esc arrives as reject(); must not arm the close flag
    still_blocked = QCloseEvent()
    dlg.closeEvent(still_blocked)
    assert still_blocked.isAccepted() is False

    dlg.allow_close()
    allowed = QCloseEvent()
    dlg.closeEvent(allowed)
    assert allowed.isAccepted() is True
    dlg.deleteLater()
