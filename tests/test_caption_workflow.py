"""Caption data-safety tests driven against the real MainWindow.

These cover the ways a user's caption could previously be destroyed or
silently altered: unguarded overwrites of a hand-edit, streamed tokens
landing on the wrong image, a "saved" checkmark shown before any write, the
raw streamed text diverging from what was cached, swallowed write failures,
and Export/Batch rewriting good sidecars from memory.

The window is real; only the inference engine and the modal dialogs are
stubbed, since neither can run unattended.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from gui.caption_io import caption_path, read_caption  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def images(tmp_path):
    paths = []
    for name in ("a.jpg", "b.jpg"):
        p = tmp_path / name
        p.write_bytes(b"stub")
        paths.append(p)
    return paths


@pytest.fixture
def win(qapp, tmp_path, images):
    w = MainWindow(model_dir=tmp_path)
    w._file_browser.add_images(images)
    yield w
    w._gpu_timer.stop()
    w.deleteLater()


def _type(win, text):
    """Simulate a hand-edit (a programmatic set_caption would not be dirty)."""
    win._caption_panel.caption_text.insertPlainText(text)


class _FakeWorker:
    """Stands in for CaptionWorker: only image_path is read by the UI."""

    def __init__(self, image_path):
        self.image_path = image_path


# ── Dirty tracking ──────────────────────────────────────────────────────

def test_selection_change_offers_to_save_a_hand_edit(win, images, monkeypatch):
    win._on_image_selected(images[0])
    _type(win, "hand written caption")
    assert win._caption_panel.is_dirty()

    asked = {}

    def fake_question(parent, title, text, buttons, default):
        asked["title"] = title
        return QMessageBox.StandardButton.Save

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    win._on_image_selected(images[1])

    assert asked["title"] == "Unsaved Caption"
    # Saving actually wrote it, rather than dropping it on the floor.
    assert read_caption(images[0]).text == "hand written caption"
    assert win._current_image == images[1]


def test_cancelling_the_prompt_keeps_the_edit_and_the_selection(win, images, monkeypatch):
    win._on_image_selected(images[0])
    _type(win, "work in progress")

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    win._on_image_selected(images[1])

    assert win._current_image == images[0]
    assert win._caption_panel.get_caption() == "work in progress"
    assert win._caption_panel.is_dirty()


def test_discard_drops_the_edit_without_writing(win, images, monkeypatch):
    win._on_image_selected(images[0])
    _type(win, "throwaway")

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    win._on_image_selected(images[1])

    assert win._current_image == images[1]
    assert read_caption(images[0]).exists is False


def test_clear_all_guards_the_edit(win, images, monkeypatch):
    win._on_image_selected(images[0])
    _type(win, "unsaved")

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    win._on_clear_all()

    # Nothing was cleared — the images and the edit are still there.
    assert win._file_browser.get_all_paths() == images
    assert win._caption_panel.get_caption() == "unsaved"


# ── Streaming to the wrong image ────────────────────────────────────────

def test_tokens_do_not_leak_onto_another_image(win, images):
    win._on_image_selected(images[0])
    win._is_generating = True
    win._caption_worker = _FakeWorker(images[0])

    win._on_new_token("a red ")
    assert win._caption_panel.get_caption() == "a red"

    # User clicks the other thumbnail mid-generation.
    win._on_image_selected(images[1])
    win._on_new_token("car on a street")

    assert win._caption_panel.get_caption() == ""
    # ...and the stream is still accumulating for its own image.
    assert win._stream_buffer == "a red car on a street"


def test_selecting_back_restores_the_partial_stream(win, images):
    win._on_image_selected(images[0])
    win._is_generating = True
    win._caption_worker = _FakeWorker(images[0])
    win._on_new_token("a red ")
    win._on_image_selected(images[1])
    win._on_new_token("car")

    win._on_image_selected(images[0])
    assert win._caption_panel.get_caption() == "a red car"


def test_save_refuses_while_generating(win, images):
    win._on_image_selected(images[0])
    win._is_generating = True
    win._caption_worker = _FakeWorker(images[0])
    win._on_new_token("partial text")

    assert win._save_current_caption() is False
    assert read_caption(images[0]).exists is False


# ── Post-generation state ───────────────────────────────────────────────

def test_generated_caption_is_not_marked_saved_until_written(win, images, monkeypatch):
    win._on_image_selected(images[0])
    win._caption_worker = _FakeWorker(images[0])
    win._is_generating = True
    monkeypatch.setattr(win._settings_panel, "get_auto_save", lambda: False)
    # Decline the save prompt.
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    win._on_caption_finished("a red car")

    assert read_caption(images[0]).exists is False
    # No green check: "done" means written to disk.
    assert win._file_browser.get_item_status(images[0]) == "generated"
    assert str(images[0]) in win._unsaved


def test_panel_shows_the_processed_caption_not_the_raw_stream(win, images, monkeypatch):
    win._on_image_selected(images[0])
    win._caption_worker = _FakeWorker(images[0])
    win._is_generating = True
    win._on_new_token("a red car")  # raw streamed text, no prefix/suffix
    monkeypatch.setattr(win._settings_panel, "get_auto_save", lambda: False)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    win._on_caption_finished("masterpiece, a red car, 8k")

    # The box now matches what was cached, so a later Save cannot strip the
    # preset's prefix/suffix back off.
    assert win._caption_panel.get_caption() == "masterpiece, a red car, 8k"
    assert win._captions[str(images[0])] == "masterpiece, a red car, 8k"


def test_close_prompts_about_unsaved_captions(win, images, monkeypatch):
    win._on_image_selected(images[0])
    win._cache_caption(images[0], "generated but never saved", saved=False)

    answers = []

    def fake_question(parent, title, text, buttons, default):
        answers.append(title)
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    class _Event:
        def __init__(self):
            self.ignored = False

        def ignore(self):
            self.ignored = True

        def accept(self):
            pass

    event = _Event()
    win.closeEvent(event)
    assert answers == ["Unsaved Captions"]
    assert event.ignored is True


# ── Write failures ──────────────────────────────────────────────────────

def test_failed_sidecar_write_is_reported_and_not_marked_done(win, images, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("gui.main_window.write_caption", boom)

    assert win._auto_save_caption(images[0], "a red car") is False
    assert win._file_browser.get_item_status(images[0]) != "done"
    # The failure reached the persistent notification store, not just a
    # transient label the next click erases.
    messages = [n.message for n in win._notification_store.entries()]
    assert any("Save failed for a.jpg" in m for m in messages)


def test_batch_reports_failures_instead_of_claiming_success(win, images, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *a, **k: shown.update(
            title=title, text=text)),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: shown.update(
            title=title, text=text)),
    )

    win._batch_index = 2
    win._batch_saved = 1
    win._batch_failed = 1
    win._on_batch_complete()

    assert shown["title"] == "Batch Finished With Errors"
    assert "1 could not be written" in shown["text"]


# ── Cache freshness ─────────────────────────────────────────────────────

def test_cache_is_revalidated_against_disk(win, images):
    caption_path(images[0]).write_text("original", encoding="utf-8")
    win._on_image_selected(images[0])
    assert win._caption_panel.get_caption() == "original"

    # Edited outside the app.
    import os
    import time
    caption_path(images[0]).write_text("edited elsewhere", encoding="utf-8")
    stat = caption_path(images[0]).stat()
    os.utime(caption_path(images[0]), (stat.st_atime, stat.st_mtime + 10))
    time.sleep(0.01)

    win._on_image_selected(images[1])
    win._on_image_selected(images[0])
    assert win._caption_panel.get_caption() == "edited elsewhere"


def test_non_utf8_sidecar_is_shown_not_treated_as_missing(win, images):
    caption_path(images[0]).write_bytes(b"caf\xe9 scene")
    win._on_image_selected(images[0])
    assert "scene" in win._caption_panel.get_caption()
    messages = [n.message for n in win._notification_store.entries()]
    assert any("not valid UTF-8" in m for m in messages)


# ── Export / batch overwrite guards ─────────────────────────────────────

def test_export_does_not_clobber_a_differing_sidecar(win, images, monkeypatch):
    caption_path(images[0]).write_text("hand written, keep me", encoding="utf-8")
    win._cache_caption(images[0], "generated, declined", saved=False)
    win._cache_caption(images[1], "brand new", saved=False)

    clicked = {}

    def fake_exec(box):
        # Default button = "write new only"; simulate accepting it.
        clicked["buttons"] = [b.text() for b in box.buttons()]
        box.setResult(0)

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(
        QMessageBox, "clickedButton",
        lambda box: box.defaultButton(),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    win._export_all_captions()

    assert read_caption(images[0]).text == "hand written, keep me"
    assert read_caption(images[1]).text == "brand new"


def test_batch_offers_to_skip_already_captioned(win, images, monkeypatch):
    caption_path(images[0]).write_text("existing", encoding="utf-8")
    monkeypatch.setattr(type(win._engine), "is_loaded", property(lambda self: True))

    labels = {}

    def fake_exec(box):
        labels["buttons"] = [b.text() for b in box.buttons()]
        box.setResult(0)

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box.defaultButton())
    # Stop before any generation actually starts.
    monkeypatch.setattr(win, "_process_next_batch_item", lambda: None)

    win._batch_caption_all()

    assert any("Skip 1 already-captioned" in b for b in labels["buttons"])
    assert win._batch_queue == [images[1]]


def test_aborted_batch_clears_queued_and_processing_badges(win, images):
    for p in images:
        win._file_browser.set_item_status(p, "queued")
    win._file_browser.set_item_status(images[0], "processing")
    win._batch_active = True
    win._batch_queue = list(images)

    win._cancel_generation()

    assert win._file_browser.get_item_status(images[0]) == "idle"
    assert win._file_browser.get_item_status(images[1]) == "idle"
    # ...and the Batch button is usable again.
    assert win._batch_active is False


# ── Model resolution ────────────────────────────────────────────────────

def test_registry_selection_never_substitutes_a_foreign_gguf(win, tmp_path, monkeypatch):
    """A not-yet-downloaded registry model must not load someone else's GGUF.

    The fallback used to return any non-mmproj GGUF on disk, so Load Model
    silently loaded an unrelated file and then paired it with the selected
    entry's vision encoder — the mismatched-mmproj crash the surrounding code
    exists to prevent.
    """
    from gui.model_download_manager import MODEL_REGISTRY

    (tmp_path / "SomeOtherModel-Q4_K_M.gguf").write_bytes(b"")
    label = next(iter(MODEL_REGISTRY))

    monkeypatch.setattr(
        win._settings_panel, "get_selected_model", lambda: ("registry", label)
    )
    monkeypatch.setattr(win, "_model_search_dirs", lambda: [tmp_path])

    assert win._find_model_file() is None


def test_unknown_selection_still_falls_back(win, tmp_path, monkeypatch):
    """A dropdown label with no registry entry keeps the old behaviour."""
    other = tmp_path / "SomeOtherModel-Q4_K_M.gguf"
    other.write_bytes(b"")

    monkeypatch.setattr(
        win._settings_panel, "get_selected_model",
        lambda: ("registry", "Not A Real Registry Label"),
    )
    monkeypatch.setattr(win, "_model_search_dirs", lambda: [tmp_path])

    assert win._find_model_file() == other


def test_cancel_confirms_before_killing_a_download(win, monkeypatch):
    """Cancel used to kill a running download outright, deleting its multi-GB
    partial, alongside the batch the user meant to stop."""
    class _Thread:
        def isRunning(self):
            return True

    win._download_thread = _Thread()
    called = []
    monkeypatch.setattr(win, "_cancel_download", lambda: called.append(True))

    win._cancel_generation()  # nothing generating, a download is running

    assert called == [True]  # routed to the path that asks first
    win._download_thread = None
