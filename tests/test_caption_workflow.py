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

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
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
    # No event loop runs here, so deliver the deletion now: leaked windows
    # otherwise pile up and slow every later app-wide setStyleSheet().
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


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


def _answer(monkeypatch, button, asked=None):
    """Stub QMessageBox.question to return *button*, recording titles."""
    def fake_question(parent, title, *a, **k):
        if asked is not None:
            asked.append(title)
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))


def test_trash_button_is_an_edit_and_saving_removes_the_sidecar(win, images, monkeypatch):
    # The trash button used to call the programmatic clear, which is never
    # dirty: the next selection silently reloaded the caption from disk.
    caption_path(images[0]).write_text("bad caption", encoding="utf-8")
    win._on_image_selected(images[0])

    win._caption_panel.delete_btn.click()
    assert win._caption_panel.get_caption() == ""
    assert win._caption_panel.is_dirty()

    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.Save, asked)
    win._on_image_selected(images[1])

    assert asked == ["Unsaved Caption"]
    assert win._current_image == images[1]  # Save did not block navigation
    assert read_caption(images[0]).exists is False
    assert str(images[0]) not in win._captions
    assert win._file_browser.get_item_status(images[0]) == "idle"

    win._on_image_selected(images[0])
    assert win._caption_panel.get_caption() == ""


def test_saving_a_hand_emptied_box_does_not_trap_the_user(win, images, monkeypatch):
    # "Nothing to save" made the prompt's default Save button fail, so the
    # user could only leave by discarding.
    caption_path(images[0]).write_text("bad caption", encoding="utf-8")
    win._on_image_selected(images[0])
    text = win._caption_panel.caption_text
    text.selectAll()
    text.textCursor().removeSelectedText()

    _answer(monkeypatch, QMessageBox.StandardButton.Save)
    win._on_image_selected(images[1])

    assert win._current_image == images[1]
    assert read_caption(images[0]).exists is False


def test_browsing_uncaptioned_images_never_prompts(win, images, monkeypatch):
    # Guards the app's own clear_caption() staying clean.
    caption_path(images[0]).write_text("good caption", encoding="utf-8")
    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel, asked)

    win._on_image_selected(images[0])
    win._on_image_selected(images[1])  # uncaptioned: box cleared by the app
    win._caption_panel.delete_btn.click()  # already empty: not an edit
    win._on_image_selected(images[0])

    assert asked == []
    assert win._current_image == images[0]
    assert read_caption(images[0]).text == "good caption"


def test_save_on_an_untouched_empty_box_writes_nothing(win, images):
    win._on_image_selected(images[1])
    assert win._save_current_caption() is False
    assert read_caption(images[1]).exists is False


def test_clearing_an_unsaved_generated_caption_asks_before_deleting_the_file(
    win, images, monkeypatch
):
    # The box showed a generated caption the user declined to save; the file
    # on disk holds a caption they never saw there, so don't just delete it.
    caption_path(images[0]).write_text("good caption on disk", encoding="utf-8")
    win._on_image_selected(images[0])
    win._caption_worker = _FakeWorker(images[0])
    win._is_generating = True
    monkeypatch.setattr(win._settings_panel, "get_auto_save", lambda: False)
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    win._on_caption_finished("generated, declined")
    win._caption_panel.delete_btn.click()

    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.Yes, asked)
    assert win._save_current_caption() is True
    assert asked == ["Delete Caption File"]
    assert read_caption(images[0]).exists is False
    assert str(images[0]) not in win._unsaved


def test_keeping_the_file_drops_the_cleared_generated_caption(
    win, images, monkeypatch
):
    # "No" used to fail the save, so each navigation re-asked both questions
    # (default buttons looped forever) and the rejected caption came back.
    caption_path(images[0]).write_text("good caption on disk", encoding="utf-8")
    win._on_image_selected(images[0])
    win._caption_worker = _FakeWorker(images[0])
    win._is_generating = True
    monkeypatch.setattr(win._settings_panel, "get_auto_save", lambda: False)
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    win._on_caption_finished("generated, declined")
    win._caption_panel.delete_btn.click()

    asked = []

    def default_button(parent, title, text, buttons, default):
        asked.append(title)
        return default  # Save on "Unsaved Caption", No on "Delete Caption File"

    monkeypatch.setattr(QMessageBox, "question", staticmethod(default_button))
    win._on_image_selected(images[1])

    assert asked == ["Unsaved Caption", "Delete Caption File"]
    assert win._current_image == images[1]
    assert read_caption(images[0]).text == "good caption on disk"
    assert str(images[0]) not in win._unsaved
    assert win._file_browser.get_item_status(images[0]) == "done"

    asked.clear()
    win._on_image_selected(images[0])
    assert win._caption_panel.get_caption() == "good caption on disk"
    assert not win._caption_panel.is_dirty()
    assert asked == []


@pytest.mark.parametrize("error", ["cancelled", "RuntimeError: decode failed"])
def test_a_stopped_stream_is_not_left_in_the_box(win, images, error):
    # A cancelled or failed stream left its fragment in the box, uncached.
    # Trash + Save then deleted the good sidecar without the "Delete Caption
    # File" question, since the key was never in _unsaved.
    caption_path(images[0]).write_text("good caption on disk", encoding="utf-8")
    win._on_image_selected(images[0])
    _start_fake_generation(win, images[0])
    win._on_new_token("a blurry ph")
    win._on_caption_error(error)

    # The box shows the image's own caption again, so what trash + Save
    # removes is what the user is looking at, and Save can't write the
    # fragment over the file.
    assert win._caption_panel.get_caption() == "good caption on disk"
    assert not win._caption_panel.is_dirty()
    win._caption_panel.save_btn.click()
    assert read_caption(images[0]).text == "good caption on disk"


def test_delete_button_marks_empty_caption_dirty(win, images):
    caption_path(images[0]).write_text("existing", encoding="utf-8")
    win._on_image_selected(images[0])

    win._caption_panel.delete_btn.click()

    assert win._caption_panel.get_caption() == ""
    assert win._caption_panel.is_dirty()


def test_saving_cleared_caption_removes_sidecar(win, images):
    caption_path(images[0]).write_text("existing", encoding="utf-8")
    win._on_image_selected(images[0])
    win._caption_panel.delete_btn.click()

    assert win._save_current_caption() is True
    assert caption_path(images[0]).exists() is False
    assert win._caption_panel.is_dirty() is False
    assert win._file_browser.get_item_status(images[0]) == "idle"


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


def _start_fake_generation(win, image):
    """Put the window in the state _generate_caption leaves it in."""
    win._is_generating = True
    win._caption_worker = _FakeWorker(image)
    win._stream_buffer = ""
    win._caption_panel.clear_caption()
    win._caption_panel.set_generating(True)


def test_caption_box_rejects_typing_while_streaming(win, images, monkeypatch):
    # Keystrokes used to land between tokens, the next token reset the dirty
    # flag, and the finished caption replaced the edit without a prompt.
    win._on_image_selected(images[0])
    _start_fake_generation(win, images[0])
    panel = win._caption_panel

    win._on_new_token("a red ")
    QTest.keyClicks(panel.caption_text, "HAND EDIT")
    win._on_new_token("car")

    assert panel.get_caption() == "a red car"
    assert not panel.is_dirty()
    assert not panel.delete_btn.isEnabled()

    monkeypatch.setattr(win._settings_panel, "get_auto_save", lambda: False)
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    win._on_caption_finished("a red car")

    # Editable again once generation ends.
    QTest.keyClicks(panel.caption_text, "!")
    assert "!" in panel.get_caption()
    assert panel.is_dirty()
    assert panel.delete_btn.isEnabled()


def test_another_images_box_rejects_typing_during_generation(win, images):
    # Typed into image B while A generated: Save refused it, and A finishing
    # wiped B's box back to its (empty) caption.
    win._on_image_selected(images[0])
    _start_fake_generation(win, images[0])
    win._on_image_selected(images[1])

    QTest.keyClicks(win._caption_panel.caption_text, "caption for b")
    assert win._caption_panel.get_caption() == ""
    assert not win._caption_panel.is_dirty()

    # A cancel (reported through the error path) makes the box editable again.
    win._on_caption_error("Generation cancelled")
    QTest.keyClicks(win._caption_panel.caption_text, "caption for b")
    assert win._caption_panel.get_caption() == "caption for b"


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


def test_batch_start_offers_to_save_a_hand_edit(win, images, monkeypatch):
    # The batch selects its first image with _batch_active already set, which
    # skips the per-selection prompt: the edit was silently wiped.
    monkeypatch.setattr(type(win._engine), "is_loaded", property(lambda self: True))
    monkeypatch.setattr(win, "_start_deferred_batch_caption", lambda: None)
    win._on_image_selected(images[1])
    _type(win, "hand typed, never saved")

    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.Save, asked)
    seen = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: seen.append(box.windowTitle()))
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box.defaultButton())

    win._batch_caption_all()

    assert asked == ["Unsaved Caption"]
    assert read_caption(images[1]).text == "hand typed, never saved"
    # Asked before counting sidecars, so the saved edit is already-captioned
    # and the default "Skip" keeps it out of the batch.
    assert seen == ["Existing Captions Found"]
    assert images[1] not in win._batch_queue
    assert win._batch_current_path == images[0]


def test_cancelling_the_unsaved_prompt_does_not_start_a_batch(win, images, monkeypatch):
    monkeypatch.setattr(type(win._engine), "is_loaded", property(lambda self: True))
    monkeypatch.setattr(win, "_start_deferred_batch_caption", lambda: None)
    win._on_image_selected(images[1])
    _type(win, "wip")

    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    win._batch_caption_all()

    assert win._batch_active is False
    assert win._current_image == images[1]
    assert win._caption_panel.get_caption() == "wip"
    assert win._caption_panel.is_dirty()


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
