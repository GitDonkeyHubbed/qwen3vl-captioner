"""UI-state and import-robustness tests (Batch E of the full-repo audit).

Covers stuck states, a process abort, wrong imports, and a selector that
silently did nothing.
"""

import sys

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QUrl  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.file_browser import (  # noqa: E402
    FileBrowserPanel,
    _stem_key,
    is_importable_image,
    scan_directory,
)
from gui.settings_panel import (  # noqa: E402
    CAPTION_LENGTHS,
    SettingsPanel,
    _build_prompt_pony,
    _build_prompt_sd,
)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


# ── Caption Length was a no-op for the tag presets ──────────────────────

@pytest.mark.parametrize("builder", [_build_prompt_sd, _build_prompt_pony])
def test_caption_length_changes_the_tag_count(builder):
    prompts = {
        key: builder(key, CAPTION_LENGTHS[key], {}, "")
        for key in ("Short", "Medium", "Long")
    }
    # Short/Medium/Long all produced "Use 15-30 tags." because the builders
    # substring-matched the instruction SENTENCE, which never contains the key.
    assert len(set(prompts.values())) == 3
    assert "5-15 tags" in prompts["Short"]
    assert "15-30 tags" in prompts["Medium"]
    assert "30-50 tags" in prompts["Long"]


def test_unknown_length_key_falls_back_to_medium():
    out = _build_prompt_sd("Nonsense", "", {}, "")
    assert "15-30 tags" in out


# ── Folder import robustness ────────────────────────────────────────────

def test_appledouble_and_dotfiles_are_skipped(tmp_path):
    # `._IMG_0001.jpg` sorts FIRST and is not a decodable image, so it used to
    # abort the whole batch on item 1.
    for name in ("._IMG_0001.jpg", ".DS_Store", "IMG_0001.jpg", "b.png"):
        (tmp_path / name).write_bytes(b"")

    found = [p.name for p in scan_directory(tmp_path)]
    assert found == ["IMG_0001.jpg", "b.png"]
    assert is_importable_image(tmp_path / "._IMG_0001.jpg") is False
    assert is_importable_image(tmp_path / "IMG_0001.jpg") is True


def test_unreadable_folder_is_reported_not_raised(qapp, tmp_path, monkeypatch):
    """An OSError inside a Qt slot aborts the whole PyQt6 process."""
    panel = FileBrowserPanel()
    errors = []
    panel.import_failed.connect(errors.append)

    def boom(self):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr("pathlib.Path.iterdir", boom)
    panel.import_directory(tmp_path)  # must not raise

    assert len(errors) == 1
    assert "Operation not permitted" in errors[0]
    panel.deleteLater()


def test_non_local_urls_are_skipped(qapp, tmp_path, monkeypatch):
    """A browser-tab image drop yielded Path(""), whose is_dir() is True for
    the process working directory — importing every image in it."""
    panel = FileBrowserPanel()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "unrelated.jpg").write_bytes(b"")

    class _Mime:
        def hasUrls(self):
            return True

        def urls(self):
            return [QUrl("https://example.com/photo.jpg")]

    class _Event:
        def __init__(self):
            self.ignored = False

        def mimeData(self):
            return _Mime()

        def ignore(self):
            self.ignored = True

        def acceptProposedAction(self):
            pass

    event = _Event()
    panel.dropEvent(event)

    assert event.ignored is True
    assert panel.get_all_paths() == []
    panel.deleteLater()


@pytest.mark.parametrize(
    "a, b",
    [("Photo.jpg", "photo.png"), ("IMG_1.JPG", "img_1.webp")],
)
def test_case_only_stem_collisions_are_detected(tmp_path, a, b):
    """Photo.jpg and photo.png share ONE photo.txt on Windows and macOS."""
    if sys.platform not in ("win32", "darwin"):
        pytest.skip("case-insensitive filesystems only")
    assert _stem_key(tmp_path / a) == _stem_key(tmp_path / b)


def test_distinct_stems_do_not_collide(tmp_path):
    assert _stem_key(tmp_path / "one.jpg") != _stem_key(tmp_path / "two.jpg")


# ── Busy state ──────────────────────────────────────────────────────────

def test_caption_finishing_does_not_re_enable_download(qapp):
    """A caption finishing during a download used to hide Cancel and re-enable
    Download, letting a second click start a duplicate download."""
    panel = SettingsPanel()

    panel.set_download_in_progress(True)
    panel.set_generating(True)
    panel.set_generating(False)          # caption finishes mid-download

    assert panel._download_btn.isEnabled() is False
    # isHidden(), not isVisible(): the panel itself is never shown here, and
    # isVisible() is False for every child of an unshown parent.
    assert panel.cancel_btn.isHidden() is False

    panel.set_download_in_progress(False)
    assert panel._download_btn.isEnabled() is True
    assert panel.cancel_btn.isHidden() is True
    panel.deleteLater()


def test_model_status_does_not_clobber_busy_state(qapp):
    panel = SettingsPanel()
    panel.set_download_in_progress(True)
    panel.set_model_status("Downloading...")
    assert panel.load_model_btn.isEnabled() is False
    panel.deleteLater()


def test_batch_button_resets_at_zero_total(qapp):
    panel = SettingsPanel()
    panel.set_batch_progress(3, 10)
    assert panel.batch_btn.isEnabled() is False
    assert "3/10" in panel.batch_btn.text()

    panel.set_batch_progress(0, 0)
    assert panel.batch_btn.isEnabled() is True
    assert "Batch Caption All" in panel.batch_btn.text()
    panel.deleteLater()
