"""UI-state and import-robustness tests (Batch E of the full-repo audit).

Covers stuck states, a process abort, wrong imports, and a selector that
silently did nothing.
"""

import os
import stat
import sys
import types

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

    # Order-independent: this is about filtering, and WindowsPath sorts
    # case-insensitively where PosixPath does not.
    found = sorted(p.name for p in scan_directory(tmp_path))
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


# ── Performance-shaped behaviour (Batch F) ──────────────────────────────

def test_thumbnails_are_not_decoded_on_the_ui_thread(qapp, tmp_path):
    """Importing used to fully decode every image inline, freezing the window
    for minutes on a PNG/WebP dataset (the scaled-decode shortcut only helps
    formats whose decoder supports one)."""
    from PIL import Image

    paths = []
    for i in range(8):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (900, 700), (i * 20, 80, 160)).save(p)
        paths.append(p)

    panel = FileBrowserPanel()
    panel.add_images(paths)

    # Rows exist immediately, thumbnails have not been installed yet.
    assert len(panel.get_all_paths()) == 8
    assert all(not item._thumb_loaded for item in panel._items.values())

    panel._thumb_pool.waitForDone(30000)
    qapp.processEvents()
    assert all(item._thumb_loaded for item in panel._items.values())
    panel.deleteLater()


def test_zoom_uses_a_view_transform_not_a_rescale():
    """Re-scaling the full-resolution pixmap per wheel notch allocated
    hundreds of MB and stalled for seconds on a camera photo."""
    import inspect

    from gui.image_viewer import ImageViewer
    src = inspect.getsource(ImageViewer._apply_zoom)
    assert "setTransform" in src
    assert ".scaled(" not in src


def test_spinner_timer_is_not_started_at_construction(qapp):
    """The spinner woke the event loop 33x a second from launch, hidden."""
    from gui.image_viewer import ProcessingOverlay

    overlay = ProcessingOverlay()
    assert overlay._spinner._timer.isActive() is False

    overlay.show_overlay()
    assert overlay._spinner._timer.isActive() is True

    overlay.hide_overlay()
    assert overlay._spinner._timer.isActive() is False
    overlay.deleteLater()


# ── Parallel download pre-allocation ────────────────────────────────────

@pytest.mark.skipif(os.name == "nt", reason="NTFS behaviour covered by the Windows tests")
def test_make_sparse_is_a_safe_noop_off_windows(tmp_path):
    from gui.model_download_manager import _make_sparse

    path = tmp_path / "x.part"
    with open(path, "wb") as f:
        assert _make_sparse(f) is False  # POSIX already creates holes
        f.truncate(1024)
    assert path.stat().st_size == 1024


def test_make_sparse_posix_branch_is_a_noop(tmp_path, monkeypatch):
    # Runs everywhere: force the non-Windows branch even on Windows.
    import gui.model_download_manager as mdm

    monkeypatch.setattr(mdm, "os", types.SimpleNamespace(name="posix"))
    with open(tmp_path / "x.part", "wb") as f:
        assert mdm._make_sparse(f) is False


@pytest.mark.skipif(os.name != "nt", reason="FSCTL_SET_SPARSE is Windows-only")
def test_make_sparse_sets_the_sparse_attribute_on_windows(tmp_path):
    from gui.model_download_manager import _make_sparse

    path = tmp_path / "x.part"
    with open(path, "wb") as f:
        if not _make_sparse(f):
            pytest.skip("temp volume does not support sparse files")
    assert os.stat(path).st_file_attributes & stat.FILE_ATTRIBUTE_SPARSE_FILE


def _allocated_bytes(path) -> int:
    """Bytes actually allocated on disk for *path* (Windows only)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCompressedFileSizeW.restype = wintypes.DWORD
    kernel32.GetCompressedFileSizeW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    high = wintypes.DWORD()
    low = kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value << 32) | low


def test_preallocate_extends_to_the_full_size(tmp_path):
    from gui.model_download_manager import _preallocate

    path = tmp_path / "m.gguf.part"
    with open(path, "wb") as f:
        _preallocate(f, 4096)
    assert path.read_bytes() == b"\0" * 4096


@pytest.mark.skipif(os.name != "nt", reason="the zero-writing truncate() is Windows-only")
def test_preallocate_does_not_write_the_body_on_windows(tmp_path):
    """CPython's truncate() goes through the CRT's _chsize_s on Windows, which
    writes every byte as zeros even on a sparse file — an 8 GB model wrote
    ~8 GB of zeros before the download started."""
    from gui.model_download_manager import _make_sparse, _preallocate

    # Probe the volume with a separate file: checking the file under test
    # would also skip when _preallocate itself forgot to make it sparse.
    with open(tmp_path / "probe", "wb") as probe:
        if not _make_sparse(probe):
            pytest.skip("temp volume does not support sparse files")

    total = 64 * 1024 * 1024
    path = tmp_path / "m.gguf.part"
    with open(path, "wb") as f:
        _preallocate(f, total)
        f.flush()
        os.fsync(f.fileno())  # allocation is only reported once cached writes land
    assert os.stat(path).st_file_attributes & stat.FILE_ATTRIBUTE_SPARSE_FILE
    assert path.stat().st_size == total
    assert _allocated_bytes(path) < total // 16


def test_parallel_download_overwrites_the_preallocated_last_byte(tmp_path, monkeypatch):
    """_preallocate writes a placeholder final byte; the last range must
    replace it so the finished file is byte-identical to the source."""
    import urllib.request

    from gui.model_download_manager import ModelDownloadWorker

    payload = bytes(range(256)) * 4096 + b"\x01\x02\xab"  # last byte non-zero
    total = len(payload)

    class FakeResponse:
        status = 206

        def __init__(self, body):
            self._body = body

        def read(self, n):
            chunk, self._body = self._body[:n], self._body[n:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeOpener:
        def open(self, request, timeout=None):
            start, end = request.get_header("Range").split("=", 1)[1].split("-")
            return FakeResponse(payload[int(start):int(end) + 1])

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: FakeOpener())

    worker = ModelDownloadWorker(
        repo_id="owner/repo", filename="m.gguf", target_dir=tmp_path,
        max_connections=4,
    )
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    target = tmp_path / "m.gguf"
    part = tmp_path / "m.gguf.part"
    worker._run_parallel("https://example.invalid/m.gguf", target, part, total)

    assert errors == []
    assert finished == [str(target)]
    assert target.read_bytes() == payload
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()
