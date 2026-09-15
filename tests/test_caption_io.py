"""Tests for the shared caption-sidecar reader/writer (gui.caption_io).

Regression guard for the Project view and the Dataset tab disagreeing about
what "captioned" means: a sidecar counted purely by existence reported
empty files as captioned, and a strict UTF-8 read made a legacy-encoded
caption look absent in one view and present in the other.
"""

import os
import subprocess
import sys

import pytest

import gui.caption_io as caption_io
from gui.caption_io import (
    caption_path,
    delete_caption,
    has_caption,
    read_caption,
    write_caption,
)

OLD = "hand-edited caption\nsecond line"


def _image(tmp_path, name="pic.jpg"):
    p = tmp_path / name
    p.write_bytes(b"not-really-an-image")
    return p


def _captioned(tmp_path):
    img = _image(tmp_path)
    caption_path(img).write_bytes(OLD.encode("utf-8"))
    return img


def _names(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


def test_caption_path_swaps_extension(tmp_path):
    assert caption_path(tmp_path / "a.jpeg").name == "a.txt"


def test_missing_sidecar(tmp_path):
    info = read_caption(_image(tmp_path))
    assert info.exists is False
    assert info.text == ""
    assert info.has_caption is False
    assert info.mtime is None


def test_reads_and_strips(tmp_path):
    img = _image(tmp_path)
    caption_path(img).write_text("  a red car \n", encoding="utf-8")
    info = read_caption(img)
    assert info.text == "a red car"
    assert info.has_caption is True
    assert info.decode_error is False
    assert info.mtime is not None


def test_bom_is_stripped(tmp_path):
    img = _image(tmp_path)
    caption_path(img).write_bytes("﻿a red car".encode("utf-8"))
    info = read_caption(img)
    assert info.text == "a red car"
    assert info.decode_error is False


def test_blank_sidecar_is_not_a_caption(tmp_path):
    # An empty/whitespace-only .txt used to report "Yes" in the Dataset tab
    # with a blank preview, inflating Coverage to 100%.
    img = _image(tmp_path)
    caption_path(img).write_text("   \n\t\n", encoding="utf-8")
    info = read_caption(img)
    assert info.exists is True
    assert info.has_caption is False
    assert has_caption(img) is False


def test_non_utf8_is_read_lossily_and_flagged(tmp_path):
    # latin-1 bytes that are not valid UTF-8: the file must still be visible
    # (so it is not treated as uncaptioned and overwritten), and flagged.
    img = _image(tmp_path)
    caption_path(img).write_bytes(b"caf\xe9 scene")
    info = read_caption(img)
    assert info.exists is True
    assert info.decode_error is True
    assert info.has_caption is True
    assert "scene" in info.text


def test_write_caption_returns_mtime(tmp_path):
    img = _image(tmp_path)
    mtime = write_caption(img, "a red car")
    assert caption_path(img).read_text(encoding="utf-8") == "a red car"
    assert read_caption(img).mtime == mtime


def test_read_error_is_reported_not_raised(tmp_path):
    img = _image(tmp_path)
    # A directory where the sidecar should be: exists, but unreadable.
    caption_path(img).mkdir()
    info = read_caption(img)
    assert info.exists is True
    assert info.read_error is not None
    assert info.has_caption is False


# ── Atomic write ────────────────────────────────────────────────────────

def test_failed_write_keeps_previous_caption(tmp_path):
    # Writing in place truncated the sidecar before encoding, so this left a
    # 0-byte file that reads as "uncaptioned".
    img = _captioned(tmp_path)
    with pytest.raises(UnicodeEncodeError):
        write_caption(img, "new \ud800 caption")
    assert read_caption(img).text == OLD
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]  # temp file cleaned up


def test_write_bytes_match_the_old_format(tmp_path):
    # UTF-8, no BOM, "\n" translated to the platform newline — as write_text.
    img = _captioned(tmp_path)
    mtime = write_caption(img, "a\nb é")
    assert caption_path(img).read_bytes() == ("a" + os.linesep + "b é").encode("utf-8")
    assert read_caption(img).mtime == mtime
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


def test_transient_lock_is_retried(tmp_path, monkeypatch):
    img = _captioned(tmp_path)
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(13, "held open by another process")
        real_replace(src, dst)

    sleeps = []
    monkeypatch.setattr(caption_io, "_WINDOWS", True)
    monkeypatch.setattr(caption_io.os, "replace", flaky_replace)
    monkeypatch.setattr(caption_io.time, "sleep", sleeps.append)

    write_caption(img, "saved after a retry")

    assert read_caption(img).text == "saved after a retry"
    assert sleeps == [0.05, 0.1]
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


def _always_locked(monkeypatch):
    def locked(src, dst):
        raise PermissionError(13, "held open by another process")

    monkeypatch.setattr(caption_io, "_WINDOWS", True)
    monkeypatch.setattr(caption_io.os, "replace", locked)
    monkeypatch.setattr(caption_io.time, "sleep", lambda s: None)


def test_persistent_lock_falls_back_to_an_in_place_write(tmp_path, monkeypatch):
    img = _captioned(tmp_path)
    _always_locked(monkeypatch)

    write_caption(img, "short")

    # Overwritten and cut to the new length, not left with the old tail.
    assert caption_path(img).read_bytes() == b"short"
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


def test_failed_in_place_fallback_restores_the_old_caption(tmp_path, monkeypatch):
    img = _captioned(tmp_path)
    _always_locked(monkeypatch)
    real_write_all = caption_io._write_all
    calls = []

    def disk_full_once(f, data):
        calls.append(data)
        if len(calls) == 1:
            f.write(data[:3])  # a partial write, then the disk fills up
            raise OSError(28, "No space left on device")
        real_write_all(f, data)

    monkeypatch.setattr(caption_io, "_write_all", disk_full_once)

    with pytest.raises(OSError):
        write_caption(img, "a much longer replacement caption than before")

    assert read_caption(img).text == OLD
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing semantics")
def test_save_succeeds_while_another_process_holds_the_sidecar(tmp_path):
    # A plain temp-file + os.replace fails here (WinError 5), while the old
    # in-place write worked — synced and indexed folders hit this.
    img = _captioned(tmp_path)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; f = open(sys.argv[1], 'rb'); "
         "print('open', flush=True); time.sleep(10)",
         str(caption_path(img))],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "open"
        write_caption(img, "saved while held")
    finally:
        holder.kill()
        holder.wait()
    assert read_caption(img).text == "saved while held"
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute")
def test_read_only_sidecar_fails_fast_and_intact(tmp_path, monkeypatch):
    img = _captioned(tmp_path)
    p = caption_path(img)
    # Count the retry back-off sleeps rather than timing the call, which a
    # slow runner can push past any wall-clock bound.
    sleeps = []
    monkeypatch.setattr(caption_io.time, "sleep", sleeps.append)
    subprocess.run(["attrib", "+r", str(p)], check=True)
    try:
        with pytest.raises(PermissionError):
            write_caption(img, "new")
        assert sleeps == []  # no pointless retries
    finally:
        subprocess.run(["attrib", "-r", str(p)], check=True)
    assert read_caption(img).text == OLD
    assert _names(tmp_path) == ["pic.jpg", "pic.txt"]


def test_long_sidecar_name_still_saves(tmp_path):
    # The temp name was the sidecar name plus 14 characters, so a 242-255
    # character name failed every save that the in-place write handled.
    try:
        img = _image(tmp_path, "x" * 241 + ".jpg")  # sidecar name: 245 chars
    except OSError:
        pytest.skip("filesystem rejects the long name itself")
    write_caption(img, "first")
    write_caption(img, "second")
    assert read_caption(img).text == "second"
    assert len(list(tmp_path.iterdir())) == 2  # no temp file left behind


def test_save_without_a_temp_file_writes_in_place(tmp_path, monkeypatch):
    # A folder that allows editing the sidecar but not creating files (or a
    # path at the length limit) can't hold the temp file.
    def cannot_create(path):
        raise PermissionError(13, "cannot create files here")

    monkeypatch.setattr(caption_io, "_create_temp", cannot_create)

    img = _captioned(tmp_path)
    mtime = write_caption(img, "a\nshort")
    written = ("a" + os.linesep + "short").encode("utf-8")
    assert caption_path(img).read_bytes() == written  # cut to the new length
    assert read_caption(img).mtime == mtime

    new = _image(tmp_path, "new.jpg")
    write_caption(new, "brand new")
    assert read_caption(new).text == "brand new"

    with pytest.raises(UnicodeEncodeError):
        write_caption(img, "bad \ud800")
    assert caption_path(img).read_bytes() == written
    assert _names(tmp_path) == ["new.jpg", "new.txt", "pic.jpg", "pic.txt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_sidecar_permissions_are_not_narrowed(tmp_path):
    img = _image(tmp_path)
    write_caption(img, "new file")
    assert caption_path(img).stat().st_mode & 0o044  # not mkstemp's 0600

    os.chmod(caption_path(img), 0o640)
    write_caption(img, "rewritten")
    assert caption_path(img).stat().st_mode & 0o777 == 0o640


# ── Delete ──────────────────────────────────────────────────────────────

def test_delete_caption_removes_the_sidecar(tmp_path):
    img = _captioned(tmp_path)
    delete_caption(img)
    assert read_caption(img).exists is False
    delete_caption(img)  # already gone: not an error


def test_delete_retries_a_transient_lock(tmp_path, monkeypatch):
    img = _captioned(tmp_path)
    real_unlink = os.unlink
    calls = []

    def flaky_unlink(path):
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError(13, "held open by another process")
        real_unlink(path)

    sleeps = []
    monkeypatch.setattr(caption_io, "_WINDOWS", True)
    monkeypatch.setattr(caption_io.os, "unlink", flaky_unlink)
    monkeypatch.setattr(caption_io.time, "sleep", sleeps.append)

    delete_caption(img)

    assert read_caption(img).exists is False
    assert sleeps == [0.05, 0.1]


def test_delete_gives_up_on_a_persistent_lock(tmp_path, monkeypatch):
    img = _captioned(tmp_path)

    def locked(path):
        raise PermissionError(13, "held open by another process")

    monkeypatch.setattr(caption_io, "_WINDOWS", True)
    monkeypatch.setattr(caption_io.os, "unlink", locked)
    monkeypatch.setattr(caption_io.time, "sleep", lambda s: None)

    with pytest.raises(PermissionError):
        delete_caption(img)
    assert read_caption(img).text == OLD  # kept, not truncated


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing semantics")
def test_delete_succeeds_after_a_brief_hold(tmp_path):
    # Save waits out a sync client or indexer holding the sidecar; clearing
    # a caption failed at once with WinError 32 in the same folders.
    img = _captioned(tmp_path)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; f = open(sys.argv[1], 'rb'); "
         "print('open', flush=True); time.sleep(0.2); f.close()",
         str(caption_path(img))],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "open"
        delete_caption(img)
    finally:
        holder.kill()
        holder.wait()
    assert read_caption(img).exists is False


def test_failed_atomic_replace_preserves_existing_sidecar(tmp_path, monkeypatch):
    img = _image(tmp_path)
    sidecar = caption_path(img)
    sidecar.write_text("keep me", encoding="utf-8")

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(caption_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_caption(img, "new caption")

    assert sidecar.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.glob(f".{sidecar.name}.*.tmp")) == []
