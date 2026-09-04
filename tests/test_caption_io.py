"""Tests for the shared caption-sidecar reader/writer (gui.caption_io).

Regression guard for the Project view and the Dataset tab disagreeing about
what "captioned" means: a sidecar counted purely by existence reported
empty files as captioned, and a strict UTF-8 read made a legacy-encoded
caption look absent in one view and present in the other.
"""

from gui.caption_io import (
    caption_path,
    has_caption,
    read_caption,
    write_caption,
)


def _image(tmp_path, name="pic.jpg"):
    p = tmp_path / name
    p.write_bytes(b"not-really-an-image")
    return p


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
