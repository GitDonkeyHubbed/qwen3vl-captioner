"""
Shared reader/writer for `.txt` caption sidecars.

Every place that touches a sidecar goes through here so the Project view, the
Dataset tab and the in-memory caption cache agree on two things they used to
disagree about:

  * **What "captioned" means.** Counting a sidecar by `Path.exists()` alone
    reported empty and whitespace-only `.txt` files as captioned, inflating
    dataset coverage to 100% with blank previews.
  * **How a non-UTF-8 sidecar decodes.** A strict `utf-8` read raised in the
    Project view (showing the image as uncaptioned, ready to be overwritten)
    while the Dataset tab's lossy read counted the same file as captioned.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional


def caption_path(image_path: Path) -> Path:
    """Return the `.txt` sidecar path for an image."""
    return Path(image_path).with_suffix(".txt")


class CaptionFile(NamedTuple):
    """The result of reading one caption sidecar."""

    text: str
    """Decoded and stripped caption text; empty when absent or unreadable."""

    exists: bool
    """True when the sidecar file is present on disk."""

    mtime: Optional[float]
    """Sidecar modification time, or None when it does not exist."""

    decode_error: bool
    """True when the bytes were not valid UTF-8 and were decoded lossily."""

    read_error: Optional[str]
    """The OS error text when the file exists but could not be read."""

    @property
    def has_caption(self) -> bool:
        """True only when the sidecar holds actual text."""
        return bool(self.text)


_MISSING = CaptionFile(
    text="", exists=False, mtime=None, decode_error=False, read_error=None
)


def read_caption(image_path: Path) -> CaptionFile:
    """Read the caption sidecar for *image_path*. Never raises."""
    path = caption_path(image_path)
    try:
        raw = path.read_bytes()
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return _MISSING
    except OSError as e:
        return CaptionFile(
            text="", exists=True, mtime=None, decode_error=False, read_error=str(e)
        )

    # utf-8-sig strips a BOM that a Windows text editor may have written; the
    # lossy retry keeps a legacy-encoded sidecar visible (and therefore
    # protected from being silently overwritten) rather than dropping it.
    try:
        text = raw.decode("utf-8-sig")
        decode_error = False
    except UnicodeDecodeError:
        text = raw.decode("utf-8-sig", errors="replace")
        decode_error = True

    return CaptionFile(
        text=text.strip(),
        exists=True,
        mtime=mtime,
        decode_error=decode_error,
        read_error=None,
    )


def has_caption(image_path: Path) -> bool:
    """True when *image_path* has a sidecar containing non-blank text."""
    return read_caption(image_path).has_caption


def write_caption(image_path: Path, text: str) -> float:
    """Write *text* to the sidecar for *image_path*; return its new mtime.

    Propagates OSError so callers can report a failed save instead of
    assuming it succeeded.
    """
    path = caption_path(image_path)
    path.write_text(text, encoding="utf-8")
    return path.stat().st_mtime
