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

import contextlib
import os
import secrets
import stat
import time
from pathlib import Path
from typing import NamedTuple, Optional

_WINDOWS = os.name == "nt"

# os.replace onto a sidecar another process holds open without delete sharing
# (OneDrive/Dropbox sync, the search indexer, antivirus, a second scan) fails
# on Windows with PermissionError, although an in-place write would succeed.
# Such holds are usually brief, so retry for about 0.75 s before falling back.
_REPLACE_BACKOFF = (0.05, 0.1, 0.2, 0.4)


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

    The new caption goes to a temp file in the same folder and only replaces
    the sidecar once it is fully on disk. Writing in place truncated the old
    caption first, so a failed encode, a full disk or a crash left a 0-byte
    file, which reads as "uncaptioned" and invites a batch to overwrite it.
    The bytes match the old `Path.write_text(text, encoding="utf-8")`: no BOM,
    "\\n" translated to os.linesep.

    Propagates OSError (and UnicodeEncodeError) so callers can report a failed
    save instead of assuming it succeeded.
    """
    path = caption_path(image_path)
    try:
        tmp, f = _create_temp(path)
    except FileExistsError:
        raise
    except OSError:
        # No new file can be made next to the sidecar (the path is too long,
        # or the folder denies creating files while the sidecar itself is
        # writable), where the old in-place write still worked.
        return _write_without_temp(path, text)
    try:
        with f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if not _WINDOWS:
            # Keep an existing sidecar's permissions, as an in-place write did.
            with contextlib.suppress(FileNotFoundError):
                os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        _replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return path.stat().st_mtime


def delete_caption(image_path: Path) -> None:
    """Remove the sidecar for *image_path*; a missing one is not an error.

    A blank sidecar already reads as "uncaptioned", so a caption the user
    cleared is stored as no file at all rather than an empty one (which
    trainers would take as a real, empty caption). Propagates OSError.
    """
    path = caption_path(image_path)
    delays = iter(_REPLACE_BACKOFF)
    while True:
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            # The same brief sharing holds that _replace waits out block a
            # delete on Windows too. A read-only sidecar won't improve.
            if not _WINDOWS or (path.exists() and not os.access(path, os.W_OK)):
                raise
            delay = next(delays, None)
            if delay is None:
                raise
            time.sleep(delay)


def _create_temp(path: Path):
    """Exclusively create the temp file for *path*; return (tmp, text file).

    The name starts with a dot and ends in `.tmp`, so neither the file
    browser (which skips dot-files) nor a scan for images or `.txt` sidecars
    picks it up. Only the start of the sidecar name is kept, so a long one
    can't push the temp name past the 255-character limit. Mode "x" applies
    the umask like the old write did, unlike mkstemp's owner-only 0600.
    """
    for _ in range(10):
        tmp = path.with_name(f".{path.name[:40]}.{secrets.token_hex(4)}.tmp")
        try:
            return tmp, open(tmp, "x", encoding="utf-8")
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a temp file next to {path}")


def _replace(tmp: Path, path: Path) -> None:
    """Move *tmp* over *path*, tolerating a brief lock on Windows."""
    delays = iter(_REPLACE_BACKOFF)
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Off Windows this is a real permission problem, and a read-only
            # sidecar fails an in-place write too: neither improves by waiting.
            if not _WINDOWS or (path.exists() and not os.access(path, os.W_OK)):
                raise
            delay = next(delays, None)
            if delay is None:
                if not path.exists():
                    raise
                break
            time.sleep(delay)

    # Still held open by a reader that allows writes but not a rename: fall
    # back to writing in place, as saves did before, with the bytes already
    # encoded and on disk in tmp.
    _overwrite_in_place(path, tmp.read_bytes())


def _write_without_temp(path: Path, text: str) -> float:
    """Write *text* straight to *path* (same bytes as write_caption); return mtime.

    Encoding happens first, so a bad character fails before the file is
    touched, and an existing caption is kept safe by _overwrite_in_place.
    """
    data = text.replace("\n", os.linesep).encode("utf-8")
    try:
        _overwrite_in_place(path, data)
    except FileNotFoundError:
        with open(path, "xb", buffering=0) as f:
            _write_all(f, data)
            os.fsync(f.fileno())
    return path.stat().st_mtime


def _overwrite_in_place(path: Path, data: bytes) -> None:
    """Overwrite *path* with *data* without truncating it up front.

    Opening with "wb" would empty the file before a byte is written. Here the
    old bytes are only cut off after the new ones are written, and put back
    if the write fails, so an I/O error leaves the previous caption readable.
    """
    # Unbuffered, so a failed write can't sit in a buffer and fail again on
    # the seek/close that the restore needs.
    with open(path, "r+b", buffering=0) as f:
        old = f.readall()
        try:
            f.seek(0)
            _write_all(f, data)
            f.truncate()
            os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                f.seek(0)
                _write_all(f, old)
                f.truncate()
            raise


def _write_all(f, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[f.write(view):]
