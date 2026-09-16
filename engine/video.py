"""
Video frame extraction for video captioning.

Both engines caption a video the same way: sample a handful of evenly-spaced
frames and send them as multiple images in a single chat turn. This module
owns the OpenCV side of that — picking frame indices, decoding, and converting
BGR numpy frames to RGB PIL images.

cv2 is imported lazily inside each function so this module (and everything
that imports it) still loads on installs without opencv-python-headless —
the actionable error surfaces only when someone actually captions a video.
"""

import warnings
from pathlib import Path

from PIL import Image

from engine.base import VideoCancelled

# How often the scan loops poll cancel_check. Often enough that a cancel
# feels instant on a long clip, rare enough that the callback is not itself
# the cost of the scan. Named so tests can shorten it — a clip under this
# many frames never polls mid-count, which is fine (that count is instant)
# but makes the behaviour untestable with a small fixture.
CANCEL_POLL_INTERVAL = 64

# Supported video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def is_video_file(path: Path) -> bool:
    """Check if a file path has a supported video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _check_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise VideoCancelled("cancelled during video frame extraction")


def _require_cv2():
    """Import cv2 on first use, with an actionable error when it's missing."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "opencv-python-headless is required for video captioning — "
            "re-run setup"
        ) from e
    return cv2


def _to_pil(cv2, frame) -> Image.Image:
    """Convert a decoded BGR frame to an RGB PIL image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _sample_by_index(
    cv2, cap, indices: list[int], cancel_check=None
) -> tuple[list[Image.Image], bool]:
    """Seek to each index and decode it.

    Returns ``(frames, seek_refused)``. Both ways of coming up short look
    identical in the frame count alone, and they mean opposite things about
    the file, so the caller needs to be told which happened:

    * ``seek_refused=False`` with fewer frames than ``indices``: the seeks
      landed but reads failed. That is how a header that over-reports shows
      up — seeking past the real end of a truncated file succeeds, and only
      the read that follows fails.
    * ``seek_refused=True`` (always with no frames): the container would not
      seek at all. The header may be perfectly truthful; this file simply
      cannot be sampled by index.

    Either way the caller re-samples sequentially.
    """
    frames: list[Image.Image] = []
    for idx in indices:
        _check_cancelled(cancel_check)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            # This container can't seek — reads after a failed seek would
            # just decode consecutive frames from wherever the decoder sits,
            # silently losing temporal coverage. Let the caller fall back to
            # the sequential pass instead.
            return [], True
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(_to_pil(cv2, frame))
    return frames, False


def _midpoint_indices(total: int, num_frames: int) -> list[int]:
    """Evenly-spaced midpoint indices: round((i+0.5)*total/n), deduped.

    Midpoints (rather than i*total/n) hit the middle of each of n equal spans,
    so a short clip's intro and outro don't dominate the sample.
    """
    indices: list[int] = []
    for i in range(num_frames):
        idx = min(total - 1, max(0, round((i + 0.5) * total / num_frames)))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def _count_frames(cv2, video_path: Path, cancel_check=None) -> int:
    """Count decodable frames with a grab-only pass (no decode cost).

    Cheap per frame, but unbounded in their number — a two-hour clip is a
    long time to ignore a cancel, so the loop checks periodically.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()  # a failed open still allocates the capture object
        return 0
    try:
        total = 0
        while cap.grab():
            total += 1
            if total % CANCEL_POLL_INTERVAL == 0:
                _check_cancelled(cancel_check)
        return total
    finally:
        cap.release()


def _sample_sequential(
    cv2, video_path: Path, num_frames: int, cancel_check=None
) -> list[Image.Image]:
    """Two-pass sampler for files whose frame count or seeking is unusable.

    Counts frames with grab() (which skips decoding), then re-reads from the
    start and decodes only at the midpoint indices — so the sample spans the
    whole clip and returns the requested number of frames. The previous
    rolling-halving approach kept only the frames it happened to survive with,
    which both under-delivered and skewed coverage toward the start.

    Memory stays bounded at the sampled frames; passes are grab-only except
    at the indices actually kept.
    """
    total = _count_frames(cv2, video_path, cancel_check)
    if total <= 0:
        return []

    wanted = _midpoint_indices(total, num_frames)
    targets = set(wanted)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()  # a failed open still allocates the capture object
        return []
    try:
        decoded: dict[int, Image.Image] = {}
        idx = 0
        while idx <= wanted[-1]:
            if idx % CANCEL_POLL_INTERVAL == 0:
                _check_cancelled(cancel_check)
            if not cap.grab():
                break
            if idx in targets:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    decoded[idx] = _to_pil(cv2, frame)
            idx += 1
    finally:
        cap.release()

    return [decoded[i] for i in wanted if i in decoded]


def sample_frames(
    video_path: str | Path, num_frames: int = 8, cancel_check=None
) -> list[Image.Image]:
    """
    Decode up to ``num_frames`` evenly-spaced RGB frames from a video.

    Sampling seeks to evenly-spaced indices when the container reports a
    frame count, and re-samples with the sequential pass whenever that
    indexed pass comes up **short**.

    "Short" is measured against the number of distinct indices asked for, not
    against ``num_frames``. That distinction is the whole rule:

    * A genuinely short clip's index list is already clamped to its own
      length — an honest 3-frame file asked for 8 frames wants 3 indices,
      fills all 3, and is not short. It returns what exists and pays for no
      second pass.
    * Fewer decoded frames than indices asked for means frames the header
      promised did not decode: a truncated file, a lying ``CAP_PROP_FRAME_COUNT``,
      or a container that refused to seek. Seeking past the real end of such
      a file *succeeds*, so the miss is silent — skipped reads used to just
      shrink the sample (a header claiming 50 frames over a 32-frame file
      returned 10 frames for ``num_frames=16``) and squeeze its coverage into
      the part of the index space that still existed.
    * A header that **under**-reports is still trusted, and that gap is not
      fixed here. Every index built from a too-small count decodes, nothing
      comes up short, no fallback runs — so the tail of the clip past the
      claimed count is silently never sampled. Only over-reporting (and
      refused seeks) are detected.

    The sequential pass counts frames itself with ``grab()``, so its indices
    span the real clip. Its result wins whenever it is at least as long as
    the seeked one; the seeked frames are kept if it somehow finds fewer, so
    a decodable clip never comes back empty.

    Args:
        video_path: Path to the video file.
        num_frames: How many frames to sample (>= 1).

    Returns:
        PIL images in temporal order. May be shorter than ``num_frames`` for
        very short videos, but never empty.

    Args:
        video_path: the clip to sample.
        num_frames: how many evenly-spaced frames to return (>= 1).
        cancel_check: optional zero-arg predicate polled during the scans.
            Extraction can run for seconds on a long clip with no token loop
            to notice a cancel, so both passes check it.

    Raises:
        ValueError: num_frames < 1.
        VideoCancelled: cancel_check() went true mid-extraction.
        RuntimeError: the file can't be opened, no frame decodes, or cv2 is
            not installed.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    cv2 = _require_cv2()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # Released before raising: a failed open still allocates the capture
        # object, and this exit is outside the try/finally that owns it.
        cap.release()
        raise RuntimeError(f"Could not open video file: {video_path}")

    try:
        # Some webm/VFR files report 0 or -1 — no usable index space.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = _midpoint_indices(total, num_frames) if total > 0 else []
        seeked, seek_refused = (
            _sample_by_index(cv2, cap, wanted, cancel_check)
            if wanted else ([], False)
        )
    finally:
        # Released before any sequential pass, so only one capture is ever
        # open on the file at a time.
        cap.release()

    frames = seeked
    if not wanted or len(seeked) < len(wanted):
        # Either the header gave us nothing to seek by, or it promised more
        # than decoded. _sample_sequential opens its own captures and is the
        # decoder of last resort.
        resampled = _sample_sequential(
            cv2, video_path, num_frames, cancel_check
        )
        if len(resampled) >= len(frames):
            frames = resampled
        if wanted:
            # Only the over-reporting/unseekable cases are worth a word: the
            # plain "header reports 0" VFR path is normal and stays quiet.
            # They are different faults and must not share a sentence — an
            # unseekable container's header is usually telling the truth, so
            # blaming it sends the reader after a file that is fine.
            if seek_refused:
                detail = (
                    f"container refused to seek, so none of the "
                    f"{len(wanted)} sampled indices could be reached"
                )
            else:
                detail = (
                    f"header claims {total} frames but only {len(seeked)} "
                    f"of {len(wanted)} sampled indices decoded"
                )
            warnings.warn(
                f"{video_path.name}: {detail}; re-sampled sequentially "
                f"and got {len(frames)}.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not frames:
        raise RuntimeError(f"Could not decode any frames from: {video_path}")
    return frames


def first_frame(video_path: str | Path) -> Image.Image:
    """
    Decode one representative frame for a browser thumbnail.

    Prefers the frame ~10% in (frame 0 is often black or a fade-in), falling
    back to frame 0, then to the first decodable frame.
    """
    cv2 = _require_cv2()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()  # outside the try/finally below — release it here
        raise RuntimeError(f"Could not open video file: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.10))
            ok, frame = cap.read()
            if ok and frame is not None:
                return _to_pil(cv2, frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # grab() advances even when a frame won't decode, so this scan can
        # step past corrupt leading frames (a plain read() loop cannot)
        while cap.grab():
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                return _to_pil(cv2, frame)
    finally:
        cap.release()

    raise RuntimeError(f"Could not decode any frames from: {video_path}")
