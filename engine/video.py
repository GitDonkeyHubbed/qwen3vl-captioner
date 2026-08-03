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

from pathlib import Path

from PIL import Image

# Supported video file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def is_video_file(path: Path) -> bool:
    """Check if a file path has a supported video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


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


def _sample_by_index(cv2, cap, total: int, num_frames: int) -> list[Image.Image]:
    """Seek to evenly-spaced indices and decode each one."""
    # Midpoint sampling — round((i+0.5)*total/n) hits the middle of each of
    # n equal spans, so intro/outro frames don't dominate short videos.
    indices: list[int] = []
    for i in range(num_frames):
        idx = min(total - 1, max(0, round((i + 0.5) * total / num_frames)))
        if not indices or idx != indices[-1]:
            indices.append(idx)

    frames: list[Image.Image] = []
    for idx in indices:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            # This container can't seek — reads after a failed seek would
            # just decode consecutive frames from wherever the decoder sits,
            # silently losing temporal coverage. Let the caller fall back to
            # the sequential pass instead.
            return []
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(_to_pil(cv2, frame))
    return frames


def _sample_sequential(cv2, cap, num_frames: int) -> list[Image.Image]:
    """Single decode pass for files with an unreliable frame count.

    Bounded rolling sample: keep every stride-th frame; when the buffer is
    full, drop every other kept frame and double the stride. Memory stays at
    <= num_frames decoded frames however long the video is, and the survivors
    remain evenly spaced and in temporal order.
    """
    frames: list[Image.Image] = []
    stride = 1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            if len(frames) == num_frames:
                frames = frames[::2]
                stride *= 2
            # Thinning changed the stride, so this frame may no longer be kept
            if idx % stride == 0:
                frames.append(_to_pil(cv2, frame))
        idx += 1
    return frames


def sample_frames(
    video_path: str | Path, num_frames: int = 8
) -> list[Image.Image]:
    """
    Decode up to ``num_frames`` evenly-spaced RGB frames from a video.

    Args:
        video_path: Path to the video file.
        num_frames: How many frames to sample (>= 1).

    Returns:
        PIL images in temporal order. May be shorter than ``num_frames`` for
        very short videos, but never empty.

    Raises:
        ValueError: num_frames < 1.
        RuntimeError: the file can't be opened, no frame decodes, or cv2 is
            not installed.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    cv2 = _require_cv2()

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            frames = _sample_by_index(cv2, cap, total, num_frames)
        else:
            # Some webm/VFR files report 0 or -1 — fall back to one pass
            frames = _sample_sequential(cv2, cap, num_frames)
    finally:
        cap.release()

    if not frames and total > 0:
        # The header claimed frames but seeking/decoding produced none
        # (lying header, unseekable container). One sequential pass on a
        # fresh capture is the decoder of last resort.
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            try:
                frames = _sample_sequential(cv2, cap, num_frames)
            finally:
                cap.release()

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
